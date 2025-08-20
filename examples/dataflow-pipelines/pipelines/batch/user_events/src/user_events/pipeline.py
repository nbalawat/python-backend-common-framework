"""
User Events Batch Processing Pipeline.

Processes user event data from GCS and writes to BigTable with comprehensive
data validation, error handling, and monitoring.

Example usage:
    python pipeline.py \
        --input_bucket=my-data-bucket \
        --input_prefix=user_events/2024/01/ \
        --project_id=my-project \
        --bigtable_instance=my-instance \
        --bigtable_table=user_events \
        --runner=DirectRunner
"""

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
import argparse
import logging
from typing import Dict, Any
from datetime import datetime, timezone

# Import common utilities
from common.transforms import GCSReader, BigTableWriter, DataValidator, ErrorHandler
from common.config import PipelineConfig, create_dataflow_options


class UserEventTransform(beam.DoFn):
    """
    User event specific data transformations.
    
    Applies business logic transformations specific to user events:
    - Standardizes event types
    - Calculates session metrics
    - Enriches with derived fields
    - Prepares data for BigTable storage
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize transform with configuration.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or {}
        self.event_type_mapping = {
            'pageview': 'page_view',
            'click': 'click', 
            'purchase': 'purchase',
            'signup': 'user_signup',
            'login': 'user_login',
            'logout': 'user_logout',
        }
    
    def process(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform user event data.
        
        Args:
            element: Raw user event record
            
        Yields:
            Transformed record ready for BigTable
        """
        try:
            # Skip error records
            if element.get("_validation_failed"):
                yield element
                return
            
            # Standardize event type
            raw_event_type = element.get('event_type', '').lower()
            element['event_type'] = self.event_type_mapping.get(
                raw_event_type, 
                raw_event_type
            )
            
            # Parse and standardize timestamp
            timestamp = self._parse_timestamp(element.get('timestamp'))
            element['timestamp'] = timestamp.isoformat() if timestamp else None
            element['date'] = timestamp.date().isoformat() if timestamp else None
            element['hour'] = timestamp.hour if timestamp else None
            
            # Extract URL components
            if 'page_url' in element:
                element.update(self._parse_url_components(element['page_url']))
            
            # Calculate session metrics (if session_id available)
            if 'session_id' in element:
                element.update(self._calculate_session_metrics(element))
            
            # Add revenue categorization
            revenue = element.get('revenue', 0)
            element['revenue_category'] = self._categorize_revenue(revenue)
            element['is_monetized'] = revenue > 0
            
            # Prepare BigTable row structure
            row_data = {
                'row_key': self._generate_row_key(element),
                'data': {
                    'event_data': {
                        'event_id': element.get('event_id'),
                        'event_type': element.get('event_type'),
                        'timestamp': element.get('timestamp'),
                        'date': element.get('date'),
                        'hour': str(element.get('hour', 0)),
                        'user_id': element.get('user_id'),
                        'session_id': element.get('session_id'),
                    },
                    'page_data': {
                        'page_url': element.get('page_url'),
                        'page_path': element.get('page_path'),
                        'page_title': element.get('page_title', ''),
                        'referrer': element.get('referrer', ''),
                    },
                    'user_data': {
                        'user_agent': element.get('user_agent'),
                        'ip_address': element.get('ip_address'),
                        'country': element.get('country'),
                        'device_type': element.get('device_type'),
                    },
                    'business_data': {
                        'revenue': str(element.get('revenue', 0)),
                        'revenue_category': element.get('revenue_category'),
                        'is_monetized': str(element.get('is_monetized', False)),
                        'conversion_value': str(element.get('conversion_value', 0)),
                    },
                    'session_data': {
                        'session_duration': str(element.get('session_duration', 0)),
                        'page_views': str(element.get('page_views', 1)),
                        'is_bounce': str(element.get('is_bounce', False)),
                    },
                    'metadata': {
                        'source_file': element.get('_source_file'),
                        'processed_at': datetime.now(timezone.utc).isoformat(),
                        'pipeline_version': '1.0.0',
                    }
                }
            }
            
            yield row_data
            
        except Exception as e:
            logging.error(f"Error transforming user event: {e}")
            # Add error information and pass through
            element['_transform_error'] = True
            element['_transform_error_message'] = str(e)
            yield element
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp from various formats."""
        if not timestamp_str:
            return None
        
        # Try common timestamp formats
        formats = [
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d',
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(timestamp_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
        
        # Try parsing as Unix timestamp
        try:
            ts = float(timestamp_str)
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except (ValueError, OSError):
            pass
        
        logging.warning(f"Could not parse timestamp: {timestamp_str}")
        return None
    
    def _parse_url_components(self, url: str) -> Dict[str, str]:
        """Extract components from URL."""
        try:
            from urllib.parse import urlparse, parse_qs
            
            parsed = urlparse(url)
            return {
                'page_path': parsed.path or '/',
                'page_query': parsed.query or '',
                'page_fragment': parsed.fragment or '',
                'page_domain': parsed.netloc or '',
            }
        except Exception as e:
            logging.warning(f"Error parsing URL {url}: {e}")
            return {'page_path': url}
    
    def _calculate_session_metrics(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate session-based metrics."""
        # Note: In a real implementation, you might need to aggregate
        # across multiple events in a session. This is a simplified version.
        return {
            'session_duration': element.get('session_duration', 0),
            'page_views': 1,  # This event counts as 1 page view
            'is_bounce': element.get('is_bounce', False),
        }
    
    def _categorize_revenue(self, revenue: float) -> str:
        """Categorize revenue amount."""
        if revenue == 0:
            return 'no_revenue'
        elif revenue < 10:
            return 'low_value'
        elif revenue < 100:
            return 'medium_value'
        else:
            return 'high_value'
    
    def _generate_row_key(self, element: Dict[str, Any]) -> str:
        """Generate BigTable row key."""
        user_id = element.get('user_id', 'unknown')
        timestamp = element.get('timestamp', datetime.now(timezone.utc).isoformat())
        event_id = element.get('event_id', '')
        
        # Format: user_id#timestamp#event_id
        # This ensures good distribution and enables efficient range scans
        return f"{user_id}#{timestamp}#{event_id}"


def create_user_events_pipeline(config) -> beam.Pipeline:
    """
    Create the user events processing pipeline.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Configured Apache Beam pipeline
    """
    
    # Create pipeline options
    pipeline_options = PipelineOptions()
    setup_options = pipeline_options.view_as(SetupOptions)
    setup_options.setup_file = './setup.py'
    
    # Create pipeline
    pipeline = beam.Pipeline(options=pipeline_options)
    
    # Define data validation rules
    def validate_user_id(user_id):
        return isinstance(user_id, str) and len(user_id) > 0
    
    def validate_timestamp(timestamp):
        return timestamp is not None and timestamp != ''
    
    def validate_event_type(event_type):
        valid_types = {'page_view', 'click', 'purchase', 'signup', 'login', 'logout', 'pageview'}
        return event_type in valid_types
    
    validation_rules = {
        'user_id': validate_user_id,
        'timestamp': validate_timestamp,
        'event_type': validate_event_type,
    }
    
    # Build pipeline
    raw_events = (
        pipeline
        | 'Create File Patterns' >> beam.Create([config.input_prefix])
        | 'Read from GCS' >> beam.ParDo(
            GCSReader(
                bucket_name=config.input_bucket,
                file_format='json',
                max_file_size_mb=100
            )
        )
    )
    
    # Validation and error handling
    validated_events = (
        raw_events
        | 'Validate Events' >> beam.ParDo(
            DataValidator(
                required_fields=['user_id', 'event_type', 'timestamp'],
                validation_rules=validation_rules,
                emit_invalid_records=True
            )
        )
    )
    
    # Transform events
    transformed_events = (
        validated_events
        | 'Transform Events' >> beam.ParDo(UserEventTransform(config=config.__dict__))
    )
    
    # Route to success/error outputs
    routed_events = (
        transformed_events
        | 'Route Events' >> beam.ParDo(ErrorHandler()).with_outputs('success', 'errors')
    )
    
    # Write successful records to BigTable
    bigtable_results = (
        routed_events.success
        | 'Write to BigTable' >> beam.ParDo(
            BigTableWriter(
                project_id=config.project_id,
                instance_id=config.bigtable_instance,
                table_id=config.bigtable_table,
                column_family=config.bigtable_column_family,
                batch_size=config.batch_size,
                row_key_extractor=lambda x: x.get('row_key')
            )
        )
    )
    
    # Handle errors (log for now, could write to dead letter queue)
    error_handling = (
        routed_events.errors
        | 'Log Errors' >> beam.Map(lambda x: logging.error(f"Error record: {x}"))
    )
    
    # Optional: Write to BigQuery for analytics
    if hasattr(config, 'bigquery_dataset') and config.bigquery_dataset:
        bq_records = (
            routed_events.success
            | 'Prepare for BigQuery' >> beam.Map(lambda x: {
                'user_id': x['data']['event_data']['user_id'],
                'event_type': x['data']['event_data']['event_type'],
                'timestamp': x['data']['event_data']['timestamp'],
                'revenue': float(x['data']['business_data']['revenue']),
                'country': x['data']['user_data']['country'],
                'processed_at': x['data']['metadata']['processed_at'],
            })
            | 'Write to BigQuery' >> beam.io.WriteToBigQuery(
                table=f"{config.project_id}:{config.bigquery_dataset}.user_events",
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED
            )
        )
    
    return pipeline


def run_pipeline(argv=None):
    """Run the user events batch processing pipeline."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='User Events Batch Processing Pipeline')
    parser.add_argument('--input_bucket', required=True, help='GCS input bucket name')
    parser.add_argument('--input_prefix', required=True, help='GCS input file prefix')
    parser.add_argument('--project_id', required=True, help='GCP project ID')
    parser.add_argument('--bigtable_instance', required=True, help='BigTable instance ID')
    parser.add_argument('--bigtable_table', required=True, help='BigTable table ID')
    parser.add_argument('--bigtable_column_family', default='cf', help='BigTable column family')
    parser.add_argument('--batch_size', type=int, default=100, help='BigTable write batch size')
    parser.add_argument('--bigquery_dataset', help='Optional BigQuery dataset for analytics')
    parser.add_argument('--config_file', help='Path to configuration file')
    
    known_args, pipeline_args = parser.parse_known_args(argv)
    
    # Create configuration
    if known_args.config_file:
        config = PipelineConfig.load_from_file(known_args.config_file)
    else:
        config = PipelineConfig.create_batch_config(
            pipeline_name="user-events-batch",
            project_id=known_args.project_id,
            input_bucket=known_args.input_bucket,
            bigtable_instance=known_args.bigtable_instance,
            bigtable_table=known_args.bigtable_table,
            input_prefix=known_args.input_prefix,
            bigtable_column_family=known_args.bigtable_column_family,
            batch_size=known_args.batch_size,
            bigquery_dataset=known_args.bigquery_dataset,
        )
    
    # Set up logging
    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"Starting user events pipeline with config: {config}")
    
    # Create and run pipeline
    pipeline = create_user_events_pipeline(config)
    result = pipeline.run()
    
    # Wait for completion if running locally
    if hasattr(result, 'wait_until_finish'):
        result.wait_until_finish()
    
    logging.info("User events pipeline completed successfully")


if __name__ == '__main__':
    run_pipeline()