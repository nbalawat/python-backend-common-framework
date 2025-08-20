"""
Real-time Events Streaming Pipeline.

Processes real-time event streams from Pub/Sub with low-latency BigTable writes,
comprehensive windowing, and advanced error handling with dead letter queues.

Example usage:
    python pipeline.py \
        --subscription=projects/my-project/subscriptions/events-sub \
        --project_id=my-project \
        --bigtable_instance=my-instance \
        --bigtable_table=real_time_events \
        --dead_letter_topic=projects/my-project/topics/dead-letters \
        --runner=DataflowRunner \
        --streaming
"""

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
from apache_beam.transforms import window
import argparse
import logging
from typing import Dict, Any, Iterator
from datetime import datetime, timezone

# Import common utilities
from common.streaming import (
    PubSubReader, StreamingBigTableWriter, MessageParser, DeadLetterHandler,
    create_pubsub_source, create_pubsub_sink
)
from common.windowing import create_fixed_windows, TriggerStrategies
from common.transforms import DataValidator, ErrorHandler
from common.config import PipelineConfig, create_dataflow_options


class RealTimeEventProcessor(beam.DoFn):
    """
    Real-time event processing with business logic transformations.
    
    Processes events with:
    - Event enrichment and standardization
    - Real-time feature calculation
    - Session tracking and metrics
    - Alert generation for critical events
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize processor with configuration."""
        self.config = config or {}
        self.alert_thresholds = {
            'high_value_transaction': 1000.0,
            'suspicious_activity': True,
            'error_rate_threshold': 0.1,
        }
    
    def process(self, element: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """
        Process real-time events with enrichment and alert generation.
        
        Args:
            element: Event from Pub/Sub
            
        Yields:
            Processed event records and potential alerts
        """
        try:
            # Skip error records from upstream processing
            if element.get("_pubsub_parse_error") or element.get("_validation_failed"):
                yield element
                return
            
            # Extract event timestamp (use Pub/Sub timestamp as fallback)
            event_timestamp = self._extract_event_timestamp(element)
            element['event_timestamp'] = event_timestamp.isoformat()
            element['processing_timestamp'] = datetime.now(timezone.utc).isoformat()
            
            # Calculate processing latency
            processing_latency = self._calculate_processing_latency(element)
            element['processing_latency_ms'] = processing_latency
            
            # Enrich event with derived fields
            enriched_event = self._enrich_event(element)
            
            # Generate alerts for critical events
            alerts = list(self._generate_alerts(enriched_event))
            for alert in alerts:
                yield alert
            
            # Prepare for BigTable storage
            bigtable_record = self._prepare_bigtable_record(enriched_event)
            yield bigtable_record
            
        except Exception as e:
            logging.error(f"Error processing real-time event: {e}")
            element['_processing_error'] = True
            element['_processing_error_message'] = str(e)
            element['_error_timestamp'] = datetime.now(timezone.utc).isoformat()
            yield element
    
    def _extract_event_timestamp(self, element: Dict[str, Any]) -> datetime:
        """Extract event timestamp with fallbacks."""
        # Try event_timestamp field first
        if 'event_timestamp' in element:
            try:
                return datetime.fromisoformat(element['event_timestamp'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass
        
        # Try timestamp field
        if 'timestamp' in element:
            try:
                return datetime.fromisoformat(element['timestamp'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass
        
        # Use Pub/Sub publish time
        if '_pubsub_publish_time' in element:
            try:
                return datetime.fromisoformat(element['_pubsub_publish_time'])
            except (ValueError, AttributeError):
                pass
        
        # Fallback to current time
        return datetime.now(timezone.utc)
    
    def _calculate_processing_latency(self, element: Dict[str, Any]) -> float:
        """Calculate processing latency in milliseconds."""
        try:
            event_time_str = element.get('_pubsub_publish_time') or element.get('event_timestamp')
            if event_time_str:
                event_time = datetime.fromisoformat(event_time_str.replace('Z', '+00:00'))
                latency = (datetime.now(timezone.utc) - event_time).total_seconds() * 1000
                return max(0, latency)  # Ensure non-negative
        except Exception:
            pass
        return 0.0
    
    def _enrich_event(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich event with derived fields and business logic."""
        # Create enriched copy
        enriched = element.copy()
        
        # Add derived fields based on event type
        event_type = element.get('event_type', '').lower()
        
        if event_type in ['purchase', 'transaction']:
            enriched.update(self._enrich_transaction_event(element))
        elif event_type in ['page_view', 'click']:
            enriched.update(self._enrich_web_event(element))
        elif event_type in ['login', 'logout']:
            enriched.update(self._enrich_auth_event(element))
        
        # Add common enrichments
        enriched['event_category'] = self._categorize_event(element)
        enriched['risk_score'] = self._calculate_risk_score(element)
        enriched['is_critical'] = self._is_critical_event(element)
        
        # Add session context if available
        if 'session_id' in element:
            enriched.update(self._enrich_session_context(element))
        
        return enriched
    
    def _enrich_transaction_event(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich transaction-specific events."""
        amount = element.get('amount', 0)
        currency = element.get('currency', 'USD')
        
        return {
            'amount_usd': self._convert_to_usd(amount, currency),
            'amount_category': self._categorize_amount(amount),
            'is_high_value': amount > self.alert_thresholds['high_value_transaction'],
            'payment_risk_level': self._assess_payment_risk(element),
        }
    
    def _enrich_web_event(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich web interaction events."""
        return {
            'page_category': self._categorize_page(element.get('page_url', '')),
            'device_category': self._categorize_device(element.get('user_agent', '')),
            'is_mobile': 'mobile' in element.get('user_agent', '').lower(),
            'geographic_region': self._get_geographic_region(element.get('country', '')),
        }
    
    def _enrich_auth_event(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich authentication events."""
        return {
            'auth_method': element.get('auth_method', 'password'),
            'is_suspicious_location': self._check_suspicious_location(element),
            'login_frequency_score': self._calculate_login_frequency(element),
        }
    
    def _enrich_session_context(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """Add session-based context."""
        # In a real implementation, this might query a session store
        session_id = element.get('session_id')
        return {
            'session_duration_estimate': 0,  # Would be calculated from session store
            'session_event_count': 1,  # Would be calculated from session store
            'is_new_session': False,  # Would be determined from session store
        }
    
    def _categorize_event(self, element: Dict[str, Any]) -> str:
        """Categorize event for analytics."""
        event_type = element.get('event_type', '').lower()
        
        if event_type in ['purchase', 'transaction', 'payment']:
            return 'commerce'
        elif event_type in ['page_view', 'click', 'navigation']:
            return 'engagement'
        elif event_type in ['login', 'logout', 'signup']:
            return 'authentication'
        elif event_type in ['error', 'exception', 'failure']:
            return 'system'
        else:
            return 'other'
    
    def _calculate_risk_score(self, element: Dict[str, Any]) -> float:
        """Calculate risk score for the event."""
        risk_score = 0.0
        
        # High amounts increase risk
        amount = element.get('amount', 0)
        if amount > 500:
            risk_score += 0.3
        elif amount > 100:
            risk_score += 0.1
        
        # Unusual locations increase risk
        if self._check_suspicious_location(element):
            risk_score += 0.4
        
        # Error events increase risk
        if element.get('event_type') == 'error':
            risk_score += 0.2
        
        # Failed authentication increases risk
        if element.get('event_type') == 'login' and element.get('status') == 'failed':
            risk_score += 0.5
        
        return min(1.0, risk_score)  # Cap at 1.0
    
    def _is_critical_event(self, element: Dict[str, Any]) -> bool:
        """Determine if event is critical and needs immediate attention."""
        # High-value transactions
        if element.get('amount', 0) > self.alert_thresholds['high_value_transaction']:
            return True
        
        # High risk scores
        if element.get('risk_score', 0) > 0.7:
            return True
        
        # System errors
        if element.get('event_type') == 'error' and element.get('severity') == 'critical':
            return True
        
        # Failed authentication attempts
        if (element.get('event_type') == 'login' and 
            element.get('status') == 'failed' and 
            element.get('attempt_count', 1) > 3):
            return True
        
        return False
    
    def _generate_alerts(self, element: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Generate alerts for critical events."""
        if not element.get('is_critical'):
            return
        
        alert_types = []
        
        # High-value transaction alert
        if element.get('is_high_value'):
            alert_types.append('high_value_transaction')
        
        # High risk score alert
        if element.get('risk_score', 0) > 0.7:
            alert_types.append('high_risk_event')
        
        # Authentication failure alert
        if (element.get('event_type') == 'login' and 
            element.get('status') == 'failed'):
            alert_types.append('authentication_failure')
        
        # Generate alert records
        for alert_type in alert_types:
            yield {
                'alert_id': f"{alert_type}_{element.get('user_id', 'unknown')}_{int(datetime.now().timestamp())}",
                'alert_type': alert_type,
                'severity': self._determine_alert_severity(alert_type, element),
                'user_id': element.get('user_id'),
                'event_id': element.get('event_id'),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'details': {
                    'original_event': {
                        'event_type': element.get('event_type'),
                        'amount': element.get('amount'),
                        'risk_score': element.get('risk_score'),
                        'country': element.get('country'),
                    }
                },
                '_is_alert': True,
            }
    
    def _determine_alert_severity(self, alert_type: str, element: Dict[str, Any]) -> str:
        """Determine alert severity level."""
        if alert_type == 'high_value_transaction':
            amount = element.get('amount', 0)
            if amount > 10000:
                return 'critical'
            elif amount > 5000:
                return 'high'
            else:
                return 'medium'
        elif alert_type == 'high_risk_event':
            risk_score = element.get('risk_score', 0)
            if risk_score > 0.9:
                return 'critical'
            elif risk_score > 0.8:
                return 'high'
            else:
                return 'medium'
        elif alert_type == 'authentication_failure':
            attempt_count = element.get('attempt_count', 1)
            if attempt_count > 5:
                return 'high'
            else:
                return 'medium'
        return 'low'
    
    def _prepare_bigtable_record(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare record for BigTable storage."""
        # Generate row key with good distribution
        user_id = element.get('user_id', 'unknown')
        timestamp = element.get('event_timestamp', datetime.now(timezone.utc).isoformat())
        event_id = element.get('event_id', '')
        
        row_key = f"{user_id}#{timestamp}#{event_id}"
        
        return {
            'row_key': row_key,
            'data': {
                'event': {
                    'event_id': element.get('event_id', ''),
                    'event_type': element.get('event_type', ''),
                    'event_timestamp': element.get('event_timestamp', ''),
                    'event_category': element.get('event_category', ''),
                },
                'user': {
                    'user_id': element.get('user_id', ''),
                    'session_id': element.get('session_id', ''),
                    'country': element.get('country', ''),
                    'device_type': element.get('device_type', ''),
                },
                'transaction': {
                    'amount': str(element.get('amount', 0)),
                    'amount_usd': str(element.get('amount_usd', 0)),
                    'currency': element.get('currency', ''),
                    'amount_category': element.get('amount_category', ''),
                },
                'risk': {
                    'risk_score': str(element.get('risk_score', 0)),
                    'is_critical': str(element.get('is_critical', False)),
                    'is_high_value': str(element.get('is_high_value', False)),
                },
                'processing': {
                    'processing_timestamp': element.get('processing_timestamp', ''),
                    'processing_latency_ms': str(element.get('processing_latency_ms', 0)),
                    'pipeline_version': '1.0.0',
                },
                'pubsub': {
                    'message_id': element.get('_pubsub_message_id', ''),
                    'publish_time': element.get('_pubsub_publish_time', ''),
                    'received_time': element.get('_pubsub_received_time', ''),
                }
            }
        }
    
    # Helper methods (simplified implementations)
    def _convert_to_usd(self, amount: float, currency: str) -> float:
        """Convert amount to USD (simplified - would use real exchange rates)."""
        rates = {'EUR': 1.1, 'GBP': 1.3, 'CAD': 0.8, 'USD': 1.0}
        return amount * rates.get(currency, 1.0)
    
    def _categorize_amount(self, amount: float) -> str:
        """Categorize transaction amount."""
        if amount < 10:
            return 'micro'
        elif amount < 100:
            return 'small'
        elif amount < 1000:
            return 'medium'
        else:
            return 'large'
    
    def _assess_payment_risk(self, element: Dict[str, Any]) -> str:
        """Assess payment risk level."""
        # Simplified risk assessment
        amount = element.get('amount', 0)
        country = element.get('country', 'US')
        
        if amount > 1000 and country not in ['US', 'CA', 'GB', 'DE', 'FR']:
            return 'high'
        elif amount > 500:
            return 'medium'
        else:
            return 'low'
    
    def _categorize_page(self, url: str) -> str:
        """Categorize page type from URL."""
        if '/checkout' in url or '/purchase' in url:
            return 'commerce'
        elif '/product' in url:
            return 'catalog'
        elif '/profile' in url or '/account' in url:
            return 'account'
        else:
            return 'content'
    
    def _categorize_device(self, user_agent: str) -> str:
        """Categorize device from user agent."""
        ua_lower = user_agent.lower()
        if 'mobile' in ua_lower or 'android' in ua_lower or 'iphone' in ua_lower:
            return 'mobile'
        elif 'tablet' in ua_lower or 'ipad' in ua_lower:
            return 'tablet'
        else:
            return 'desktop'
    
    def _get_geographic_region(self, country: str) -> str:
        """Get geographic region from country."""
        regions = {
            'US': 'North America', 'CA': 'North America', 'MX': 'North America',
            'GB': 'Europe', 'DE': 'Europe', 'FR': 'Europe', 'IT': 'Europe',
            'JP': 'Asia Pacific', 'CN': 'Asia Pacific', 'AU': 'Asia Pacific',
        }
        return regions.get(country, 'Other')
    
    def _check_suspicious_location(self, element: Dict[str, Any]) -> bool:
        """Check if location seems suspicious (simplified)."""
        # In real implementation, would check against user's typical locations
        return False
    
    def _calculate_login_frequency(self, element: Dict[str, Any]) -> float:
        """Calculate login frequency score (simplified)."""
        # In real implementation, would query historical data
        return 0.5


def create_real_time_pipeline(config) -> beam.Pipeline:
    """
    Create the real-time events processing pipeline.
    
    Args:
        config: Streaming pipeline configuration
        
    Returns:
        Configured Apache Beam streaming pipeline
    """
    
    # Create pipeline options
    pipeline_options = PipelineOptions()
    setup_options = pipeline_options.view_as(SetupOptions)
    setup_options.setup_file = './setup.py'
    
    # Create streaming pipeline
    pipeline = beam.Pipeline(options=pipeline_options)
    
    # Read from Pub/Sub
    raw_events = (
        pipeline
        | 'Read from Pub/Sub' >> create_pubsub_source(
            subscription=config.subscription,
            with_attributes=True,
            timestamp_attribute='event_timestamp'
        )
    )
    
    # Parse and validate messages
    parsed_events = (
        raw_events
        | 'Parse Messages' >> beam.ParDo(
            MessageParser(
                field_mappings={'ts': 'timestamp', 'uid': 'user_id'},
                data_transformations={
                    'amount': lambda x: float(x) if x else 0.0,
                    'timestamp': lambda x: x.replace('Z', '+00:00') if isinstance(x, str) else x,
                }
            )
        )
    )
    
    # Validate events
    validated_events = (
        parsed_events
        | 'Validate Events' >> beam.ParDo(
            DataValidator(
                required_fields=['user_id', 'event_type'],
                validation_rules={
                    'user_id': lambda x: isinstance(x, str) and len(x) > 0,
                    'event_type': lambda x: x in ['page_view', 'click', 'purchase', 'login', 'logout', 'error'],
                    'amount': lambda x: x >= 0 if x is not None else True,
                },
                emit_invalid_records=True
            )
        )
    )
    
    # Process events with business logic
    processed_events = (
        validated_events
        | 'Process Events' >> beam.ParDo(RealTimeEventProcessor(config=config.__dict__))
    )
    
    # Separate alerts from regular events
    alerts = (
        processed_events
        | 'Filter Alerts' >> beam.Filter(lambda x: x.get('_is_alert', False))
    )
    
    # Filter non-alert events for BigTable
    bigtable_events = (
        processed_events
        | 'Filter Events' >> beam.Filter(lambda x: not x.get('_is_alert', False))
    )
    
    # Route to success/error outputs
    routed_events = (
        bigtable_events
        | 'Route Events' >> beam.ParDo(ErrorHandler()).with_outputs('success', 'errors')
    )
    
    # Write successful events to BigTable
    bigtable_writes = (
        routed_events.success
        | 'Write to BigTable' >> beam.ParDo(
            StreamingBigTableWriter(
                project_id=config.project_id,
                instance_id=config.bigtable_instance,
                table_id=config.bigtable_table,
                column_family=config.bigtable_column_family,
                batch_size=config.bigtable_batch_size,
                flush_interval_seconds=config.bigtable_flush_interval_seconds,
                row_key_template='{user_id}#{timestamp}#{event_id}',
                timestamp_field='event_timestamp'
            )
        )
    )
    
    # Handle errors with dead letter queue
    if config.dead_letter_topic:
        error_handling = (
            routed_events.errors
            | 'Handle Dead Letters' >> beam.ParDo(
                DeadLetterHandler(
                    project_id=config.project_id,
                    dead_letter_topic=config.dead_letter_topic,
                    max_retry_attempts=3,
                    add_error_metadata=True
                )
            )
        )
    
    # Publish alerts to separate topic (if configured)
    if hasattr(config, 'alert_topic') and config.alert_topic:
        alert_publishing = (
            alerts
            | 'Format Alert Messages' >> beam.Map(
                lambda x: json.dumps(x).encode('utf-8')
            )
            | 'Publish Alerts' >> create_pubsub_sink(
                topic=config.alert_topic,
                with_attributes=True
            )
        )
    
    return pipeline


def run_pipeline(argv=None):
    """Run the real-time events streaming pipeline."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Real-time Events Streaming Pipeline')
    parser.add_argument('--subscription', required=True, help='Pub/Sub subscription path')
    parser.add_argument('--project_id', required=True, help='GCP project ID')
    parser.add_argument('--bigtable_instance', required=True, help='BigTable instance ID')
    parser.add_argument('--bigtable_table', required=True, help='BigTable table ID')
    parser.add_argument('--bigtable_column_family', default='cf', help='BigTable column family')
    parser.add_argument('--bigtable_batch_size', type=int, default=100, help='BigTable batch size')
    parser.add_argument('--bigtable_flush_interval_seconds', type=int, default=10, help='Flush interval')
    parser.add_argument('--dead_letter_topic', help='Dead letter topic path')
    parser.add_argument('--alert_topic', help='Alert topic path')
    parser.add_argument('--window_size_seconds', type=int, default=300, help='Window size in seconds')
    parser.add_argument('--config_file', help='Path to configuration file')
    
    known_args, pipeline_args = parser.parse_known_args(argv)
    
    # Create configuration
    if known_args.config_file:
        config = PipelineConfig.load_from_file(known_args.config_file)
    else:
        config = PipelineConfig.create_streaming_config(
            pipeline_name="real-time-events",
            project_id=known_args.project_id,
            subscription=known_args.subscription,
            bigtable_instance=known_args.bigtable_instance,
            bigtable_table=known_args.bigtable_table,
            bigtable_column_family=known_args.bigtable_column_family,
            bigtable_batch_size=known_args.bigtable_batch_size,
            bigtable_flush_interval_seconds=known_args.bigtable_flush_interval_seconds,
            dead_letter_topic=known_args.dead_letter_topic,
            window_size_seconds=known_args.window_size_seconds,
        )
    
    # Add alert topic if specified
    if known_args.alert_topic:
        config.alert_topic = known_args.alert_topic
    
    # Set up logging
    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"Starting real-time events pipeline with config: {config}")
    
    # Create and run pipeline
    pipeline = create_real_time_pipeline(config)
    result = pipeline.run()
    
    # For streaming pipelines, this will run indefinitely
    logging.info("Real-time events pipeline started successfully")
    
    return result


if __name__ == '__main__':
    run_pipeline()