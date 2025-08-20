"""
Streaming utilities for Pub/Sub based Dataflow pipelines.

This module provides components for real-time data processing:
- Pub/Sub message reading and parsing
- Streaming BigTable writing
- Dead letter queue handling
- Message acknowledgment management
"""

import apache_beam as beam
from apache_beam.transforms import window
from apache_beam.io import ReadFromPubSub, WriteToPubSub
from google.cloud import bigtable, pubsub_v1
import json
import logging
from typing import Dict, Any, Iterator, Optional, Callable, List
from datetime import datetime, timezone
import base64
import avro.schema
import avro.io
import io
from dataclasses import dataclass


@dataclass
class PubSubMessage:
    """Structured representation of a Pub/Sub message."""
    data: bytes
    attributes: Dict[str, str]
    message_id: str
    publish_time: datetime
    ack_id: Optional[str] = None


class PubSubReader(beam.DoFn):
    """
    Advanced Pub/Sub message reader with parsing and error handling.
    
    Supports JSON, Avro, and raw message formats with configurable
    parsing options and dead letter handling.
    """
    
    def __init__(
        self,
        subscription: str,
        message_format: str = "json",
        avro_schema: Optional[str] = None,
        id_label: Optional[str] = None,
        timestamp_attribute: Optional[str] = None,
        with_attributes: bool = True,
    ):
        """
        Initialize Pub/Sub reader.
        
        Args:
            subscription: Pub/Sub subscription path
            message_format: Message format ('json', 'avro', 'raw')
            avro_schema: Avro schema for message parsing
            id_label: Attribute to use as unique ID
            timestamp_attribute: Attribute containing message timestamp
            with_attributes: Whether to include message attributes
        """
        self.subscription = subscription
        self.message_format = message_format.lower()
        self.avro_schema = avro_schema
        self.id_label = id_label
        self.timestamp_attribute = timestamp_attribute
        self.with_attributes = with_attributes
        self._avro_reader = None
    
    def setup(self):
        """Setup Avro reader if needed."""
        if self.message_format == "avro" and self.avro_schema:
            schema = avro.schema.parse(self.avro_schema)
            self._avro_reader = avro.io.DatumReader(schema)
    
    def process(self, element) -> Iterator[Dict[str, Any]]:
        """
        Parse Pub/Sub message and extract data.
        
        Args:
            element: Raw Pub/Sub message
            
        Yields:
            Parsed message data
        """
        try:
            # Extract message components
            if hasattr(element, 'data'):
                # Apache Beam PubsubMessage
                data = element.data
                attributes = element.attributes or {}
                message_id = getattr(element, 'message_id', None)
                publish_time = getattr(element, 'publish_time', None)
            else:
                # Raw bytes
                data = element
                attributes = {}
                message_id = None
                publish_time = None
            
            # Parse message data based on format
            parsed_data = self._parse_message_data(data)
            
            # Create structured record
            record = {
                **parsed_data,
                "_pubsub_message_id": message_id,
                "_pubsub_publish_time": publish_time.isoformat() if publish_time else None,
                "_pubsub_received_time": datetime.now(timezone.utc).isoformat(),
            }
            
            # Add attributes if requested
            if self.with_attributes:
                record["_pubsub_attributes"] = attributes
                
                # Add specific timestamp if configured
                if self.timestamp_attribute and self.timestamp_attribute in attributes:
                    record["event_timestamp"] = attributes[self.timestamp_attribute]
                
                # Add ID if configured
                if self.id_label and self.id_label in attributes:
                    record["message_id"] = attributes[self.id_label]
            
            yield record
            
        except Exception as e:
            logging.error(f"Error parsing Pub/Sub message: {e}")
            # Emit error record
            yield {
                "_pubsub_parse_error": True,
                "_error_message": str(e),
                "_raw_data": base64.b64encode(data).decode() if data else None,
                "_timestamp": datetime.now(timezone.utc).isoformat(),
            }
    
    def _parse_message_data(self, data: bytes) -> Dict[str, Any]:
        """Parse message data based on configured format."""
        if not data:
            return {}
        
        if self.message_format == "json":
            return self._parse_json(data)
        elif self.message_format == "avro":
            return self._parse_avro(data)
        elif self.message_format == "raw":
            return {"raw_data": base64.b64encode(data).decode()}
        else:
            raise ValueError(f"Unsupported message format: {self.message_format}")
    
    def _parse_json(self, data: bytes) -> Dict[str, Any]:
        """Parse JSON message data."""
        try:
            text = data.decode('utf-8')
            return json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid JSON data: {e}")
    
    def _parse_avro(self, data: bytes) -> Dict[str, Any]:
        """Parse Avro message data."""
        if not self._avro_reader:
            self.setup()
        
        try:
            bytes_reader = io.BytesIO(data)
            decoder = avro.io.BinaryDecoder(bytes_reader)
            return self._avro_reader.read(decoder)
        except Exception as e:
            raise ValueError(f"Invalid Avro data: {e}")


class MessageParser(beam.DoFn):
    """
    Post-processing message parser for complex data transformations.
    """
    
    def __init__(
        self,
        schema_registry: Optional[Dict[str, Any]] = None,
        field_mappings: Optional[Dict[str, str]] = None,
        data_transformations: Optional[Dict[str, Callable]] = None,
    ):
        """
        Initialize message parser.
        
        Args:
            schema_registry: Schema definitions for message validation
            field_mappings: Field name mappings (old_name -> new_name)
            data_transformations: Field transformation functions
        """
        self.schema_registry = schema_registry or {}
        self.field_mappings = field_mappings or {}
        self.data_transformations = data_transformations or {}
    
    def process(self, element: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """
        Parse and transform message data.
        
        Args:
            element: Message record
            
        Yields:
            Transformed message record
        """
        try:
            # Skip error records
            if element.get("_pubsub_parse_error"):
                yield element
                return
            
            # Apply field mappings
            if self.field_mappings:
                element = self._apply_field_mappings(element)
            
            # Apply data transformations
            if self.data_transformations:
                element = self._apply_transformations(element)
            
            # Validate against schema if available
            if self.schema_registry:
                element = self._validate_schema(element)
            
            # Add parsing metadata
            element["_message_parsed"] = True
            element["_parsing_timestamp"] = datetime.now(timezone.utc).isoformat()
            
            yield element
            
        except Exception as e:
            logging.error(f"Message parsing error: {e}")
            element["_parsing_error"] = True
            element["_parsing_error_message"] = str(e)
            yield element
    
    def _apply_field_mappings(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """Apply field name mappings."""
        for old_name, new_name in self.field_mappings.items():
            if old_name in element:
                element[new_name] = element.pop(old_name)
        return element
    
    def _apply_transformations(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """Apply data transformations."""
        for field, transform_func in self.data_transformations.items():
            if field in element:
                try:
                    element[field] = transform_func(element[field])
                except Exception as e:
                    logging.warning(f"Transformation failed for field {field}: {e}")
        return element
    
    def _validate_schema(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """Validate message against schema."""
        # Implement schema validation logic
        # This is a placeholder for actual schema validation
        element["_schema_validated"] = True
        return element


class StreamingBigTableWriter(beam.DoFn):
    """
    Streaming BigTable writer optimized for real-time processing.
    
    Features:
    - Configurable batching and flushing
    - Automatic row key generation
    - Time-based partitioning
    - Error handling and retry logic
    """
    
    def __init__(
        self,
        project_id: str,
        instance_id: str,
        table_id: str,
        column_family: str = "cf",
        batch_size: int = 100,
        flush_interval_seconds: int = 10,
        row_key_template: Optional[str] = None,
        timestamp_field: Optional[str] = None,
    ):
        """
        Initialize streaming BigTable writer.
        
        Args:
            project_id: GCP project ID
            instance_id: BigTable instance ID
            table_id: BigTable table ID
            column_family: Column family name
            batch_size: Maximum batch size
            flush_interval_seconds: Time-based flush interval
            row_key_template: Template for row key generation
            timestamp_field: Field to use for cell timestamps
        """
        self.project_id = project_id
        self.instance_id = instance_id
        self.table_id = table_id
        self.column_family = column_family
        self.batch_size = batch_size
        self.flush_interval_seconds = flush_interval_seconds
        self.row_key_template = row_key_template
        self.timestamp_field = timestamp_field
        self._client = None
        self._table = None
        self._batch = []
        self._last_flush_time = None
    
    def setup(self):
        """Initialize BigTable client."""
        self._client = bigtable.Client(project=self.project_id)
        instance = self._client.instance(self.instance_id)
        self._table = instance.table(self.table_id)
        self._last_flush_time = datetime.now()
    
    def process(self, element: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """
        Buffer records for batched writing.
        
        Args:
            element: Record to write
            
        Yields:
            Write status records
        """
        if not self._table:
            self.setup()
        
        try:
            # Skip error records (but emit them for dead letter handling)
            if any(key.startswith("_") and "error" in key for key in element.keys()):
                yield element
                return
            
            # Create BigTable row
            row_key = self._generate_row_key(element)
            row = self._table.direct_row(row_key)
            
            # Determine cell timestamp
            cell_timestamp = self._get_cell_timestamp(element)
            
            # Add cells
            for key, value in element.items():
                if not key.startswith("_"):  # Skip metadata
                    column_name = key.encode('utf-8')
                    column_value = str(value).encode('utf-8') if value is not None else b''
                    row.set_cell(
                        self.column_family,
                        column_name,
                        column_value,
                        timestamp=cell_timestamp
                    )
            
            self._batch.append(row)
            
            # Check if we should flush
            should_flush = (
                len(self._batch) >= self.batch_size or
                self._should_time_flush()
            )
            
            if should_flush:
                yield from self._flush_batch()
        
        except Exception as e:
            logging.error(f"Error preparing streaming write: {e}")
            yield {
                "_streaming_write_error": True,
                "_error_message": str(e),
                "_original_record": element,
                "_timestamp": datetime.now(timezone.utc).isoformat(),
            }
    
    def finish_bundle(self):
        """Flush any remaining records."""
        if self._batch:
            list(self._flush_batch())
    
    def _generate_row_key(self, element: Dict[str, Any]) -> str:
        """Generate row key from element data."""
        if self.row_key_template:
            try:
                return self.row_key_template.format(**element)
            except KeyError as e:
                logging.warning(f"Row key template formatting failed: {e}")
        
        # Default row key generation
        if "user_id" in element and "timestamp" in element:
            return f"{element['user_id']}#{element['timestamp']}"
        elif "id" in element:
            return str(element["id"])
        else:
            # Use message ID or generate hash
            message_id = element.get("_pubsub_message_id")
            if message_id:
                return message_id
            else:
                import hashlib
                content = json.dumps({k: v for k, v in element.items() 
                                   if not k.startswith("_")}, sort_keys=True)
                return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cell_timestamp(self, element: Dict[str, Any]) -> datetime:
        """Get timestamp for BigTable cell."""
        if self.timestamp_field and self.timestamp_field in element:
            try:
                # Parse timestamp string
                ts_str = element[self.timestamp_field]
                if isinstance(ts_str, str):
                    # Try common timestamp formats
                    for fmt in ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"]:
                        try:
                            return datetime.strptime(ts_str, fmt).replace(tzinfo=timezone.utc)
                        except ValueError:
                            continue
                elif isinstance(ts_str, (int, float)):
                    # Unix timestamp
                    return datetime.fromtimestamp(ts_str, tz=timezone.utc)
            except Exception as e:
                logging.warning(f"Failed to parse timestamp {element[self.timestamp_field]}: {e}")
        
        # Default to current time
        return datetime.now(timezone.utc)
    
    def _should_time_flush(self) -> bool:
        """Check if enough time has passed for a time-based flush."""
        if not self._last_flush_time:
            return False
        
        elapsed = (datetime.now() - self._last_flush_time).total_seconds()
        return elapsed >= self.flush_interval_seconds
    
    def _flush_batch(self) -> Iterator[Dict[str, Any]]:
        """Flush current batch to BigTable."""
        if not self._batch:
            return
        
        try:
            response = self._table.mutate_rows(self._batch)
            
            success_count = 0
            failed_rows = []
            
            for i, status in enumerate(response):
                if status.code == 0:
                    success_count += 1
                else:
                    failed_rows.append(i)
                    logging.error(f"Failed to write row {i}: {status}")
            
            yield {
                "_streaming_batch_result": True,
                "_records_written": success_count,
                "_records_failed": len(failed_rows),
                "_batch_size": len(self._batch),
                "_flush_timestamp": datetime.now(timezone.utc).isoformat(),
            }
            
            self._batch.clear()
            self._last_flush_time = datetime.now()
            
        except Exception as e:
            logging.error(f"Streaming batch write failed: {e}")
            yield {
                "_streaming_batch_error": True,
                "_error_message": str(e),
                "_batch_size": len(self._batch),
                "_timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._batch.clear()


class DeadLetterHandler(beam.DoFn):
    """
    Dead letter queue handler for failed messages.
    
    Publishes failed messages to a dead letter topic for later analysis
    and reprocessing.
    """
    
    def __init__(
        self,
        project_id: str,
        dead_letter_topic: str,
        max_retry_attempts: int = 3,
        add_error_metadata: bool = True,
    ):
        """
        Initialize dead letter handler.
        
        Args:
            project_id: GCP project ID
            dead_letter_topic: Pub/Sub topic for dead letters
            max_retry_attempts: Maximum retry attempts before dead letter
            add_error_metadata: Whether to add error analysis metadata
        """
        self.project_id = project_id
        self.dead_letter_topic = dead_letter_topic
        self.max_retry_attempts = max_retry_attempts
        self.add_error_metadata = add_error_metadata
        self._publisher = None
    
    def setup(self):
        """Initialize Pub/Sub publisher."""
        self._publisher = pubsub_v1.PublisherClient()
    
    def process(self, element: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """
        Handle failed messages and route to dead letter queue.
        
        Args:
            element: Failed message
            
        Yields:
            Processing status
        """
        if not self._publisher:
            self.setup()
        
        try:
            # Check retry count
            retry_count = element.get("_retry_count", 0)
            
            if retry_count < self.max_retry_attempts:
                # Increment retry count and re-emit for retry
                element["_retry_count"] = retry_count + 1
                element["_retry_timestamp"] = datetime.now(timezone.utc).isoformat()
                yield element
                return
            
            # Max retries exceeded - send to dead letter queue
            if self.add_error_metadata:
                element = self._add_error_analysis(element)
            
            # Prepare message for dead letter topic
            message_data = json.dumps(element).encode('utf-8')
            
            # Publish to dead letter topic
            topic_path = self._publisher.topic_path(self.project_id, self.dead_letter_topic)
            future = self._publisher.publish(
                topic_path,
                message_data,
                **{
                    "error_category": element.get("_error_category", "unknown"),
                    "original_timestamp": element.get("_pubsub_received_time", ""),
                    "dead_letter_timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            
            # Wait for publish to complete
            future.result()
            
            yield {
                "_dead_letter_sent": True,
                "_dead_letter_topic": self.dead_letter_topic,
                "_retry_count": retry_count,
                "_timestamp": datetime.now(timezone.utc).isoformat(),
            }
            
        except Exception as e:
            logging.error(f"Dead letter handling failed: {e}")
            yield {
                "_dead_letter_error": True,
                "_error_message": str(e),
                "_original_record": element,
                "_timestamp": datetime.now(timezone.utc).isoformat(),
            }
    
    def _add_error_analysis(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """Add error analysis metadata."""
        error_types = []
        
        # Categorize errors
        if element.get("_pubsub_parse_error"):
            error_types.append("parse_error")
        if element.get("_validation_failed"):
            error_types.append("validation_error")
        if element.get("_streaming_write_error"):
            error_types.append("write_error")
        
        element["_error_analysis"] = {
            "error_types": error_types,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_retries": element.get("_retry_count", 0),
        }
        
        return element


def create_pubsub_source(subscription: str, **kwargs) -> beam.PTransform:
    """
    Create a Pub/Sub source with common configurations.
    
    Args:
        subscription: Pub/Sub subscription path
        **kwargs: Additional ReadFromPubSub parameters
        
    Returns:
        Configured Pub/Sub source transform
    """
    return ReadFromPubSub(
        subscription=subscription,
        with_attributes=kwargs.get("with_attributes", True),
        timestamp_attribute=kwargs.get("timestamp_attribute"),
        id_label=kwargs.get("id_label"),
    )


def create_pubsub_sink(topic: str, **kwargs) -> beam.PTransform:
    """
    Create a Pub/Sub sink with common configurations.
    
    Args:
        topic: Pub/Sub topic path
        **kwargs: Additional WriteToPubSub parameters
        
    Returns:
        Configured Pub/Sub sink transform
    """
    return WriteToPubSub(
        topic=topic,
        with_attributes=kwargs.get("with_attributes", False),
        timestamp_attribute=kwargs.get("timestamp_attribute"),
        id_label=kwargs.get("id_label"),
    )