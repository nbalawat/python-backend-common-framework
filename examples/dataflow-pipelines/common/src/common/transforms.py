"""
Core Apache Beam transforms for batch processing pipelines.

This module provides reusable DoFn transforms for common operations:
- Reading from GCS
- Writing to BigTable and BigQuery
- Data validation and quality checks
- Error handling and dead letter processing
"""

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from google.cloud import storage, bigtable, bigquery
import json
import csv
import logging
from typing import Dict, Any, Iterator, List, Optional, Union, Callable
from datetime import datetime, timezone
import pytz
import traceback
from io import StringIO


class GCSReader(beam.DoFn):
    """
    Reusable GCS file reader with support for multiple formats.
    
    Supports JSON, CSV, and text files with configurable parsing options.
    """
    
    def __init__(
        self,
        bucket_name: str,
        file_format: str = "json",
        encoding: str = "utf-8",
        csv_delimiter: str = ",",
        csv_headers: bool = True,
        max_file_size_mb: int = 100,
    ):
        """
        Initialize GCS reader.
        
        Args:
            bucket_name: GCS bucket name (without gs:// prefix)
            file_format: File format ('json', 'csv', 'text')
            encoding: File encoding
            csv_delimiter: CSV delimiter character
            csv_headers: Whether CSV files have headers
            max_file_size_mb: Maximum file size to process
        """
        self.bucket_name = bucket_name
        self.file_format = file_format.lower()
        self.encoding = encoding
        self.csv_delimiter = csv_delimiter
        self.csv_headers = csv_headers
        self.max_file_size_mb = max_file_size_mb
        self._client = None
    
    def setup(self):
        """Initialize GCS client."""
        self._client = storage.Client()
    
    def process(self, element: str) -> Iterator[Dict[str, Any]]:
        """
        Read and parse files from GCS.
        
        Args:
            element: GCS file path or prefix
            
        Yields:
            Parsed records from the files
        """
        if not self._client:
            self.setup()
        
        try:
            bucket = self._client.bucket(self.bucket_name)
            
            # Handle both individual files and prefixes
            if element.endswith(('.json', '.csv', '.txt')):
                # Single file
                blobs = [bucket.blob(element)]
            else:
                # Prefix - list matching files
                blobs = bucket.list_blobs(prefix=element)
            
            for blob in blobs:
                try:
                    # Skip if file is too large
                    if blob.size and blob.size > self.max_file_size_mb * 1024 * 1024:
                        logging.warning(f"Skipping large file {blob.name}: {blob.size} bytes")
                        continue
                    
                    # Read file content
                    content = blob.download_as_text(encoding=self.encoding)
                    
                    # Parse based on format
                    if self.file_format == "json":
                        yield from self._parse_json(content, blob.name)
                    elif self.file_format == "csv":
                        yield from self._parse_csv(content, blob.name)
                    elif self.file_format == "text":
                        yield from self._parse_text(content, blob.name)
                    
                except Exception as e:
                    logging.error(f"Error processing file {blob.name}: {e}")
                    # Emit error record for dead letter processing
                    yield {
                        "_error": True,
                        "_error_message": str(e),
                        "_error_traceback": traceback.format_exc(),
                        "_source_file": blob.name,
                        "_timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    
        except Exception as e:
            logging.error(f"Error accessing GCS bucket {self.bucket_name}: {e}")
            raise
    
    def _parse_json(self, content: str, filename: str) -> Iterator[Dict[str, Any]]:
        """Parse JSON content."""
        try:
            # Handle both single JSON objects and JSONL
            for line in content.strip().split('\n'):
                if line.strip():
                    record = json.loads(line)
                    # Add metadata
                    record["_source_file"] = filename
                    record["_processed_at"] = datetime.now(timezone.utc).isoformat()
                    yield record
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error in {filename}: {e}")
            raise
    
    def _parse_csv(self, content: str, filename: str) -> Iterator[Dict[str, Any]]:
        """Parse CSV content."""
        try:
            reader = csv.DictReader(
                StringIO(content),
                delimiter=self.csv_delimiter
            ) if self.csv_headers else csv.reader(
                StringIO(content),
                delimiter=self.csv_delimiter
            )
            
            for i, row in enumerate(reader):
                if isinstance(row, dict):
                    record = dict(row)
                else:
                    # Convert list to dict with column indices
                    record = {f"col_{j}": val for j, val in enumerate(row)}
                
                # Add metadata
                record["_source_file"] = filename
                record["_row_number"] = i + (1 if self.csv_headers else 0)
                record["_processed_at"] = datetime.now(timezone.utc).isoformat()
                yield record
                
        except Exception as e:
            logging.error(f"CSV parsing error in {filename}: {e}")
            raise
    
    def _parse_text(self, content: str, filename: str) -> Iterator[Dict[str, Any]]:
        """Parse plain text content."""
        for i, line in enumerate(content.strip().split('\n')):
            if line.strip():
                yield {
                    "text": line.strip(),
                    "line_number": i + 1,
                    "_source_file": filename,
                    "_processed_at": datetime.now(timezone.utc).isoformat(),
                }


class DataValidator(beam.DoFn):
    """
    Configurable data validation transform.
    
    Validates records against required fields and custom rules.
    """
    
    def __init__(
        self,
        required_fields: List[str],
        validation_rules: Optional[Dict[str, Callable]] = None,
        emit_invalid_records: bool = True,
    ):
        """
        Initialize data validator.
        
        Args:
            required_fields: List of required field names
            validation_rules: Dict of field_name -> validation_function
            emit_invalid_records: Whether to emit invalid records with error info
        """
        self.required_fields = required_fields or []
        self.validation_rules = validation_rules or {}
        self.emit_invalid_records = emit_invalid_records
    
    def process(self, element: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """
        Validate element and emit if valid.
        
        Args:
            element: Record to validate
            
        Yields:
            Valid records or invalid records with error info
        """
        errors = []
        
        # Skip error records from upstream
        if element.get("_error"):
            yield element
            return
        
        # Check required fields
        for field in self.required_fields:
            if field not in element or element[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Apply validation rules
        for field, rule in self.validation_rules.items():
            if field in element:
                try:
                    if not rule(element[field]):
                        errors.append(f"Validation failed for field {field}")
                except Exception as e:
                    errors.append(f"Validation error for field {field}: {e}")
        
        if errors:
            if self.emit_invalid_records:
                # Add error information to record
                element["_validation_errors"] = errors
                element["_validation_failed"] = True
                element["_validation_timestamp"] = datetime.now(timezone.utc).isoformat()
                yield element
            else:
                logging.warning(f"Validation failed: {errors}")
        else:
            # Add validation success metadata
            element["_validation_passed"] = True
            element["_validation_timestamp"] = datetime.now(timezone.utc).isoformat()
            yield element


class BigTableWriter(beam.DoFn):
    """
    Batched BigTable writer with error handling.
    
    Efficiently writes records to BigTable with configurable batching
    and automatic retry logic.
    """
    
    def __init__(
        self,
        project_id: str,
        instance_id: str,
        table_id: str,
        column_family: str = "cf",
        batch_size: int = 100,
        max_retry_attempts: int = 3,
        row_key_extractor: Optional[Callable[[Dict[str, Any]], str]] = None,
    ):
        """
        Initialize BigTable writer.
        
        Args:
            project_id: GCP project ID
            instance_id: BigTable instance ID
            table_id: BigTable table ID
            column_family: Column family name
            batch_size: Number of rows to batch
            max_retry_attempts: Maximum retry attempts
            row_key_extractor: Function to extract row key from record
        """
        self.project_id = project_id
        self.instance_id = instance_id
        self.table_id = table_id
        self.column_family = column_family
        self.batch_size = batch_size
        self.max_retry_attempts = max_retry_attempts
        self.row_key_extractor = row_key_extractor
        self._client = None
        self._table = None
        self._batch = []
    
    def setup(self):
        """Initialize BigTable client and table."""
        self._client = bigtable.Client(project=self.project_id)
        instance = self._client.instance(self.instance_id)
        self._table = instance.table(self.table_id)
    
    def process(self, element: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """
        Buffer and write elements to BigTable.
        
        Args:
            element: Record to write
            
        Yields:
            Write status records
        """
        if not self._table:
            self.setup()
        
        try:
            # Extract row key
            if self.row_key_extractor:
                row_key = self.row_key_extractor(element)
            else:
                row_key = self._default_row_key(element)
            
            # Create row
            row = self._table.direct_row(row_key)
            
            # Add columns (skip metadata fields)
            for key, value in element.items():
                if not key.startswith("_"):
                    column_name = key.encode('utf-8') if isinstance(key, str) else str(key).encode('utf-8')
                    column_value = str(value).encode('utf-8') if value is not None else b''
                    row.set_cell(
                        self.column_family,
                        column_name,
                        column_value,
                        timestamp=datetime.now(timezone.utc)
                    )
            
            self._batch.append(row)
            
            # Flush if batch is full
            if len(self._batch) >= self.batch_size:
                yield from self._flush_batch()
        
        except Exception as e:
            logging.error(f"Error preparing BigTable write: {e}")
            yield {
                "_write_error": True,
                "_error_message": str(e),
                "_original_record": element,
                "_timestamp": datetime.now(timezone.utc).isoformat(),
            }
    
    def finish_bundle(self):
        """Flush remaining batch on bundle completion."""
        if self._batch:
            list(self._flush_batch())  # Consume generator
    
    def _flush_batch(self) -> Iterator[Dict[str, Any]]:
        """Write batch to BigTable with retry logic."""
        for attempt in range(self.max_retry_attempts):
            try:
                response = self._table.mutate_rows(self._batch)
                
                success_count = 0
                error_count = 0
                
                # Check results
                for i, status in enumerate(response):
                    if status.code == 0:
                        success_count += 1
                    else:
                        error_count += 1
                        logging.error(f"Failed to write row {i}: {status}")
                
                # Yield batch result
                yield {
                    "_batch_write_success": True,
                    "_records_written": success_count,
                    "_records_failed": error_count,
                    "_batch_size": len(self._batch),
                    "_timestamp": datetime.now(timezone.utc).isoformat(),
                }
                
                self._batch.clear()
                return
                
            except Exception as e:
                logging.warning(f"Batch write attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retry_attempts - 1:
                    # Final attempt failed
                    yield {
                        "_batch_write_error": True,
                        "_error_message": str(e),
                        "_batch_size": len(self._batch),
                        "_timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    self._batch.clear()
    
    def _default_row_key(self, element: Dict[str, Any]) -> str:
        """Generate default row key from element."""
        # Use common key fields or hash
        if "id" in element:
            return str(element["id"])
        elif "user_id" in element and "timestamp" in element:
            return f"{element['user_id']}#{element['timestamp']}"
        else:
            # Use hash of record as fallback
            import hashlib
            content = json.dumps(element, sort_keys=True).encode()
            return hashlib.md5(content).hexdigest()


class BigQueryWriter(beam.DoFn):
    """
    BigQuery writer with automatic schema detection and error handling.
    """
    
    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        table_id: str,
        batch_size: int = 1000,
        create_disposition: str = "CREATE_IF_NEEDED",
        write_disposition: str = "WRITE_APPEND",
    ):
        """
        Initialize BigQuery writer.
        
        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            batch_size: Number of rows to batch
            create_disposition: Table creation behavior
            write_disposition: Write behavior
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.batch_size = batch_size
        self.create_disposition = create_disposition
        self.write_disposition = write_disposition
        self._client = None
        self._table = None
        self._batch = []
    
    def setup(self):
        """Initialize BigQuery client."""
        self._client = bigquery.Client(project=self.project_id)
        dataset_ref = self._client.dataset(self.dataset_id)
        self._table = dataset_ref.table(self.table_id)
    
    def process(self, element: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Buffer and write to BigQuery."""
        if not self._client:
            self.setup()
        
        # Clean element (remove metadata fields)
        clean_element = {
            k: v for k, v in element.items()
            if not k.startswith("_") or k in ["_processed_at"]  # Keep some metadata
        }
        
        self._batch.append(clean_element)
        
        if len(self._batch) >= self.batch_size:
            yield from self._flush_batch()
    
    def finish_bundle(self):
        """Flush remaining batch."""
        if self._batch:
            list(self._flush_batch())
    
    def _flush_batch(self) -> Iterator[Dict[str, Any]]:
        """Write batch to BigQuery."""
        try:
            errors = self._client.insert_rows_json(self._table, self._batch)
            
            if errors:
                logging.error(f"BigQuery insert errors: {errors}")
                yield {
                    "_bq_write_error": True,
                    "_error_details": errors,
                    "_batch_size": len(self._batch),
                    "_timestamp": datetime.now(timezone.utc).isoformat(),
                }
            else:
                yield {
                    "_bq_write_success": True,
                    "_records_written": len(self._batch),
                    "_timestamp": datetime.now(timezone.utc).isoformat(),
                }
            
            self._batch.clear()
            
        except Exception as e:
            logging.error(f"BigQuery write error: {e}")
            yield {
                "_bq_write_error": True,
                "_error_message": str(e),
                "_batch_size": len(self._batch),
                "_timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._batch.clear()


class ErrorHandler(beam.DoFn):
    """
    Central error handling transform for dead letter processing.
    """
    
    def process(self, element: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """
        Route records to appropriate output based on error status.
        
        Args:
            element: Record to process
            
        Yields:
            Records tagged for routing
        """
        # Check for various error conditions
        has_error = any([
            element.get("_error"),
            element.get("_validation_failed"),
            element.get("_write_error"),
            element.get("_batch_write_error"),
            element.get("_bq_write_error"),
        ])
        
        if has_error:
            # Tag for dead letter queue
            element["_error_category"] = self._categorize_error(element)
            element["_error_severity"] = self._determine_severity(element)
            beam.pvalue.TaggedOutput("errors", element)
        else:
            # Tag for success output
            beam.pvalue.TaggedOutput("success", element)
    
    def _categorize_error(self, element: Dict[str, Any]) -> str:
        """Categorize the type of error."""
        if element.get("_validation_failed"):
            return "validation"
        elif element.get("_write_error"):
            return "write"
        elif element.get("_error"):
            return "processing"
        else:
            return "unknown"
    
    def _determine_severity(self, element: Dict[str, Any]) -> str:
        """Determine error severity."""
        if element.get("_validation_failed"):
            return "medium"
        elif element.get("_write_error"):
            return "high"
        else:
            return "low"