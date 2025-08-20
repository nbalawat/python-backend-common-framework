"""
Testing utilities for Dataflow pipelines.

This module provides base test classes, mock sources and sinks,
test data generators, and pipeline testing patterns for both
batch and streaming pipelines.
"""

import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to, is_not_empty
from apache_beam.testing.test_stream import TestStream
from apache_beam.transforms import window
import unittest
from unittest.mock import Mock, patch
from typing import Dict, Any, List, Iterator, Optional, Callable
import json
import random
from datetime import datetime, timezone, timedelta
import logging
from io import StringIO
import tempfile
import os


class DataflowTestCase(unittest.TestCase):
    """
    Base test case for Dataflow pipelines with common utilities.
    """
    
    def setUp(self):
        """Set up test environment."""
        self.test_pipeline = TestPipeline()
        self.temp_files = []
    
    def tearDown(self):
        """Clean up test environment."""
        # Clean up temporary files
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def create_test_data(
        self,
        count: int = 10,
        data_type: str = "user_events",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Create test data for pipelines.
        
        Args:
            count: Number of records to generate
            data_type: Type of test data ('user_events', 'transactions', etc.)
            **kwargs: Additional data generation parameters
            
        Returns:
            List of test records
        """
        if data_type == "user_events":
            return self._create_user_events(count, **kwargs)
        elif data_type == "transactions":
            return self._create_transactions(count, **kwargs)
        elif data_type == "iot_sensors":
            return self._create_iot_data(count, **kwargs)
        elif data_type == "audit_logs":
            return self._create_audit_logs(count, **kwargs)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def _create_user_events(self, count: int, **kwargs) -> List[Dict[str, Any]]:
        """Generate user event test data."""
        events = []
        base_time = kwargs.get('base_time', datetime.now(timezone.utc))
        
        for i in range(count):
            event_time = base_time + timedelta(minutes=i)
            events.append({
                'user_id': f'user_{i % 100}',  # 100 unique users
                'event_id': f'event_{i}',
                'event_type': random.choice(['page_view', 'click', 'purchase', 'signup']),
                'timestamp': event_time.isoformat(),
                'session_id': f'session_{i // 10}',  # 10 events per session
                'page_url': f'/page_{random.randint(1, 20)}',
                'user_agent': 'Mozilla/5.0 (Test Browser)',
                'ip_address': f'192.168.1.{random.randint(1, 254)}',
                'country': random.choice(['US', 'CA', 'UK', 'DE', 'FR']),
                'device_type': random.choice(['desktop', 'mobile', 'tablet']),
                'revenue': random.uniform(0, 100) if random.random() < 0.1 else 0,  # 10% have revenue
            })
        return events
    
    def _create_transactions(self, count: int, **kwargs) -> List[Dict[str, Any]]:
        """Generate transaction test data."""
        transactions = []
        base_time = kwargs.get('base_time', datetime.now(timezone.utc))
        
        for i in range(count):
            transaction_time = base_time + timedelta(minutes=i * 2)
            transactions.append({
                'transaction_id': f'txn_{i}',
                'user_id': f'user_{i % 50}',  # 50 unique users
                'amount': round(random.uniform(10.0, 1000.0), 2),
                'currency': random.choice(['USD', 'EUR', 'GBP', 'CAD']),
                'timestamp': transaction_time.isoformat(),
                'merchant_id': f'merchant_{random.randint(1, 20)}',
                'category': random.choice(['groceries', 'gas', 'restaurants', 'shopping', 'utilities']),
                'payment_method': random.choice(['credit_card', 'debit_card', 'paypal', 'bank_transfer']),
                'status': random.choices(
                    ['completed', 'pending', 'failed'],
                    weights=[0.85, 0.10, 0.05]
                )[0],
                'location': {
                    'city': random.choice(['New York', 'San Francisco', 'Chicago', 'Austin', 'Seattle']),
                    'country': 'US',
                    'lat': round(random.uniform(25.0, 49.0), 6),
                    'lng': round(random.uniform(-125.0, -66.0), 6),
                }
            })
        return transactions
    
    def _create_iot_data(self, count: int, **kwargs) -> List[Dict[str, Any]]:
        """Generate IoT sensor test data."""
        sensors = []
        base_time = kwargs.get('base_time', datetime.now(timezone.utc))
        
        for i in range(count):
            reading_time = base_time + timedelta(seconds=i * 30)  # Reading every 30 seconds
            sensors.append({
                'sensor_id': f'sensor_{i % 20}',  # 20 unique sensors
                'timestamp': reading_time.isoformat(),
                'temperature': round(random.uniform(18.0, 35.0), 1),
                'humidity': round(random.uniform(30.0, 80.0), 1),
                'pressure': round(random.uniform(980.0, 1040.0), 2),
                'battery_level': random.randint(10, 100),
                'location': f'building_{i % 5}_floor_{(i % 20) // 4 + 1}',
                'device_type': random.choice(['temperature', 'motion', 'air_quality', 'occupancy']),
                'firmware_version': random.choice(['v1.2.3', 'v1.2.4', 'v1.3.0']),
                'signal_strength': random.randint(-90, -30),  # dBm
                'status': random.choices(
                    ['normal', 'warning', 'error'],
                    weights=[0.90, 0.08, 0.02]
                )[0],
            })
        return sensors
    
    def _create_audit_logs(self, count: int, **kwargs) -> List[Dict[str, Any]]:
        """Generate audit log test data."""
        logs = []
        base_time = kwargs.get('base_time', datetime.now(timezone.utc))
        
        for i in range(count):
            log_time = base_time + timedelta(seconds=i * 10)
            logs.append({
                'log_id': f'log_{i}',
                'timestamp': log_time.isoformat(),
                'user_id': f'user_{i % 30}',  # 30 unique users
                'action': random.choice([
                    'login', 'logout', 'create_resource', 'update_resource',
                    'delete_resource', 'access_resource', 'download_file', 'upload_file'
                ]),
                'resource_type': random.choice(['file', 'database', 'api', 'admin_panel']),
                'resource_id': f'resource_{random.randint(1, 100)}',
                'source_ip': f'{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}',
                'user_agent': random.choice([
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                    'API Client v1.0'
                ]),
                'result': random.choices(
                    ['success', 'failure', 'unauthorized'],
                    weights=[0.85, 0.10, 0.05]
                )[0],
                'session_id': f'session_{i // 20}',  # 20 actions per session
            })
        return logs
    
    def create_temp_json_file(self, data: List[Dict[str, Any]]) -> str:
        """
        Create a temporary JSON file with test data.
        
        Args:
            data: List of records to write
            
        Returns:
            Path to temporary file
        """
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        for record in data:
            json.dump(record, temp_file)
            temp_file.write('\n')
        temp_file.close()
        
        self.temp_files.append(temp_file.name)
        return temp_file.name
    
    def assert_pipeline_output(
        self,
        pipeline: beam.Pipeline,
        expected_output: List[Any],
        label: str = "Assert Output"
    ):
        """
        Assert pipeline output matches expected results.
        
        Args:
            pipeline: Pipeline output PCollection
            expected_output: Expected results
            label: Label for assertion
        """
        assert_that(pipeline, equal_to(expected_output), label=label)
    
    def assert_pipeline_not_empty(
        self,
        pipeline: beam.Pipeline,
        label: str = "Assert Not Empty"
    ):
        """
        Assert pipeline output is not empty.
        
        Args:
            pipeline: Pipeline output PCollection
            label: Label for assertion
        """
        assert_that(pipeline, is_not_empty(), label=label)


class MockSource(beam.DoFn):
    """
    Mock source for testing pipelines.
    """
    
    def __init__(self, test_data: List[Dict[str, Any]]):
        """
        Initialize mock source.
        
        Args:
            test_data: List of records to emit
        """
        self.test_data = test_data
    
    def process(self, element=None) -> Iterator[Dict[str, Any]]:
        """Emit test data."""
        for record in self.test_data:
            yield record


class MockSink(beam.DoFn):
    """
    Mock sink for capturing pipeline output.
    """
    
    def __init__(self):
        """Initialize mock sink."""
        self.output = []
    
    def process(self, element: Dict[str, Any]):
        """Capture output element."""
        self.output.append(element)
    
    def get_output(self) -> List[Dict[str, Any]]:
        """Get captured output."""
        return self.output.copy()


class StreamingTestCase(DataflowTestCase):
    """
    Extended test case for streaming pipelines.
    """
    
    def create_test_stream(
        self,
        data: List[Dict[str, Any]],
        timestamps: Optional[List[int]] = None,
        advance_watermark_to_infinity: bool = True
    ) -> TestStream:
        """
        Create a test stream for streaming pipeline testing.
        
        Args:
            data: List of test records
            timestamps: Optional list of timestamps (Unix seconds)
            advance_watermark_to_infinity: Whether to advance watermark
            
        Returns:
            Configured TestStream
        """
        stream = TestStream()
        
        if timestamps:
            # Add elements with specific timestamps
            for i, (record, ts) in enumerate(zip(data, timestamps)):
                stream = stream.add_elements([record], timestamp=ts)
                # Advance watermark after each element
                stream = stream.advance_watermark_to(ts + 1)
        else:
            # Add all elements at once
            stream = stream.add_elements(data)
        
        if advance_watermark_to_infinity:
            stream = stream.advance_watermark_to_infinity()
        
        return stream
    
    def create_windowed_test_data(
        self,
        window_count: int = 3,
        elements_per_window: int = 5,
        window_size_seconds: int = 60,
        data_type: str = "user_events"
    ) -> List[Dict[str, Any]]:
        """
        Create test data distributed across multiple windows.
        
        Args:
            window_count: Number of windows
            elements_per_window: Elements per window
            window_size_seconds: Window size in seconds
            data_type: Type of test data
            
        Returns:
            List of test records with timestamps
        """
        all_data = []
        base_time = datetime.now(timezone.utc)
        
        for window_idx in range(window_count):
            window_start = base_time + timedelta(seconds=window_idx * window_size_seconds)
            window_data = self.create_test_data(
                elements_per_window,
                data_type,
                base_time=window_start
            )
            all_data.extend(window_data)
        
        return all_data


def create_test_data(
    data_type: str,
    count: int = 10,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Standalone function to create test data.
    
    Args:
        data_type: Type of test data
        count: Number of records
        **kwargs: Additional parameters
        
    Returns:
        List of test records
    """
    test_case = DataflowTestCase()
    return test_case.create_test_data(count, data_type, **kwargs)


class PipelineValidator:
    """
    Utility class for validating pipeline behavior.
    """
    
    def __init__(self):
        """Initialize validator."""
        self.validation_errors = []
    
    def validate_schema(
        self,
        records: List[Dict[str, Any]],
        required_fields: List[str],
        field_types: Optional[Dict[str, type]] = None
    ) -> bool:
        """
        Validate record schema.
        
        Args:
            records: List of records to validate
            required_fields: Required field names
            field_types: Expected field types
            
        Returns:
            True if validation passes
        """
        field_types = field_types or {}
        
        for i, record in enumerate(records):
            # Check required fields
            missing_fields = set(required_fields) - set(record.keys())
            if missing_fields:
                self.validation_errors.append(
                    f"Record {i}: Missing required fields {missing_fields}"
                )
            
            # Check field types
            for field, expected_type in field_types.items():
                if field in record and not isinstance(record[field], expected_type):
                    self.validation_errors.append(
                        f"Record {i}: Field {field} has type {type(record[field])}, "
                        f"expected {expected_type}"
                    )
        
        return len(self.validation_errors) == 0
    
    def validate_data_quality(
        self,
        records: List[Dict[str, Any]],
        quality_checks: Dict[str, Callable[[Any], bool]]
    ) -> bool:
        """
        Validate data quality.
        
        Args:
            records: List of records to validate
            quality_checks: Dict of field -> validation function
            
        Returns:
            True if validation passes
        """
        for i, record in enumerate(records):
            for field, check_func in quality_checks.items():
                if field in record:
                    try:
                        if not check_func(record[field]):
                            self.validation_errors.append(
                                f"Record {i}: Quality check failed for field {field}"
                            )
                    except Exception as e:
                        self.validation_errors.append(
                            f"Record {i}: Quality check error for field {field}: {e}"
                        )
        
        return len(self.validation_errors) == 0
    
    def get_validation_report(self) -> Dict[str, Any]:
        """
        Get validation report.
        
        Returns:
            Validation report with errors and statistics
        """
        return {
            "is_valid": len(self.validation_errors) == 0,
            "error_count": len(self.validation_errors),
            "errors": self.validation_errors,
        }


class MockBigTableWriter:
    """
    Mock BigTable writer for testing.
    """
    
    def __init__(self):
        """Initialize mock writer."""
        self.written_rows = []
        self.write_errors = []
    
    def write_row(self, row_key: str, data: Dict[str, Any]):
        """Mock write row operation."""
        self.written_rows.append({
            "row_key": row_key,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    
    def simulate_error(self, error_message: str):
        """Simulate a write error."""
        self.write_errors.append({
            "error": error_message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    
    def get_written_data(self) -> List[Dict[str, Any]]:
        """Get all written data."""
        return self.written_rows.copy()
    
    def reset(self):
        """Reset mock state."""
        self.written_rows.clear()
        self.write_errors.clear()


# Test decorators and utilities

def skip_if_no_gcp():
    """Skip test if GCP credentials are not available."""
    def decorator(test_func):
        def wrapper(*args, **kwargs):
            try:
                from google.auth import default
                default()
                return test_func(*args, **kwargs)
            except Exception:
                raise unittest.SkipTest("GCP credentials not available")
        return wrapper
    return decorator


def with_temp_gcs_bucket(bucket_name: str):
    """Decorator to create/cleanup temporary GCS bucket for testing."""
    def decorator(test_func):
        def wrapper(*args, **kwargs):
            # This would create a temporary bucket in real implementation
            # For now, just mock the bucket operations
            with patch('google.cloud.storage.Client') as mock_client:
                mock_bucket = Mock()
                mock_client.return_value.bucket.return_value = mock_bucket
                return test_func(*args, **kwargs)
        return wrapper
    return decorator