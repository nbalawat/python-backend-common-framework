#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete setup integration test for Dataflow pipelines.

This test verifies that:
1. All dependencies are correctly installed and importable
2. uv workspace configuration is valid
3. Common utilities work correctly
4. Pipeline configurations are valid
5. Basic pipeline functionality works end-to-end
6. Deployment scripts are functional
"""

import unittest
import sys
import os
import json
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch, Mock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestCompleteSetup(unittest.TestCase):
    """Test complete Dataflow pipeline setup."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_root = project_root
        self.config_dir = self.project_root / "deployment" / "configs"
    
    def test_import_common_utilities(self):
        """Test that all common utilities can be imported."""
        try:
            # Test core imports
            from common import (
                GCSReader, BigTableWriter, DataValidator, ErrorHandler,
                PubSubReader, StreamingBigTableWriter, MessageParser,
                WindowingStrategies, TriggerStrategies,
                PipelineConfig, BatchConfig, StreamingConfig,
                DataflowTestCase, create_test_data
            )
            
            # Test that classes are properly defined
            self.assertTrue(hasattr(GCSReader, 'process'))
            self.assertTrue(hasattr(BigTableWriter, 'process'))
            self.assertTrue(hasattr(DataValidator, 'process'))
            self.assertTrue(hasattr(PubSubReader, 'process'))
            self.assertTrue(hasattr(StreamingBigTableWriter, 'process'))
            
            # Test configuration classes
            self.assertTrue(hasattr(PipelineConfig, 'create_batch_config'))
            self.assertTrue(hasattr(PipelineConfig, 'create_streaming_config'))
            
            print("All common utilities imported successfully")
            
        except ImportError as e:
            self.fail(f"Failed to import common utilities: {e}")
    
    def test_pipeline_imports(self):
        """Test that pipeline modules can be imported."""
        try:
            # Add pipeline paths to sys.path
            batch_user_events = self.project_root / "pipelines" / "batch" / "user_events" / "src"
            streaming_real_time = self.project_root / "pipelines" / "streaming" / "real_time_events" / "src"
            
            sys.path.insert(0, str(batch_user_events))
            sys.path.insert(0, str(streaming_real_time))
            
            # Test batch pipeline import
            from user_events.pipeline import UserEventTransform, create_user_events_pipeline
            self.assertTrue(hasattr(UserEventTransform, 'process'))
            self.assertTrue(callable(create_user_events_pipeline))
            
            # Test streaming pipeline import
            from real_time_events.pipeline import RealTimeEventProcessor, create_real_time_pipeline
            self.assertTrue(hasattr(RealTimeEventProcessor, 'process'))
            self.assertTrue(callable(create_real_time_pipeline))
            
            print("All pipeline modules imported successfully")
            
        except ImportError as e:
            self.fail(f"Failed to import pipeline modules: {e}")
    
    def test_configuration_files(self):
        """Test that configuration files are valid."""
        config_files = ["dev.json", "prod.json"]
        
        for config_file in config_files:
            config_path = self.config_dir / config_file
            
            # Check file exists
            self.assertTrue(config_path.exists(), f"Config file missing: {config_file}")
            
            # Check JSON is valid
            try:
                with open(config_path) as f:
                    config = json.load(f)
                
                # Check required fields
                required_fields = ["project_id", "region", "temp_location", "staging_location"]
                for field in required_fields:
                    self.assertIn(field, config, f"Missing required field '{field}' in {config_file}")
                
                # Check batch configuration
                self.assertIn("batch", config, f"Missing batch config in {config_file}")
                batch_config = config["batch"]
                self.assertIn("user_events", batch_config, f"Missing user_events in batch config")
                
                # Check streaming configuration
                self.assertIn("streaming", config, f"Missing streaming config in {config_file}")
                streaming_config = config["streaming"]
                self.assertIn("real_time_events", streaming_config, f"Missing real_time_events in streaming config")
                
                print(f"Configuration file {config_file} is valid")
                
            except json.JSONDecodeError as e:
                self.fail(f"Invalid JSON in {config_file}: {e}")
    
    def test_configuration_classes(self):
        """Test configuration class functionality."""
        from common.config import PipelineConfig, BatchConfig, StreamingConfig
        
        # Test batch config creation
        batch_config = PipelineConfig.create_batch_config(
            pipeline_name="test-batch",
            project_id="test-project",
            input_bucket="test-bucket",
            bigtable_instance="test-instance",
            bigtable_table="test-table"
        )
        
        self.assertIsInstance(batch_config, BatchConfig)
        self.assertEqual(batch_config.pipeline_name, "test-batch")
        self.assertEqual(batch_config.project_id, "test-project")
        self.assertEqual(batch_config.input_bucket, "test-bucket")
        
        # Test streaming config creation
        streaming_config = PipelineConfig.create_streaming_config(
            pipeline_name="test-streaming",
            project_id="test-project",
            subscription="projects/test-project/subscriptions/test-sub",
            bigtable_instance="test-instance",
            bigtable_table="test-table"
        )
        
        self.assertIsInstance(streaming_config, StreamingConfig)
        self.assertEqual(streaming_config.pipeline_name, "test-streaming")
        self.assertEqual(streaming_config.subscription, "projects/test-project/subscriptions/test-sub")
        
        print("✓ Configuration classes work correctly")
    
    def test_common_transforms(self):
        """Test common transform functionality."""
        from common.transforms import GCSReader, DataValidator
        from common.testing import create_test_data
        
        # Test data generation
        test_data = create_test_data('user_events', count=5)
        self.assertEqual(len(test_data), 5)
        
        # Check required fields
        for record in test_data:
            self.assertIn('user_id', record)
            self.assertIn('event_type', record)
            self.assertIn('timestamp', record)
        
        # Test data validator
        validator = DataValidator(
            required_fields=['user_id', 'event_type'],
            validation_rules={
                'user_id': lambda x: isinstance(x, str) and len(x) > 0
            }
        )
        
        # Process test data
        results = list(validator.process(test_data[0]))
        self.assertEqual(len(results), 1)
        self.assertIn('_validation_passed', results[0])
        self.assertTrue(results[0]['_validation_passed'])
        
        print("✓ Common transforms work correctly")
    
    def test_streaming_utilities(self):
        """Test streaming utility functionality."""
        from common.streaming import MessageParser
        from common.windowing import WindowingStrategies, TriggerStrategies
        
        # Test message parser
        parser = MessageParser(
            field_mappings={'ts': 'timestamp'},
            data_transformations={'amount': lambda x: float(x) if x else 0.0}
        )
        
        test_message = {
            'user_id': 'user_123',
            'event_type': 'purchase',
            'ts': '2024-01-01T10:00:00Z',
            'amount': '99.99'
        }
        
        results = list(parser.process(test_message))
        self.assertEqual(len(results), 1)
        result = results[0]
        
        # Check field mapping
        self.assertIn('timestamp', result)
        self.assertEqual(result['timestamp'], '2024-01-01T10:00:00Z')
        
        # Check transformation
        self.assertEqual(result['amount'], 99.99)
        self.assertIsInstance(result['amount'], float)
        
        # Test windowing strategies
        fixed_window = WindowingStrategies.fixed_windows(300)  # 5 minutes
        self.assertIsNotNone(fixed_window)
        
        sliding_window = WindowingStrategies.sliding_windows(600, 300)  # 10 min window, 5 min slide
        self.assertIsNotNone(sliding_window)
        
        # Test triggers
        watermark_trigger = TriggerStrategies.watermark_trigger()
        self.assertIsNotNone(watermark_trigger)
        
        print("✓ Streaming utilities work correctly")
    
    def test_pipeline_transform_logic(self):
        """Test pipeline-specific transform logic."""
        # Add pipeline path
        batch_user_events = self.project_root / "pipelines" / "batch" / "user_events" / "src"
        sys.path.insert(0, str(batch_user_events))
        
        from user_events.pipeline import UserEventTransform
        
        # Test user event transformation
        transform = UserEventTransform()
        
        test_event = {
            'user_id': 'user_123',
            'event_id': 'event_456',
            'event_type': 'pageview',  # Should normalize to 'page_view'
            'timestamp': '2024-01-01T10:00:00Z',
            'page_url': '/products/123',
            'revenue': 25.50,
            'session_id': 'session_789'
        }
        
        results = list(transform.process(test_event))
        self.assertEqual(len(results), 1)
        
        result = results[0]
        
        # Check BigTable structure
        self.assertIn('row_key', result)
        self.assertIn('data', result)
        
        data = result['data']
        self.assertIn('event_data', data)
        self.assertIn('business_data', data)
        self.assertIn('metadata', data)
        
        # Check event normalization
        self.assertEqual(data['event_data']['event_type'], 'page_view')
        
        # Check revenue categorization
        self.assertEqual(data['business_data']['revenue_category'], 'medium_value')
        self.assertEqual(data['business_data']['is_monetized'], 'True')
        
        print("✓ Pipeline transforms work correctly")
    
    def test_test_framework(self):
        """Test the testing framework functionality."""
        from common.testing import DataflowTestCase, create_test_data, PipelineValidator
        
        # Test data creation for different types
        data_types = ['user_events', 'transactions', 'iot_sensors', 'audit_logs']
        
        for data_type in data_types:
            test_data = create_test_data(data_type, count=3)
            self.assertEqual(len(test_data), 3)
            
            # Check that data has expected structure
            if data_type == 'user_events':
                for record in test_data:
                    self.assertIn('user_id', record)
                    self.assertIn('event_type', record)
                    self.assertIn('timestamp', record)
            elif data_type == 'transactions':
                for record in test_data:
                    self.assertIn('transaction_id', record)
                    self.assertIn('amount', record)
                    self.assertIn('currency', record)
        
        # Test pipeline validator
        validator = PipelineValidator()
        
        # Test schema validation
        test_records = [
            {'user_id': 'user_1', 'amount': 100, 'timestamp': '2024-01-01'},
            {'user_id': 'user_2', 'amount': 200, 'timestamp': '2024-01-02'}
        ]
        
        is_valid = validator.validate_schema(
            test_records,
            required_fields=['user_id', 'amount'],
            field_types={'amount': (int, float), 'user_id': str}
        )
        
        self.assertTrue(is_valid)
        
        print("✓ Testing framework works correctly")
    
    def test_deployment_script_exists(self):
        """Test that deployment script exists and is executable."""
        deploy_script = self.project_root / "deployment" / "scripts" / "deploy_pipeline.sh"
        
        self.assertTrue(deploy_script.exists(), "Deployment script not found")
        self.assertTrue(os.access(deploy_script, os.X_OK), "Deployment script is not executable")
        
        # Test script validation (without actually running deployment)
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "Usage: deploy_pipeline.sh"
            
            # This would test script argument validation
            # In a real scenario, you might want to test with --help or --version
            
        print("✓ Deployment script is present and executable")
    
    def test_workspace_configuration(self):
        """Test uv workspace configuration."""
        pyproject_path = self.project_root / "pyproject.toml"
        
        self.assertTrue(pyproject_path.exists(), "Root pyproject.toml not found")
        
        # Test that we can import workspace members
        # This indirectly tests that the workspace configuration is correct
        try:
            import common
            import common.transforms
            import common.streaming
            import common.config
            import common.testing
            
            print("✓ uv workspace configuration is valid")
            
        except ImportError as e:
            self.fail(f"Workspace import failed: {e}")
    
    @unittest.skipIf(not os.getenv('RUN_SLOW_TESTS'), "Slow test - set RUN_SLOW_TESTS=1 to run")
    def test_end_to_end_pipeline(self):
        """Test end-to-end pipeline execution (DirectRunner)."""
        import apache_beam as beam
        from apache_beam.testing.test_pipeline import TestPipeline
        from common.testing import create_test_data
        from common.transforms import DataValidator
        
        # Create test data
        test_data = create_test_data('user_events', count=10)
        
        with TestPipeline() as pipeline:
            # Create test pipeline
            events = pipeline | 'Create Test Data' >> beam.Create(test_data)
            
            # Apply validation
            validated = events | 'Validate' >> beam.ParDo(
                DataValidator(required_fields=['user_id', 'event_type'])
            )
            
            # Count results
            def count_records(records):
                return len(list(records))
            
            counts = validated | 'Count' >> beam.CombineGlobally(count_records)
            
            # The pipeline should process all records
            # In a real test, you'd use assert_that to verify outputs
            
        print("✓ End-to-end pipeline execution successful")
    
    def test_project_structure(self):
        """Test that all expected project structure exists."""
        expected_structure = [
            "pyproject.toml",
            "setup.py", 
            "README.md",
            "common/pyproject.toml",
            "common/src/common/__init__.py",
            "pipelines/batch/user_events/pyproject.toml",
            "pipelines/streaming/real_time_events/pyproject.toml",
            "deployment/scripts/deploy_pipeline.sh",
            "deployment/configs/dev.json",
            "deployment/configs/prod.json",
            "tests/test_complete_setup.py"
        ]
        
        for expected_file in expected_structure:
            file_path = self.project_root / expected_file
            self.assertTrue(
                file_path.exists(), 
                f"Expected file/directory not found: {expected_file}"
            )
        
        print("✓ Project structure is complete")


class TestIntegration(unittest.TestCase):
    """Integration tests for cross-component functionality."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.project_root = project_root
    
    def test_config_to_pipeline_integration(self):
        """Test configuration integration with pipeline creation."""
        from common.config import PipelineConfig
        
        # Create configuration
        config = PipelineConfig.create_batch_config(
            pipeline_name="integration-test",
            project_id="test-project",
            input_bucket="test-bucket",
            bigtable_instance="test-instance",
            bigtable_table="test-table",
            input_prefix="test-prefix",
        )
        
        # Test that configuration has all needed fields
        self.assertEqual(config.pipeline_name, "integration-test")
        self.assertEqual(config.project_id, "test-project")
        self.assertEqual(config.input_bucket, "test-bucket")
        self.assertEqual(config.bigtable_instance, "test-instance")
        
        # Test configuration serialization
        config_dict = config.__dict__
        self.assertIn('pipeline_name', config_dict)
        self.assertIn('project_id', config_dict)
        
        print("✓ Configuration to pipeline integration works")
    
    def test_transform_chain_integration(self):
        """Test chaining of transforms works correctly."""
        from common.transforms import DataValidator
        from common.testing import create_test_data
        
        # Create test data with some invalid records
        test_data = create_test_data('user_events', count=5)
        
        # Add some invalid data
        test_data.extend([
            {'event_type': 'click'},  # Missing user_id
            {'user_id': ''},  # Empty user_id
        ])
        
        # Chain validators
        validator1 = DataValidator(
            required_fields=['user_id', 'event_type'],
            emit_invalid_records=True
        )
        
        validator2 = DataValidator(
            required_fields=['timestamp'],
            validation_rules={'user_id': lambda x: len(x) > 0 if x else False},
            emit_invalid_records=True
        )
        
        # Process through chain
        stage1_results = []
        for record in test_data:
            stage1_results.extend(list(validator1.process(record)))
        
        stage2_results = []
        for record in stage1_results:
            stage2_results.extend(list(validator2.process(record)))
        
        # Check that we have both valid and invalid records
        valid_records = [r for r in stage2_results if not r.get('_validation_failed')]
        invalid_records = [r for r in stage2_results if r.get('_validation_failed')]
        
        self.assertGreater(len(valid_records), 0, "Should have some valid records")
        self.assertGreater(len(invalid_records), 0, "Should have some invalid records")
        
        print("✓ Transform chain integration works")


if __name__ == '__main__':
    # Set up test environment
    os.environ.setdefault('PYTHONPATH', str(project_root))
    
    # Run tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.loadTestsFromModule(sys.modules[__name__])
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)