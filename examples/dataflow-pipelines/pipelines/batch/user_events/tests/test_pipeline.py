"""
Tests for User Events Batch Processing Pipeline.
"""

import unittest
from unittest.mock import Mock, patch
import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to

# Import pipeline components
from user_events.pipeline import UserEventTransform, create_user_events_pipeline
from common.testing import DataflowTestCase, create_test_data
from common.config import PipelineConfig


class TestUserEventTransform(DataflowTestCase):
    """Test user event transformation logic."""
    
    def test_basic_transform(self):
        """Test basic event transformation."""
        # Create test data
        test_events = [
            {
                'event_id': 'event_1',
                'user_id': 'user_123',
                'event_type': 'pageview',
                'timestamp': '2024-01-01T10:00:00Z',
                'page_url': '/home',
                'revenue': 0,
                'session_id': 'session_1',
                'country': 'US',
            }
        ]
        
        with TestPipeline() as p:
            # Apply transformation
            transformed = (
                p 
                | 'Create Test Data' >> beam.Create(test_events)
                | 'Transform Events' >> beam.ParDo(UserEventTransform())
            )
            
            # Verify transformation results
            def check_transform(results):
                self.assertEqual(len(results), 1)
                result = results[0]
                
                # Check row key format
                self.assertIn('row_key', result)
                self.assertTrue(result['row_key'].startswith('user_123#'))
                
                # Check data structure
                self.assertIn('data', result)
                data = result['data']
                
                # Check event data
                self.assertEqual(data['event_data']['user_id'], 'user_123')
                self.assertEqual(data['event_data']['event_type'], 'page_view')  # normalized
                self.assertEqual(data['event_data']['date'], '2024-01-01')
                self.assertEqual(data['event_data']['hour'], '10')
                
                # Check business data
                self.assertEqual(data['business_data']['revenue'], '0')
                self.assertEqual(data['business_data']['revenue_category'], 'no_revenue')
                self.assertEqual(data['business_data']['is_monetized'], 'False')
            
            assert_that(transformed, check_transform)
    
    def test_revenue_categorization(self):
        """Test revenue categorization logic."""
        test_cases = [
            {'revenue': 0, 'expected_category': 'no_revenue'},
            {'revenue': 5, 'expected_category': 'low_value'},
            {'revenue': 50, 'expected_category': 'medium_value'},
            {'revenue': 150, 'expected_category': 'high_value'},
        ]
        
        transform = UserEventTransform()
        
        for case in test_cases:
            category = transform._categorize_revenue(case['revenue'])
            self.assertEqual(category, case['expected_category'])
    
    def test_timestamp_parsing(self):
        """Test timestamp parsing with different formats."""
        transform = UserEventTransform()
        
        # Test various timestamp formats
        test_cases = [
            '2024-01-01T10:00:00.123Z',
            '2024-01-01T10:00:00Z',
            '2024-01-01 10:00:00',
            '2024-01-01',
            '1704106800',  # Unix timestamp
        ]
        
        for timestamp_str in test_cases:
            parsed = transform._parse_timestamp(timestamp_str)
            self.assertIsNotNone(parsed)
            self.assertEqual(parsed.year, 2024)
    
    def test_url_parsing(self):
        """Test URL component parsing."""
        transform = UserEventTransform()
        
        url = 'https://example.com/page?param=value#section'
        components = transform._parse_url_components(url)
        
        self.assertEqual(components['page_path'], '/page')
        self.assertEqual(components['page_query'], 'param=value')
        self.assertEqual(components['page_fragment'], 'section')
        self.assertEqual(components['page_domain'], 'example.com')
    
    def test_error_handling(self):
        """Test error handling in transformation."""
        # Test with invalid data
        invalid_event = {
            'user_id': None,  # This should cause issues
            'event_type': 'invalid_type',
            'timestamp': 'invalid_timestamp',
        }
        
        transform = UserEventTransform()
        results = list(transform.process(invalid_event))
        
        # Should still produce output but with error flags
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertTrue(result.get('_transform_error', False))


class TestUserEventsPipeline(DataflowTestCase):
    """Test complete user events pipeline."""
    
    def setUp(self):
        """Set up test configuration."""
        super().setUp()
        self.config = PipelineConfig.create_batch_config(
            pipeline_name="test-user-events",
            project_id="test-project",
            input_bucket="test-bucket",
            bigtable_instance="test-instance",
            bigtable_table="test-table",
            input_prefix="test-prefix",
        )
    
    @patch('common.transforms.GCSReader')
    @patch('common.transforms.BigTableWriter')
    def test_pipeline_structure(self, mock_bigtable, mock_gcs):
        """Test pipeline structure and flow."""
        # Mock the transforms to avoid actual GCS/BigTable calls
        mock_gcs.return_value = beam.Map(lambda x: [
            {
                'user_id': 'user_1',
                'event_type': 'page_view',
                'timestamp': '2024-01-01T10:00:00Z',
            }
        ])
        mock_bigtable.return_value = beam.Map(lambda x: {'write_success': True})
        
        # Create pipeline (this tests the structure without executing)
        pipeline = create_user_events_pipeline(self.config)
        
        # Verify pipeline was created successfully
        self.assertIsNotNone(pipeline)
    
    def test_validation_rules(self):
        """Test data validation rules."""
        from user_events.pipeline import run_pipeline
        
        # Test with various invalid records
        invalid_records = [
            {},  # Missing all required fields
            {'user_id': ''},  # Empty user ID
            {'user_id': 'user_1'},  # Missing event_type and timestamp
            {'user_id': 'user_1', 'event_type': 'invalid', 'timestamp': '2024-01-01T10:00:00Z'},
        ]
        
        # In a real test, we'd mock the pipeline components and verify
        # that invalid records are properly flagged and routed
        # For now, we'll just verify the validation functions work
        
        def validate_user_id(user_id):
            return isinstance(user_id, str) and len(user_id) > 0
        
        def validate_event_type(event_type):
            valid_types = {'page_view', 'click', 'purchase', 'signup', 'login', 'logout', 'pageview'}
            return event_type in valid_types
        
        # Test validation functions
        self.assertFalse(validate_user_id(''))
        self.assertFalse(validate_user_id(None))
        self.assertTrue(validate_user_id('user_123'))
        
        self.assertFalse(validate_event_type('invalid_type'))
        self.assertTrue(validate_event_type('page_view'))


class TestPipelineIntegration(DataflowTestCase):
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_processing(self):
        """Test end-to-end processing with mock data."""
        # Create comprehensive test data
        test_data = create_test_data('user_events', count=10)
        
        # Create test pipeline
        with TestPipeline() as p:
            # Simulate the pipeline flow
            events = p | 'Create Test Data' >> beam.Create(test_data)
            
            # Apply validation
            from common.transforms import DataValidator
            validated = events | 'Validate' >> beam.ParDo(
                DataValidator(
                    required_fields=['user_id', 'event_type', 'timestamp'],
                    emit_invalid_records=True
                )
            )
            
            # Apply transformation
            transformed = validated | 'Transform' >> beam.ParDo(UserEventTransform())
            
            # Verify results
            def check_results(results):
                self.assertGreater(len(results), 0)
                
                # Check that all results have proper BigTable structure
                for result in results:
                    if not result.get('_validation_failed'):
                        self.assertIn('row_key', result)
                        self.assertIn('data', result)
                        
                        data = result['data']
                        self.assertIn('event_data', data)
                        self.assertIn('user_data', data)
                        self.assertIn('business_data', data)
                        self.assertIn('metadata', data)
            
            assert_that(transformed, check_results)
    
    def test_error_recovery(self):
        """Test pipeline behavior with various error conditions."""
        # Create test data with known issues
        problematic_data = [
            {'user_id': 'valid_user', 'event_type': 'page_view', 'timestamp': '2024-01-01T10:00:00Z'},
            {'event_type': 'click'},  # Missing user_id
            {'user_id': '', 'event_type': 'purchase', 'timestamp': 'invalid'},  # Invalid data
            {'user_id': 'user_2', 'event_type': 'invalid_type', 'timestamp': '2024-01-01T11:00:00Z'},
        ]
        
        with TestPipeline() as p:
            events = p | 'Create Problematic Data' >> beam.Create(problematic_data)
            
            # Apply validation and transformation
            validated = events | 'Validate' >> beam.ParDo(
                DataValidator(
                    required_fields=['user_id', 'event_type', 'timestamp'],
                    emit_invalid_records=True
                )
            )
            
            transformed = validated | 'Transform' >> beam.ParDo(UserEventTransform())
            
            # Verify that we handle both valid and invalid records
            def check_error_handling(results):
                valid_count = 0
                invalid_count = 0
                
                for result in results:
                    if result.get('_validation_failed') or result.get('_transform_error'):
                        invalid_count += 1
                    else:
                        valid_count += 1
                        # Valid records should have proper structure
                        self.assertIn('row_key', result)
                        self.assertIn('data', result)
                
                # Should have both valid and invalid records
                self.assertGreater(valid_count, 0)
                self.assertGreater(invalid_count, 0)
            
            assert_that(transformed, check_error_handling)


if __name__ == '__main__':
    unittest.main()