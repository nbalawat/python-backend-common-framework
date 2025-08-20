#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple setup test for Dataflow pipelines.
Tests basic imports and configuration.
"""

import sys
import os
from pathlib import Path

# Add project root and common module to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "common" / "src"))

def test_imports():
    """Test that all common utilities can be imported."""
    try:
        # Test core imports first (these should always work)
        from common.config import PipelineConfig, BatchConfig, StreamingConfig
        from common.testing import DataflowTestCase, create_test_data
        
        # Test batch transforms
        from common.transforms import DataValidator, ErrorHandler
        
        print("[OK] Core utilities imported successfully")
        
        # Test advanced imports (may have optional dependencies)
        try:
            from common.transforms import GCSReader, BigTableWriter, BigQueryWriter
            print("[OK] Batch transforms imported successfully")
        except ImportError as e:
            print(f"[WARN] Some batch transforms unavailable: {e}")
        
        try:
            from common.streaming import PubSubReader, StreamingBigTableWriter, MessageParser
            print("[OK] Streaming transforms imported successfully")  
        except ImportError as e:
            print(f"[WARN] Some streaming transforms unavailable: {e}")
        
        try:
            from common.windowing import WindowingStrategies, TriggerStrategies
            print("[OK] Windowing utilities imported successfully")
        except ImportError as e:
            print(f"[WARN] Some windowing utilities unavailable: {e}")
        
        print("PASS: All available utilities imported successfully")
        return True
        
    except ImportError as e:
        print(f"FAIL: Failed to import core utilities: {e}")
        return False

def test_configuration():
    """Test configuration functionality."""
    try:
        from common.config import PipelineConfig
        
        # Test batch config creation
        batch_config = PipelineConfig.create_batch_config(
            pipeline_name="test-batch",
            project_id="test-project", 
            input_bucket="test-bucket",
            bigtable_instance="test-instance",
            bigtable_table="test-table"
        )
        
        assert batch_config.pipeline_name == "test-batch"
        assert batch_config.project_id == "test-project"
        
        print("PASS: Configuration classes work correctly")
        return True
        
    except Exception as e:
        print(f"FAIL: Configuration test failed: {e}")
        return False

def test_transforms():
    """Test common transform functionality."""
    try:
        from common.transforms import DataValidator
        from common.testing import create_test_data
        
        # Test data generation
        test_data = create_test_data('user_events', count=3)
        assert len(test_data) == 3
        
        # Test validator
        validator = DataValidator(
            required_fields=['user_id', 'event_type'],
            validation_rules={'user_id': lambda x: isinstance(x, str) and len(x) > 0}
        )
        
        results = list(validator.process(test_data[0]))
        assert len(results) == 1
        
        print("PASS: Transform functionality works correctly")
        return True
        
    except Exception as e:
        print(f"FAIL: Transform test failed: {e}")
        return False

def test_project_structure():
    """Test that expected files exist."""
    expected_files = [
        "pyproject.toml",
        "setup.py",
        "README.md",
        "common/pyproject.toml",
        "common/src/common/__init__.py",
        "deployment/configs/dev.json",
        "deployment/scripts/deploy_pipeline.sh"
    ]
    
    missing_files = []
    for expected_file in expected_files:
        file_path = project_root / expected_file
        if not file_path.exists():
            missing_files.append(expected_file)
    
    if missing_files:
        print(f"FAIL: Missing files: {missing_files}")
        return False
    else:
        print("PASS: All expected files exist")
        return True

def main():
    """Run all tests."""
    print("Running Dataflow Pipeline Setup Tests...")
    print("=" * 50)
    
    tests = [
        test_project_structure,
        test_imports,
        test_configuration,
        test_transforms,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"ERROR in {test.__name__}: {e}")
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("SUCCESS: All tests passed! Dataflow pipeline setup is working correctly.")
        return 0
    else:
        print("FAILURE: Some tests failed. Please check the setup.")
        return 1

if __name__ == '__main__':
    sys.exit(main())