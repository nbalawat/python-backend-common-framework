#!/usr/bin/env python3
"""Test script to verify all Python Commons modules are working correctly."""

import sys
import importlib
import traceback
from typing import List, Tuple

# Define all modules and their key components to test
MODULES_TO_TEST = [
    ("commons_core", ["ConfigManager", "BaseModel", "get_logger", "CommonsError"]),
    ("commons_testing", ["AsyncTestCase", "fixture", "pytest"]),
    ("commons_cloud", ["CloudProvider", "StorageClient", "SecretManager"]),
    ("commons_k8s", ["K8sClient", "ResourceSpec", "DeploymentManager"]),
    ("commons_events", ["Event", "EventPublisher", "KafkaClient"]),
    ("commons_llm", ["LLMClient", "Message", "ChatRequest"]),
    ("commons_pipelines", ["Pipeline", "DataFrame", "SparkPipeline"]),
    ("commons_workflows", ["WorkflowEngine", "Step", "WorkflowState"]),
    ("commons_agents", ["BaseAgent", "AgentMemory", "Tool"]),
    ("commons_data", ["DatabaseClient", "Model", "SQLAlchemyRepository"]),
]

def test_module_import(module_name: str, components: List[str]) -> Tuple[bool, str]:
    """Test if a module can be imported and its key components are accessible."""
    try:
        # Import the module
        module = importlib.import_module(module_name)
        
        # Check if key components exist
        missing_components = []
        for component in components:
            if not hasattr(module, component):
                missing_components.append(component)
        
        if missing_components:
            return False, f"Missing components: {', '.join(missing_components)}"
        
        return True, "Success"
    except ImportError as e:
        return False, f"Import error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def main():
    """Run tests for all modules."""
    print("Testing Python Commons Modules")
    print("=" * 50)
    
    results = []
    all_passed = True
    
    for module_name, components in MODULES_TO_TEST:
        print(f"\nTesting {module_name}...", end=" ")
        success, message = test_module_import(module_name, components)
        
        if success:
            print("✓ PASSED")
        else:
            print(f"✗ FAILED: {message}")
            all_passed = False
        
        results.append((module_name, success, message))
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY:")
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if not all_passed:
        print("\nFailed modules:")
        for module_name, success, message in results:
            if not success:
                print(f"  - {module_name}: {message}")
        sys.exit(1)
    else:
        print("\nAll modules passed! ✓")
        sys.exit(0)

if __name__ == "__main__":
    main()