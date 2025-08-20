#!/usr/bin/env python3
"""Basic import test for all modules."""

import subprocess

def test_basic_import(module_name: str) -> bool:
    """Test basic module import without complex functionality."""
    test_code = f'''
try:
    import {module_name}
    print(f"✓ {module_name} imported successfully")
    
    # Try to get version
    if hasattr({module_name}, "__version__"):
        print(f"✓ Version: {{{module_name}.__version__}}")
    
    # Try to access main module attributes
    attrs = [attr for attr in dir({module_name}) if not attr.startswith("_")]
    print(f"✓ Available attributes: {{len(attrs)}}")
    
    print(f"✓ {module_name} basic import working!")
    
except ImportError as e:
    print(f"✗ {module_name} import failed: {{e}}")
    exit(1)
except Exception as e:
    print(f"✗ {module_name} error: {{e}}")
    exit(1)
'''
    
    result = subprocess.run(
        ["uv", "run", "--no-project", "python", "-c", test_code],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"✅ {module_name}: SUCCESS")
        print(result.stdout)
        return True
    else:
        print(f"❌ {module_name}: FAILED") 
        print(result.stderr)
        return False

def main():
    """Test all modules."""
    modules = [
        "commons_core",
        "commons_testing", 
        "commons_cloud",
        "commons_k8s",
        "commons_events",
        "commons_llm",
        "commons_pipelines", 
        "commons_workflows",
        "commons_agents",
        "commons_data"
    ]
    
    print("Testing basic imports for all Python Commons modules")
    print("=" * 60)
    
    results = []
    for module in modules:
        print(f"\nTesting {module}...")
        success = test_basic_import(module)
        results.append((module, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"Modules tested: {total}")
    print(f"Passed: {passed}")  
    print(f"Failed: {total - passed}")
    
    for module, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {module}: {status}")
    
    if passed == total:
        print(f"\n🎉 ALL {total} MODULES CAN BE IMPORTED!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} modules have import issues")
        return 1

if __name__ == "__main__":
    exit(main())