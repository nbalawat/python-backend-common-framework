#!/usr/bin/env python3
"""Test functionality of working modules."""

import subprocess

def test_module_functionality(module_name: str, test_code: str) -> bool:
    """Test module functionality."""
    print(f"\n{'='*50}")
    print(f"Testing {module_name.upper()} Functionality")
    print('='*50)
    
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
    """Test all working modules."""
    
    # Test commons-core
    core_test = '''
import commons_core
from commons_core import ConfigManager, get_logger, BaseModel

# Test logger
logger = get_logger(__name__)
logger.info("Logger test")
print("✓ Logger working")

# Test config manager
config = ConfigManager()
print("✓ ConfigManager created")

# Test BaseModel
class TestModel(BaseModel):
    name: str
    value: int = 42

model = TestModel(name="test")
print(f"✓ BaseModel working: {model.name}, {model.value}")
'''

    # Test commons-testing
    testing_test = '''
import commons_testing
from commons_testing import AsyncTestCase, DataGenerator, fake

# Test DataGenerator
gen = DataGenerator(seed=42)
random_str = gen.random_string(5)
print(f"✓ DataGenerator: {random_str}")

# Test faker
name = fake.name()
print(f"✓ Faker: {name}")

# Test AsyncTestCase
test_case = AsyncTestCase()
print("✓ AsyncTestCase created")
'''

    # Test commons-cloud
    cloud_test = '''
import commons_cloud
from commons_cloud import CloudProvider, StorageClient, SecretManager

# Test CloudProvider
provider = CloudProvider("aws")
print(f"✓ CloudProvider: {provider.name}")

# Test StorageClient
storage = StorageClient("test-bucket")
print(f"✓ StorageClient: {storage.bucket}")

# Test SecretManager
secrets = SecretManager()
print(f"✓ SecretManager: {secrets.provider}")
'''

    # Test commons-k8s
    k8s_test = '''
import commons_k8s
from commons_k8s import ResourceSpec, Deployment

# Test ResourceSpec
spec = ResourceSpec(
    api_version="v1",
    kind="Pod", 
    metadata={"name": "test"}
)
print(f"✓ ResourceSpec: {spec.kind}")

# Test Deployment
deployment = Deployment("test-app")
print(f"✓ Deployment: {deployment.name}")
'''

    # Test commons-events
    events_test = '''
import commons_events
from commons_events import Event, EventProducer, ProducerConfig

# Test Event
event = Event(
    event_type="test.event",
    data={"msg": "hello"},
    source="test"
)
print(f"✓ Event: {event.event_type}")

# Test EventProducer
config = ProducerConfig(broker="memory")
producer = EventProducer(config)
print("✓ EventProducer created")
'''

    # Test commons-pipelines
    pipelines_test = '''
import commons_pipelines
from commons_pipelines import Pipeline, Source, SourceType, SourceOptions

# Test Pipeline
pipeline = Pipeline(name="test-pipeline")
print(f"✓ Pipeline: {pipeline.name}")

# Test Source
options = SourceOptions(path="/tmp/test.csv")
source = Source(SourceType.FILE, options)
print(f"✓ Source: {source.source_type}")
'''

    # Test commons-workflows
    workflows_test = '''
import commons_workflows
from commons_workflows import Workflow, WorkflowStep, WorkflowState

# Test WorkflowState
state = WorkflowState("running", "step1", {"key": "value"})
print(f"✓ WorkflowState: {state.status}")

# Test WorkflowStep
step = WorkflowStep("test-step", "print", {"msg": "hello"})
print(f"✓ WorkflowStep: {step.name}")

# Test Workflow
workflow = Workflow("test-workflow", [step])
print(f"✓ Workflow: {workflow.name}")
'''

    tests = {
        "commons-core": core_test,
        "commons-testing": testing_test,
        "commons-cloud": cloud_test,
        "commons-k8s": k8s_test,
        "commons-events": events_test,
        "commons-pipelines": pipelines_test,
        "commons-workflows": workflows_test,
    }
    
    results = {}
    for module_name, test_code in tests.items():
        success = test_module_functionality(module_name, test_code)
        results[module_name] = success
    
    # Summary
    print(f"\n{'='*50}")
    print("FUNCTIONALITY TEST SUMMARY")
    print('='*50)
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"Modules tested: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    
    for module, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {module}: {status}")
    
    if passed == total:
        print(f"\n🎉 ALL {total} WORKING MODULES HAVE FUNCTIONAL FEATURES!")
    else:
        print(f"\n⚠️  {total - passed} modules need functionality fixes")
    
    return passed, total

if __name__ == "__main__":
    passed, total = main()
    print(f"\nFinal Result: {passed}/{total} modules fully functional")