#!/usr/bin/env python3
"""Comprehensive test for all Python Commons modules."""

import subprocess
import sys
from typing import Dict, List, Tuple

def test_module(module_name: str, test_code: str) -> Tuple[bool, str]:
    """Test a module with given test code."""
    print(f"\n{'='*60}")
    print(f"Testing {module_name.upper()}")
    print('='*60)
    
    result = subprocess.run(
        ["uv", "run", "--no-project", "python", "-c", test_code],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"✅ {module_name}: SUCCESS")
        print(result.stdout)
        return True, result.stdout
    else:
        print(f"❌ {module_name}: FAILED")
        print(result.stderr)
        return False, result.stderr

def main():
    """Test all modules systematically."""
    
    # Test commons-testing
    testing_code = '''
import commons_testing
from commons_testing import AsyncTestCase, fake, DataGenerator
from commons_testing.fixtures import temp_dir, temp_file
from commons_testing.factories import Factory, LazyAttribute

print("✓ commons-testing imported successfully")

# Test data generator
gen = DataGenerator(seed=42)
random_str = gen.random_string(10)
random_email = gen.random_email()
print(f"✓ DataGenerator: {random_str}, {email}")

# Test faker
fake_name = fake.name()
print(f"✓ Faker: {fake_name}")

print("✓ commons-testing basic functionality working")
'''.replace('{email}', 'random_email')
    
    # Test commons-cloud
    cloud_code = '''
import commons_cloud
from commons_cloud import CloudProvider, StorageClient, SecretManager

print("✓ commons-cloud imported successfully")

# Test cloud provider
provider = CloudProvider("aws")
print(f"✓ CloudProvider created: {provider.name}")

# Test storage client
storage = StorageClient("s3://test-bucket")
print(f"✓ StorageClient created for: {storage.bucket}")

print("✓ commons-cloud basic functionality working")
'''
    
    # Test commons-k8s (simplified)
    k8s_code = '''
import commons_k8s
from commons_k8s.types import ResourceSpec, PodSpec
from commons_k8s.resources import Deployment

print("✓ commons-k8s imported successfully")

# Test resource spec
spec = ResourceSpec(
    api_version="v1",
    kind="Pod",
    metadata={"name": "test-pod"},
    spec={}
)
print(f"✓ ResourceSpec created: {spec.kind}")

# Test deployment
deployment = Deployment("test-deployment", "default")
print(f"✓ Deployment created: {deployment.name}")

print("✓ commons-k8s basic functionality working")
'''
    
    # Test commons-events
    events_code = '''
import commons_events
from commons_events.abstractions import Event
from commons_events.abstractions.producer import EventProducer, ProducerConfig

print("✓ commons-events imported successfully")

# Test event
event = Event(
    event_type="test.event",
    data={"message": "hello"},
    source="test"
)
print(f"✓ Event created: {event.event_type}")

# Test producer
config = ProducerConfig(broker="kafka")
producer = EventProducer(config)
print(f"✓ EventProducer created")

print("✓ commons-events basic functionality working")
'''
    
    # Test commons-llm
    llm_code = '''
import commons_llm
from commons_llm.abstractions import Message, ChatRequest
from commons_llm.abstractions.functions import Function, FunctionParameter

print("✓ commons-llm imported successfully")

# Test message
msg = Message(role="user", content="Hello")
print(f"✓ Message created: {msg.role}")

# Test function
func = Function(name="test_func", description="Test function")
print(f"✓ Function created: {func.name}")

print("✓ commons-llm basic functionality working")
'''
    
    # Test commons-pipelines
    pipelines_code = '''
import commons_pipelines
from commons_pipelines.abstractions import Pipeline
from commons_pipelines.abstractions.source import Source, SourceType, SourceOptions

print("✓ commons-pipelines imported successfully")

# Test pipeline
pipeline = Pipeline(name="test-pipeline")
print(f"✓ Pipeline created: {pipeline.name}")

# Test source
options = SourceOptions(path="/tmp/test.csv")
source = Source(SourceType.FILE, options)
print(f"✓ Source created: {source.source_type}")

print("✓ commons-pipelines basic functionality working")
'''
    
    # Test commons-workflows  
    workflows_code = '''
import commons_workflows
from commons_workflows.abstractions import Workflow, WorkflowStep, WorkflowState

print("✓ commons-workflows imported successfully")

# Test workflow state
state = WorkflowState(
    status="running", 
    current_step="step1",
    context={"key": "value"}
)
print(f"✓ WorkflowState created: {state.status}")

# Test step
step = WorkflowStep(
    name="test-step",
    action="print",
    parameters={"message": "hello"}
)
print(f"✓ WorkflowStep created: {step.name}")

print("✓ commons-workflows basic functionality working")
'''
    
    # Test commons-agents
    agents_code = '''
import commons_agents
from commons_agents.memory import AgentMemory
from commons_agents.tools import Tool

print("✓ commons-agents imported successfully")

# Test agent memory
memory = AgentMemory(max_size=100)
print(f"✓ AgentMemory created with max_size: {memory.max_size}")

# Test tool
tool = Tool(
    name="calculator",
    description="Basic calculator",
    parameters={"type": "number"}
)
print(f"✓ Tool created: {tool.name}")

print("✓ commons-agents basic functionality working")
'''
    
    # Test commons-data
    data_code = '''
import commons_data
from commons_data.abstractions import DatabaseClient
from commons_data.abstractions.query import Query, QueryBuilder

print("✓ commons-data imported successfully")

# Test database client
client = DatabaseClient("sqlite:///:memory:")
print(f"✓ DatabaseClient created: {client.url}")

# Test query builder
builder = QueryBuilder("users")
query = builder.where("name", "=", "test").build()
print(f"✓ QueryBuilder created query for: {query.table}")

print("✓ commons-data basic functionality working")
'''
    
    # Run all tests
    tests = {
        "commons-testing": testing_code,
        "commons-cloud": cloud_code, 
        "commons-k8s": k8s_code,
        "commons-events": events_code,
        "commons-llm": llm_code,
        "commons-pipelines": pipelines_code,
        "commons-workflows": workflows_code,
        "commons-agents": agents_code,
        "commons-data": data_code
    }
    
    results = {}
    
    for module_name, test_code in tests.items():
        success, output = test_module(module_name, test_code)
        results[module_name] = success
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print('='*60)
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"Modules tested: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    
    for module, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {module}: {status}")
    
    if passed == total:
        print(f"\n🎉 ALL {total} MODULES WORKING!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} modules need attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())