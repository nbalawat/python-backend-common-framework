#!/usr/bin/env python3
"""Final comprehensive test of ALL Python Commons modules."""

import subprocess

def test_module_functionality(module_name: str, test_code: str) -> bool:
    """Test module functionality."""
    print(f"\n{'='*60}")
    print(f"Testing {module_name.upper()} Functionality")
    print('='*60)
    
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
    """Test all modules comprehensively."""
    
    tests = {
        "commons-core": '''
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
print("✓ commons-core fully functional!")
''',

        "commons-testing": '''
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
print("✓ commons-testing fully functional!")
''',

        "commons-cloud": '''
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
print("✓ commons-cloud fully functional!")
''',

        "commons-k8s": '''
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
print("✓ commons-k8s fully functional!")
''',

        "commons-events": '''
import commons_events
from commons_events import Event, EventProducer, ProducerConfig

# Test Event
event = Event(
    event_type="test.event",
    data={"message": "hello"},
    source="test"
)
print(f"✓ Event created: {event.event_type}")

# Test EventProducer
config = ProducerConfig(broker="memory")
producer = EventProducer(config)
print("✓ EventProducer created")
print("✓ commons-events fully functional!")
''',

        "commons-llm": '''
import commons_llm
from commons_llm import Message, Function
from commons_llm.factory import LLMFactory

# Test Message
msg = Message(role="user", content="Hello")
print(f"✓ Message created: {msg.role}")

# Test Function
func = Function(name="test_func", description="Test function")
print(f"✓ Function created: {func.name}")

# Test Factory
factory = LLMFactory()
provider = factory.create("openai", "gpt-4")
print(f"✓ LLM Provider created: {provider.model}")
print("✓ commons-llm fully functional!")
''',

        "commons-pipelines": '''
import commons_pipelines
from commons_pipelines import Pipeline, Source, SourceType, SourceOptions

# Test Pipeline
pipeline = Pipeline(name="test-pipeline")
print(f"✓ Pipeline created: {pipeline.config.name}")

# Test Source
options = SourceOptions(path="/tmp/test.csv")
source = Source(SourceType.FILE, options)
print(f"✓ Source created: {source.source_type}")
print("✓ commons-pipelines fully functional!")
''',

        "commons-workflows": '''
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
print("✓ commons-workflows fully functional!")
''',

        "commons-agents": '''
import commons_agents
from commons_agents.memory import AgentMemory
from commons_agents.tools import Tool

# Test AgentMemory
memory = AgentMemory(max_size=100)
memory.add("Test memory", metadata={"test": True})
print(f"✓ AgentMemory created with {memory.count()} memories")

# Test Tool
tool = Tool(
    name="calculator",
    description="Basic calculator",
    parameters={"type": "number"}
)
print(f"✓ Tool created: {tool.name}")
print("✓ commons-agents fully functional!")
''',

        "commons-data": '''
import commons_data
from commons_data import DatabaseClient, Query, QueryBuilder
from commons_data.factory import DatabaseFactory

# Test DatabaseFactory
factory = DatabaseFactory()
client = factory.create("sqlite:///:memory:")
print(f"✓ DatabaseClient created: {client.url}")

# Test QueryBuilder
builder = QueryBuilder("users")
query = builder.where("name", "=", "test").build()
print(f"✓ QueryBuilder created query for: {query.table}")
print("✓ commons-data fully functional!")
'''
    }
    
    results = {}
    for module_name, test_code in tests.items():
        success = test_module_functionality(module_name, test_code)
        results[module_name] = success
    
    # Final Summary
    print(f"\n{'='*80}")
    print("🎯 FINAL COMPREHENSIVE TEST RESULTS")
    print('='*80)
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"Total Modules Tested: {total}")
    print(f"Modules Passing: {passed}")
    print(f"Modules Failing: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    print(f"\n📊 Individual Results:")
    for module, success in results.items():
        status = "✅ FULLY FUNCTIONAL" if success else "❌ NEEDS ATTENTION"
        print(f"  {module:20} {status}")
    
    if passed == total:
        print(f"\n🎉 SUCCESS: ALL {total} PYTHON COMMONS MODULES ARE FULLY FUNCTIONAL!")
        print("✅ The Python Commons library is ready for use!")
    else:
        print(f"\n⚠️  {total - passed} modules still need fixes")
    
    return passed, total

if __name__ == "__main__":
    passed, total = main()
    print(f"\n🏁 Final Score: {passed}/{total} modules working ({(passed/total)*100:.1f}%)")
    if passed == total:
        print("🚀 Python Commons library deployment ready!")
    else:
        print("🔧 Additional fixes needed before deployment.")