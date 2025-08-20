#!/usr/bin/env python3
"""Test the fixed modules."""

import subprocess

def test_module(module_name: str, test_code: str) -> bool:
    """Test a module."""
    print(f"\nTesting {module_name}...")
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

# Test commons-events
events_test = '''
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

print("✓ commons-events working!")
'''

# Test commons-pipelines
pipelines_test = '''
import commons_pipelines
from commons_pipelines import Pipeline, Source, SourceType, SourceOptions

# Test Pipeline
pipeline = Pipeline(name="test-pipeline")
print(f"✓ Pipeline created: {pipeline.config.name}")

# Test Source
options = SourceOptions(path="/tmp/test.csv")
source = Source(SourceType.FILE, options)
print(f"✓ Source created: {source.source_type}")

print("✓ commons-pipelines working!")
'''

# Test commons-llm
llm_test = '''
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

print("✓ commons-llm working!")
'''

# Test commons-agents
agents_test = '''
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

print("✓ commons-agents working!")
'''

# Test commons-data
data_test = '''
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

print("✓ commons-data working!")
'''

tests = {
    "commons-events": events_test,
    "commons-pipelines": pipelines_test,
    "commons-llm": llm_test,
    "commons-agents": agents_test,
    "commons-data": data_test
}

results = {}
for module_name, test_code in tests.items():
    success = test_module(module_name, test_code)
    results[module_name] = success

# Summary
print(f"\n{'='*50}")
print("FIXED MODULES TEST SUMMARY")
print('='*50)

passed = sum(results.values())
total = len(results)

for module, success in results.items():
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"  {module}: {status}")

print(f"\nFixed: {passed}/{total} modules now working")