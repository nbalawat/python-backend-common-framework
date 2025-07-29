# Getting Started with Python Commons

Welcome to Python Commons, a comprehensive collection of enterprise-grade Python modules for modern application development.

## Installation

### Install Individual Modules

```bash
# Core utilities
pip install commons-core

# Cloud abstractions
pip install commons-cloud[aws,gcp,azure]

# Kubernetes utilities
pip install commons-k8s[argo]

# Testing utilities
pip install commons-testing[databases,cloud]

# Event-driven architecture
pip install commons-events[kafka,rabbitmq]

# Data pipelines
pip install commons-pipelines[spark,polars]

# Workflow management
pip install commons-workflows[temporal]

# LLM integrations
pip install commons-llm[openai,anthropic]

# Agent orchestration
pip install commons-agents[langchain]

# Database abstractions
pip install commons-data[postgres,mongodb,redis]
```

### Install All Modules

```bash
# Install everything
pip install commons-core commons-cloud commons-k8s commons-testing \
    commons-events commons-pipelines commons-workflows commons-llm \
    commons-agents commons-data
```

## Quick Examples

### Configuration Management

```python
from commons_core import ConfigManager

# Load configuration from multiple sources
config = ConfigManager()
config.load_from_env()
config.load_from_file("config.yaml")

# Access configuration
api_key = config.get("services.api.key", secret=True)
```

### Cloud Storage

```python
from commons_cloud import StorageFactory

# Create storage client
storage = await StorageFactory.create(
    provider="aws",
    bucket="my-bucket"
)

# Upload file
await storage.upload("data.json", b'{"hello": "world"}')

# Download file
data = await storage.download("data.json")
```

### Kubernetes Operations

```python
from commons_k8s import K8sAsyncClient

# Create client
async with K8sAsyncClient.from_config() as client:
    # List pods
    pods = await client.list_pods(namespace="default")
    
    # Scale deployment
    await client.scale_deployment("my-app", replicas=3)
```

### Event Processing

```python
from commons_events import KafkaProducer, Event, EventMetadata

# Define event
@EventSchema(name="user.created", version="1.0.0")
class UserCreatedEvent(Event):
    user_id: str
    email: str

# Send event
async with KafkaProducer(bootstrap_servers="localhost:9092") as producer:
    event = UserCreatedEvent(
        user_id="123",
        email="user@example.com",
        metadata=EventMetadata()
    )
    await producer.send("user-events", event)
```

### LLM Integration

```python
from commons_llm import LLMProvider

# Create LLM client
llm = LLMProvider.create(
    provider="openai",
    model="gpt-4",
    api_key="your-key"
)

# Generate response
response = await llm.complete("Explain quantum computing in simple terms")
print(response.content)
```

### Agent Orchestration

```python
from commons_agents import Agent, Tool, AgentExecutor
from commons_llm import LLMProvider

# Define tools
@Tool(description="Search the web")
async def web_search(query: str) -> str:
    # Implementation
    return f"Results for: {query}"

# Create agent
llm = LLMProvider.create("openai", model="gpt-4")
agent = Agent(llm=llm, tools=[web_search])

# Execute task
executor = AgentExecutor(agent)
result = await executor.run("Find the latest news on AI")
```

### Data Pipelines

```python
from commons_pipelines import Pipeline, Source, Sink

# Create pipeline
pipeline = Pipeline("etl-job")

# Process data
result = (
    pipeline
    .read(Source.csv("input/*.csv"))
    .filter("age > 18")
    .groupby("country")
    .agg({"user_id": "count"})
    .write(Sink.parquet("output/"))
)

await pipeline.run()
```

### Database Operations

```python
from commons_data import DatabaseFactory

# Create database client
db = await DatabaseFactory.create(
    type="postgres",
    connection_string="postgresql://localhost/mydb"
)

# Query data
async with db.transaction() as tx:
    users = await tx.select(
        "users",
        where={"status": "active"},
        order_by="created_at DESC",
        limit=10
    )
```

## Module Overview

### commons-core
Foundation utilities including configuration management, structured logging, error handling with retry/circuit breaker patterns, and a Pydantic-based type system.

### commons-cloud
Multi-cloud abstractions for AWS, Google Cloud, and Azure services including storage (S3/GCS/Blob), compute (EC2/GCE/VMs), and secrets management.

### commons-k8s
Kubernetes utilities with multi-cluster support, resource management, custom operators, and Argo Workflows integration.

### commons-testing
Advanced testing utilities for async code, database fixtures using testcontainers, API mocking, and test data generation.

### commons-events
Event-driven architecture components supporting Kafka, RabbitMQ, Google Pub/Sub, and AWS SNS with schema management.

### commons-pipelines
Data pipeline abstractions supporting Spark, Polars, DuckDB, and other engines with unified transformation API.

### commons-workflows
Business process workflow management with saga pattern, parallel execution, and integrations with Temporal and Argo.

### commons-llm
Unified LLM API supporting OpenAI, Anthropic, Google, and other providers with caching, streaming, and framework integrations.

### commons-agents
Agent orchestration framework with ReAct agents, planning, memory systems, and integrations with LangChain and other frameworks.

### commons-data
Database abstractions for SQL, NoSQL, time-series, and vector databases with connection pooling and ORM support.

## Best Practices

1. **Use Type Hints**: All modules are fully typed for better IDE support
2. **Async First**: Most operations support async/await for better performance
3. **Configuration**: Use commons-core for centralized configuration
4. **Error Handling**: Leverage retry and circuit breaker patterns
5. **Logging**: Use structured logging for better observability
6. **Testing**: Use commons-testing fixtures for consistent test setup

## Next Steps

- Check out the [API Documentation](./api/)
- See [Example Applications](../examples/)
- Read the [Architecture Guide](./architecture.md)
- Join the [Community Discussion](https://github.com/python-commons/python-commons/discussions)