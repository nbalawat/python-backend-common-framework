# Python Commons

A comprehensive collection of enterprise-grade Python modules providing abstractions for common development needs.

## Overview

Python Commons is a modular library designed to accelerate Python development by providing battle-tested implementations of common patterns and integrations. Each module is independently installable and focuses on a specific domain.

## Modules

### Core Modules

- **commons-core**: Foundation utilities including configuration management, structured logging, error handling, and type systems
- **commons-cloud**: Multi-cloud abstractions for AWS, GCP, and Azure services
- **commons-k8s**: Kubernetes client utilities and custom operator patterns
- **commons-testing**: Advanced testing utilities for async code, databases, and integration tests

### Event & Data Processing

- **commons-events**: Event-driven architecture with support for Kafka, Pub/Sub, RabbitMQ
- **commons-pipelines**: Data pipeline abstractions for Spark, Polars, DuckDB, and more
- **commons-workflows**: Business process workflow management with Temporal and Argo
- **commons-data**: Unified database abstractions for SQL, NoSQL, and vector databases

### AI/ML Modules

- **commons-llm**: Unified API for LLM providers (OpenAI, Anthropic, Google, etc.)
- **commons-agents**: Agent orchestration with LangChain, CrewAI, and AutoGen integration

## Installation

Install individual modules as needed:

```bash
# Install core module
pip install commons-core

# Install with specific extras
pip install commons-cloud[aws,gcp]
pip install commons-llm[openai,anthropic]

# Install multiple modules
pip install commons-core commons-cloud commons-k8s
```

## Quick Start

### Configuration Management

```python
from commons_core.config import ConfigManager

config = ConfigManager()
config.load_from_env()
config.load_from_file("config.yaml")

# Access nested configuration
database_url = config.get("database.url")
api_key = config.get("services.openai.api_key", secret=True)
```

### Structured Logging

```python
from commons_core.logging import get_logger

logger = get_logger(__name__)
logger.info("Application started", user_id=123, action="startup")
```

### Cloud Storage

```python
from commons_cloud import StorageFactory

storage = StorageFactory.create("s3", bucket="my-bucket")
await storage.upload("path/to/file.txt", b"content")
content = await storage.download("path/to/file.txt")
```

### Kubernetes Operations

```python
from commons_k8s import K8sClient

client = K8sClient()
deployments = await client.list_deployments(namespace="default")
await client.scale_deployment("my-app", replicas=3)
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/python-commons/python-commons.git
cd python-commons

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run tests
uv run pytest
```

### Project Structure

```
python-commons/
├── core/           # Foundation utilities
├── cloud/          # Cloud provider abstractions
├── k8s/            # Kubernetes utilities
├── testing/        # Testing utilities
├── events/         # Event-driven architecture
├── pipelines/      # Data pipeline abstractions
├── workflows/      # Workflow management
├── llm/            # LLM abstractions
├── agents/         # Agent orchestration
└── data/           # Database abstractions
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- Documentation: [https://python-commons.readthedocs.io](https://python-commons.readthedocs.io)
- Issues: [GitHub Issues](https://github.com/python-commons/python-commons/issues)
- Discussions: [GitHub Discussions](https://github.com/python-commons/python-commons/discussions)