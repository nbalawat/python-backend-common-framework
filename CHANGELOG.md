# Changelog

All notable changes to Python Commons will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of Python Commons library
- `commons-core`: Foundation utilities including configuration management, structured logging, error handling, and type system
- `commons-cloud`: Multi-cloud abstractions for AWS, Google Cloud, and Azure services
- `commons-k8s`: Kubernetes utilities with client management, operators, and Argo integration
- `commons-testing`: Advanced testing utilities for async code, databases, and integration tests
- `commons-events`: Event-driven architecture components with support for Kafka, RabbitMQ, Pub/Sub
- `commons-pipelines`: Data pipeline abstractions for Spark, Polars, DuckDB
- `commons-workflows`: Business process workflow management with Temporal and Argo
- `commons-llm`: Unified LLM API for OpenAI, Anthropic, Google, and other providers
- `commons-agents`: Agent orchestration with LangChain and CrewAI integration
- `commons-data`: Database abstractions for SQL, NoSQL, and vector databases

### Infrastructure
- Monorepo structure with uv workspace management
- Comprehensive CI/CD pipelines with GitHub Actions
- Security scanning with Bandit, Safety, and CodeQL
- Pre-commit hooks for code quality
- Makefile for common development tasks

## [0.1.0] - TBD

### Core Module
- Configuration management with multi-source support (env, files, CLI)
- Structured JSON logging with context propagation
- Retry decorators with exponential backoff
- Circuit breaker pattern implementation
- Pydantic-based type system with validators
- Async utilities including rate limiting and batching
- DateTime utilities with timezone support
- Common decorators (cached, measure_time, deprecated, singleton)

### Cloud Module
- Storage abstractions for S3, GCS, and Azure Blob
- Compute abstractions for EC2, GCE, and Azure VMs
- Secrets management for AWS Secrets Manager, GCP Secret Manager, Azure Key Vault
- Multi-cloud factory pattern for easy provider switching
- Async/await support throughout

### K8s Module
- Multi-cluster client management
- Resource CRUD operations with watch support
- Custom Resource Definition (CRD) support
- Operator pattern base classes
- Argo Workflows integration
- Batch operations support

### Testing Module
- Enhanced async testing fixtures
- Database test containers (PostgreSQL, MySQL, MongoDB, Redis)
- API testing client with mock server
- Test data factories and generators
- Cloud service mocking (AWS, GCP, Azure)
- Integration with testcontainers

### Events Module
- Flexible event schemas with Avro, JSON Schema, Protobuf
- Producer implementations for Kafka, RabbitMQ, Pub/Sub, SNS
- Consumer implementations with at-least-once delivery
- Dead letter queue support
- Schema registry integration
- Basic stream processing capabilities

### LLM Module
- Unified API for multiple LLM providers
- Streaming support for all providers
- Function calling abstractions
- Embedding support with dimension control
- Response caching with Redis/in-memory backends
- Cost tracking and monitoring
- LangChain and LlamaIndex integrations

[Unreleased]: https://github.com/python-commons/python-commons/compare/v0.1.0...HEAD