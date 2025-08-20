# Google Dataflow Pipeline Examples

Comprehensive examples demonstrating best practices for Google Dataflow pipelines using uv workspace management, with support for both batch (GCS-to-BigTable) and streaming (Pub/Sub-to-BigTable) data processing patterns.

## 🚀 Key Features

- **Multi-Engine Support**: Unified patterns for batch and streaming processing
- **Code Reuse**: Shared utilities and transforms across pipelines to eliminate duplication
- **Enterprise Ready**: Production-grade error handling, monitoring, and security
- **uv Workspace**: Modern Python dependency management with workspace support
- **Comprehensive Testing**: Unit, integration, and end-to-end testing frameworks
- **CI/CD Integration**: Automated deployment and monitoring
- **Advanced Streaming**: Windowing, watermarks, late data handling, and dead letter queues

## 📁 Project Structure

```
examples/dataflow-pipelines/
├── pyproject.toml                    # Root workspace configuration
├── setup.py                          # Dataflow packaging
├── README.md                         # This file
├── common/                           # Shared utilities module
│   ├── src/common/
│   │   ├── __init__.py              # Common exports
│   │   ├── transforms.py            # Batch processing transforms
│   │   ├── streaming.py             # Streaming processing utilities
│   │   ├── windowing.py             # Windowing and trigger strategies
│   │   ├── config.py                # Configuration management
│   │   └── testing.py               # Testing utilities
├── pipelines/                        # Pipeline implementations
│   ├── batch/                        # Batch processing pipelines
│   │   ├── user_events/             # Web analytics events
│   │   ├── transaction_data/        # Financial transactions
│   │   └── audit_logs/              # System audit logs
│   └── streaming/                    # Streaming processing pipelines
│       ├── real_time_events/        # Real-time event processing
│       ├── clickstream_analytics/   # Windowed clickstream analysis
│       ├── iot_sensors/             # IoT sensor data processing
│       └── fraud_detection/         # Real-time fraud detection
├── deployment/                       # Deployment and configuration
│   ├── scripts/
│   │   └── deploy_pipeline.sh       # Automated deployment script
│   ├── configs/                     # Environment configurations
│   │   ├── dev.json                 # Development environment
│   │   ├── staging.json             # Staging environment
│   │   └── prod.json                # Production environment
│   └── templates/                   # Dataflow Flex templates
├── tests/                           # Integration tests
└── .github/workflows/               # CI/CD workflows
```

## 🛠 Prerequisites

- **Python 3.9+**: Required for all pipeline code
- **uv**: Modern Python package manager (recommended) or pip
- **Google Cloud SDK**: For deployment and authentication
- **Docker**: For Flex template creation (optional)

### Install Prerequisites

```bash
# Install uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
gcloud init
gcloud auth application-default login

# Authenticate for Dataflow
gcloud auth login
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Navigate to the examples directory
cd examples/dataflow-pipelines

# Install all dependencies using uv workspace
uv sync --all-extras

# Or using pip
pip install -e .
pip install -e common/
```

### 2. Configure Environment

```bash
# Copy and customize configuration
cp deployment/configs/dev.json deployment/configs/local.json

# Edit with your project details
vim deployment/configs/local.json
```

Update the configuration with your GCP project details:

```json
{
  "project_id": "your-gcp-project-id",
  "region": "us-central1",
  "batch": {
    "user_events": {
      "input_bucket": "your-data-bucket",
      "input_prefix": "user_events/raw/",
      "bigtable_instance": "your-bigtable-instance",
      "bigtable_table": "user_events"
    }
  },
  "streaming": {
    "real_time_events": {
      "subscription": "projects/your-project/subscriptions/events-sub",
      "bigtable_instance": "your-bigtable-instance",
      "bigtable_table": "real_time_events"
    }
  }
}
```

### 3. Run Tests

```bash
# Run all tests
uv run pytest

# Run specific pipeline tests
uv run pytest pipelines/batch/user_events/tests/

# Run integration tests
uv run pytest tests/ -m integration
```

### 4. Deploy Pipeline

```bash
# Deploy batch pipeline
./deployment/scripts/deploy_pipeline.sh batch user_events dev

# Deploy streaming pipeline  
./deployment/scripts/deploy_pipeline.sh streaming real_time_events dev
```

## 📊 Pipeline Examples

### Batch Processing Pipelines

#### 1. User Events Pipeline (`pipelines/batch/user_events/`)

Processes web analytics events from GCS with comprehensive data validation and BigTable storage.

**Features:**
- JSON/CSV file parsing from GCS
- Data validation and quality checks
- Event standardization and enrichment
- Efficient BigTable writing with batching
- BigQuery analytics integration

**Usage:**
```bash
python pipelines/batch/user_events/src/user_events/pipeline.py \
  --input_bucket=my-data-bucket \
  --input_prefix=user_events/2024/01/ \
  --project_id=my-project \
  --bigtable_instance=my-instance \
  --bigtable_table=user_events \
  --runner=DirectRunner
```

#### 2. Transaction Data Pipeline (`pipelines/batch/transaction_data/`)

Processes financial transaction data with fraud detection and risk assessment.

**Features:**
- Multi-currency transaction processing
- Fraud risk scoring
- PCI compliance patterns
- Encrypted field handling

### Streaming Processing Pipelines

#### 1. Real-time Events Pipeline (`pipelines/streaming/real_time_events/`)

Processes real-time events from Pub/Sub with low-latency BigTable writes.

**Features:**
- Pub/Sub message parsing (JSON/Avro)
- Real-time event enrichment
- Alert generation for critical events
- Dead letter queue handling
- Streaming BigTable writes

**Usage:**
```bash
python pipelines/streaming/real_time_events/src/real_time_events/pipeline.py \
  --subscription=projects/my-project/subscriptions/events-sub \
  --project_id=my-project \
  --bigtable_instance=my-instance \
  --bigtable_table=real_time_events \
  --runner=DataflowRunner \
  --streaming
```

#### 2. Clickstream Analytics Pipeline (`pipelines/streaming/clickstream_analytics/`)

Windowed analytics processing for web clickstream data.

**Features:**
- Fixed and sliding time windows
- Session-based windowing
- Late data handling with watermarks
- Multi-sink output (BigTable + BigQuery)

## 🏗 Common Utilities

### Core Transforms (`common/src/common/transforms.py`)

**Batch Processing:**
- `GCSReader`: Multi-format file reader (JSON, CSV, text)
- `BigTableWriter`: Batched BigTable writer with retry logic
- `BigQueryWriter`: BigQuery writer with schema management
- `DataValidator`: Configurable data validation
- `ErrorHandler`: Central error routing and dead letter processing

### Streaming Utilities (`common/src/common/streaming.py`)

**Real-time Processing:**
- `PubSubReader`: Advanced Pub/Sub message parsing
- `StreamingBigTableWriter`: Streaming BigTable writes with windowing
- `MessageParser`: Message transformation and enrichment
- `DeadLetterHandler`: Failed message handling

### Windowing Strategies (`common/src/common/windowing.py`)

**Time-based Processing:**
- `WindowingStrategies`: Fixed, sliding, and session windows
- `TriggerStrategies`: Watermark, processing time, and hybrid triggers
- `create_fixed_windows()`: Pre-configured fixed windows
- `LateDataHandler`: Late-arriving data management

### Configuration Management (`common/src/common/config.py`)

**Environment Management:**
- `PipelineConfig`: Pydantic-based configuration
- `BatchConfig`: Batch-specific configuration
- `StreamingConfig`: Streaming-specific configuration
- Environment-specific overrides
- Validation and type safety

## 🧪 Testing Framework

### Test Types

1. **Unit Tests**: Individual transform testing
2. **Integration Tests**: End-to-end pipeline testing
3. **Performance Tests**: Throughput and latency validation
4. **Streaming Tests**: TestStream-based streaming validation

### Testing Utilities (`common/src/common/testing.py`)

- `DataflowTestCase`: Base test class with utilities
- `MockSource`/`MockSink`: Test data sources and sinks
- `create_test_data()`: Synthetic data generation
- `StreamingTestCase`: Streaming-specific test patterns
- `PipelineValidator`: Data quality validation

### Running Tests

```bash
# All tests
uv run pytest

# Unit tests only
uv run pytest -m unit

# Integration tests
uv run pytest -m integration

# Specific pipeline tests
uv run pytest pipelines/batch/user_events/tests/

# Streaming tests
uv run pytest -m streaming

# With coverage
uv run pytest --cov=common --cov=pipelines
```

## 🚀 Deployment

### Environment Configuration

Each environment has its own configuration file in `deployment/configs/`:

- `dev.json`: Development environment with minimal resources
- `staging.json`: Staging environment for testing
- `prod.json`: Production environment with full resources

### Deployment Script

The `deploy_pipeline.sh` script provides automated deployment:

```bash
# Deploy to development
./deployment/scripts/deploy_pipeline.sh batch user_events dev

# Deploy to production with custom arguments
./deployment/scripts/deploy_pipeline.sh streaming real_time_events prod \
  --max_num_workers=100 \
  --worker_machine_type=n1-standard-8

# Create Flex Template
CREATE_TEMPLATE=true ./deployment/scripts/deploy_pipeline.sh batch user_events prod
```

### Manual Deployment

```bash
# Batch pipeline
python pipelines/batch/user_events/src/user_events/pipeline.py \
  --runner=DataflowRunner \
  --project=my-project \
  --region=us-central1 \
  --temp_location=gs://my-project-temp/ \
  --staging_location=gs://my-project-staging/ \
  --setup_file=./setup.py \
  --config_file=deployment/configs/prod.json

# Streaming pipeline
python pipelines/streaming/real_time_events/src/real_time_events/pipeline.py \
  --runner=DataflowRunner \
  --streaming \
  --enable_streaming_engine \
  --project=my-project \
  --region=us-central1 \
  --setup_file=./setup.py \
  --config_file=deployment/configs/prod.json
```

## 📈 Monitoring and Observability

### Built-in Monitoring

- **Pipeline Metrics**: Throughput, latency, error rates
- **Data Quality Metrics**: Validation failures, schema violations
- **Resource Utilization**: Worker CPU, memory, disk usage
- **Cost Tracking**: Processing costs and resource optimization

### Custom Metrics

```python
# In your transforms
beam.metrics.Metrics.counter('my_pipeline', 'events_processed').inc()
beam.metrics.Metrics.distribution('my_pipeline', 'processing_latency').update(latency_ms)
beam.metrics.Metrics.gauge('my_pipeline', 'queue_depth').set(queue_size)
```

### Alerting

Configure alerts in environment configs:

```json
{
  "monitoring": {
    "alert_email": "team@example.com",
    "slack_webhook": "https://hooks.slack.com/...",
    "pagerduty_key": "your-pagerduty-key"
  }
}
```

## 🔒 Security Best Practices

### IAM and Authentication

- Service accounts with minimal required permissions
- Workload Identity for GKE integration
- Customer-managed encryption keys (CMEK)

### Network Security

- VPC networking with private IP addresses
- Private Google Access for external API calls
- Firewall rules for worker communication

### Data Protection

- Encryption at rest and in transit
- PII handling and anonymization
- Audit logging for compliance

## ⚡ Performance Optimization

### Batch Processing

- **Partitioning**: Optimal file sizes (100MB-1GB per file)
- **Parallelism**: Configure appropriate worker counts
- **Caching**: Cache expensive operations
- **Fusion**: Enable pipeline fusion for better performance

### Streaming Processing

- **Windowing**: Choose appropriate window sizes
- **Watermarks**: Configure allowed lateness
- **State Management**: Minimize state size
- **Batching**: Batch writes to external systems

### Resource Tuning

```json
{
  "max_num_workers": 50,
  "machine_type": "n1-standard-4",
  "disk_size_gb": 100,
  "autoscaling_algorithm": "THROUGHPUT_BASED",
  "enable_streaming_engine": true
}
```

## 🛠 Development Workflow

### 1. Local Development

```bash
# Install in development mode
uv sync --dev

# Run with local runner
python pipeline.py --runner=DirectRunner

# Test with small datasets
python pipeline.py --runner=DirectRunner --input_prefix=test_data/
```

### 2. Testing Strategy

```bash
# Unit tests during development
uv run pytest pipelines/batch/user_events/tests/test_transforms.py

# Integration tests before deployment
uv run pytest tests/test_integration.py

# Performance tests
uv run pytest tests/test_performance.py --slow
```

### 3. Code Quality

```bash
# Linting and formatting
uv run ruff check .
uv run black .
uv run mypy .

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure workspace is synced
   uv sync --all-extras
   
   # Check Python path
   python -c "import common.transforms; print('OK')"
   ```

2. **Authentication Issues**
   ```bash
   gcloud auth application-default login
   gcloud config set project your-project-id
   ```

3. **Resource Exhaustion**
   ```bash
   # Check Dataflow job logs
   gcloud dataflow jobs describe JOB_ID --region=us-central1
   
   # Increase worker resources
   --machine_type=n1-standard-4 --disk_size_gb=200
   ```

4. **Permission Errors**
   ```bash
   # Grant required permissions
   gcloud projects add-iam-policy-binding PROJECT_ID \
     --member="serviceAccount:SERVICE_ACCOUNT" \
     --role="roles/dataflow.worker"
   ```

### Debugging Tips

1. **Use DirectRunner** for local debugging
2. **Enable verbose logging** with `--log_level=DEBUG`
3. **Check worker logs** in Cloud Logging
4. **Monitor metrics** in Cloud Monitoring
5. **Validate data** at each pipeline stage

## 📚 Additional Resources

### Google Cloud Documentation
- [Dataflow Documentation](https://cloud.google.com/dataflow/docs)
- [Apache Beam Programming Guide](https://beam.apache.org/documentation/programming-guide/)
- [BigTable Best Practices](https://cloud.google.com/bigtable/docs/performance)

### Python Ecosystem
- [uv Documentation](https://docs.astral.sh/uv/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [pytest Documentation](https://docs.pytest.org/)

### Example Applications
- [Real-time Analytics](docs/examples/real_time_analytics.md)
- [Fraud Detection](docs/examples/fraud_detection.md)
- [IoT Data Processing](docs/examples/iot_processing.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the full test suite
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.

---

**Built with ❤️ using uv, Apache Beam, and Google Cloud Dataflow**