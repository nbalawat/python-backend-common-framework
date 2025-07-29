# Commons Cloud

Multi-cloud abstractions for AWS, Google Cloud, and Azure services.

## Features

- **Storage**: Unified API for S3, Google Cloud Storage, and Azure Blob Storage
- **Compute**: VM management across EC2, Compute Engine, and Azure VMs
- **Secrets**: Consistent interface for AWS Secrets Manager, GCP Secret Manager, and Azure Key Vault
- **Additional Services**: Message queues, DNS, CDN, monitoring, and more

## Installation

```bash
# Install with all providers
pip install commons-cloud[aws,gcp,azure]

# Install with specific provider
pip install commons-cloud[aws]
pip install commons-cloud[gcp]
pip install commons-cloud[azure]
```

## Usage

### Storage

```python
from commons_cloud import StorageFactory

# Create storage client
storage = await StorageFactory.create(
    provider="aws",  # or "gcp", "azure"
    bucket="my-bucket",
    region="us-east-1"
)

# Upload file
await storage.upload("path/to/file.txt", b"Hello, World!")

# Download file
content = await storage.download("path/to/file.txt")

# List files
files = await storage.list_objects(prefix="path/")

# Generate signed URL
url = await storage.create_signed_url("path/to/file.txt", expires_in=3600)

# Delete file
await storage.delete("path/to/file.txt")
```

### Compute

```python
from commons_cloud import ComputeFactory

# Create compute client
compute = await ComputeFactory.create(
    provider="aws",
    region="us-east-1"
)

# List instances
instances = await compute.list_instances()

# Create instance
instance = await compute.create_instance(
    name="my-instance",
    image_id="ami-12345678",
    instance_type="t3.micro",
    key_name="my-key"
)

# Start/stop instance
await compute.start_instance(instance.id)
await compute.stop_instance(instance.id)

# Terminate instance
await compute.terminate_instance(instance.id)
```

### Secrets Management

```python
from commons_cloud import SecretsFactory

# Create secrets client
secrets = await SecretsFactory.create(
    provider="aws",
    region="us-east-1"
)

# Store secret
await secrets.create_secret(
    name="api-key",
    value="secret-value",
    description="API key for external service"
)

# Retrieve secret
value = await secrets.get_secret("api-key")

# Update secret
await secrets.update_secret("api-key", "new-secret-value")

# List secrets
all_secrets = await secrets.list_secrets()

# Delete secret
await secrets.delete_secret("api-key")

# Enable automatic rotation
await secrets.enable_rotation(
    name="database-password",
    rotation_lambda="arn:aws:lambda:..."
)
```

### Multi-Provider Support

```python
from commons_cloud import CloudProvider

# Configure providers
providers = {
    "aws": CloudProvider(
        provider="aws",
        credentials={"access_key": "...", "secret_key": "..."},
        region="us-east-1"
    ),
    "gcp": CloudProvider(
        provider="gcp",
        credentials={"project_id": "...", "service_account": "..."},
        region="us-central1"
    ),
    "azure": CloudProvider(
        provider="azure",
        credentials={"subscription_id": "...", "tenant_id": "..."},
        region="eastus"
    )
}

# Use specific provider
aws_storage = await providers["aws"].get_storage("my-bucket")
gcp_compute = await providers["gcp"].get_compute()
azure_secrets = await providers["azure"].get_secrets()
```

### Advanced Features

#### Multipart Upload

```python
# Upload large file with multipart
async with storage.multipart_upload("large-file.zip") as upload:
    for chunk in read_file_chunks("large-file.zip"):
        await upload.upload_part(chunk)
```

#### Lifecycle Policies

```python
# Set lifecycle policy
await storage.set_lifecycle_policy([
    {
        "id": "archive-old-logs",
        "prefix": "logs/",
        "transitions": [
            {"days": 30, "storage_class": "GLACIER"}
        ],
        "expiration": {"days": 365}
    }
])
```

#### Cross-Region Replication

```python
# Enable replication
await storage.enable_replication(
    destination_bucket="backup-bucket",
    destination_region="us-west-2"
)
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev,aws,gcp,azure]"

# Run tests
pytest

# Run tests for specific provider
pytest -k aws
pytest -k gcp
pytest -k azure
```