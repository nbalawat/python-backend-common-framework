# Commons Cloud

Unified multi-cloud abstractions providing consistent APIs across AWS, Google Cloud Platform, and Microsoft Azure. Simplify cloud operations with provider-agnostic interfaces for storage, compute, secrets, and more.

## Installation

```bash
pip install commons-cloud
```

## Features

- **Storage**: Unified object storage API (S3, GCS, Azure Blob)
- **Compute**: Virtual machine management (EC2, Compute Engine, Azure VMs)
- **Secrets**: Secret management (AWS Secrets Manager, GCP Secret Manager, Azure Key Vault)
- **Messaging**: Queue and pub/sub services (SQS, Pub/Sub, Service Bus)
- **DNS**: Domain name management (Route 53, Cloud DNS, Azure DNS)
- **CDN**: Content delivery networks (CloudFront, Cloud CDN, Azure CDN)
- **Monitoring**: Logging and metrics (CloudWatch, Cloud Monitoring, Azure Monitor)
- **Load Balancing**: Traffic distribution (ALB, Cloud Load Balancer, Azure Load Balancer)

## Quick Start

```python
from commons_cloud import CloudProvider, StorageClient, SecretManager

# Initialize cloud provider
provider = CloudProvider("aws")  # or "gcp", "azure"

# Storage operations
storage = StorageClient("my-bucket")
await storage.upload("file.txt", b"Hello Cloud!")
content = await storage.download("file.txt")

# Secrets management
secrets = SecretManager()
await secrets.create_secret("api-key", "secret-value")
api_key = await secrets.get_secret("api-key")

print(f"Downloaded: {content}")
print(f"API Key: {api_key}")
```

## Detailed Usage Examples

### Cloud Storage Operations

#### Basic Storage Operations
```python
import asyncio
from commons_cloud.storage import StorageClient, StorageConfig
from commons_cloud.providers import AWSProvider, GCPProvider, AzureProvider
from datetime import datetime, timedelta
import json

# Configure storage clients for different providers
configs = {
    "aws": StorageConfig(
        provider="aws",
        bucket="my-aws-bucket",
        region="us-east-1",
        credentials={
            "access_key_id": "AKIA...",
            "secret_access_key": "..."
        }
    ),
    "gcp": StorageConfig(
        provider="gcp",
        bucket="my-gcp-bucket",
        region="us-central1",
        credentials={
            "project_id": "my-project",
            "service_account_path": "/path/to/service-account.json"
        }
    ),
    "azure": StorageConfig(
        provider="azure",
        bucket="my-azure-container",
        region="eastus",
        credentials={
            "account_name": "mystorageaccount",
            "account_key": "..."
        }
    )
}

async def demonstrate_storage_operations():
    """Demonstrate storage operations across providers."""
    
    for provider_name, config in configs.items():
        print(f"\n=== {provider_name.upper()} Storage Operations ===")
        
        storage = StorageClient(config)
        
        # Upload different types of content
        test_data = {
            "text_file.txt": b"Hello from Commons Cloud!",
            "json_data.json": json.dumps({
                "timestamp": datetime.now().isoformat(),
                "provider": provider_name,
                "data": [1, 2, 3, 4, 5]
            }).encode(),
            "binary_data.bin": bytes(range(256))
        }
        
        # Upload files with metadata
        for filename, content in test_data.items():
            metadata = {
                "uploaded_by": "commons-cloud-demo",
                "content_type": "text/plain" if filename.endswith(".txt") else "application/octet-stream",
                "upload_time": datetime.now().isoformat()
            }
            
            await storage.upload(
                key=f"demo/{filename}",
                data=content,
                metadata=metadata,
                content_type=metadata["content_type"]
            )
            print(f"✓ Uploaded {filename} ({len(content)} bytes)")
        
        # List objects with pagination
        print("\nListing objects:")
        async for page in storage.list_objects_paginated(prefix="demo/", page_size=10):
            for obj in page.objects:
                print(f"  - {obj.key} ({obj.size} bytes, modified: {obj.last_modified})")
        
        # Download and verify content
        print("\nDownloading and verifying:")
        for filename in test_data.keys():
            key = f"demo/{filename}"
            
            # Download with metadata
            download_result = await storage.download_with_metadata(key)
            content = download_result.data
            metadata = download_result.metadata
            
            print(f"  - {filename}: {len(content)} bytes")
            print(f"    Metadata: {metadata.get('uploaded_by', 'N/A')}")
            
            # Verify content matches
            assert content == test_data[filename], f"Content mismatch for {filename}"
        
        # Generate signed URLs
        print("\nGenerating signed URLs:")
        for filename in ["text_file.txt", "json_data.json"]:
            key = f"demo/{filename}"
            
            # Read URL (1 hour expiry)
            read_url = await storage.create_signed_url(
                key=key,
                expires_in=3600,
                method="GET"
            )
            
            # Write URL (30 minutes expiry)
            write_url = await storage.create_signed_url(
                key=f"uploads/{filename}",
                expires_in=1800,
                method="PUT",
                content_type="text/plain"
            )
            
            print(f"  - Read URL for {filename}: {read_url[:50]}...")
            print(f"  - Write URL for uploads/{filename}: {write_url[:50]}...")
        
        # Copy and move operations
        print("\nCopy and move operations:")
        source_key = "demo/text_file.txt"
        copy_key = "backup/text_file_copy.txt"
        move_key = "archive/text_file_moved.txt"
        
        # Copy file
        await storage.copy(source_key, copy_key)
        print(f"✓ Copied {source_key} to {copy_key}")
        
        # Move file (copy + delete)
        await storage.move(copy_key, move_key)
        print(f"✓ Moved {copy_key} to {move_key}")
        
        # Verify copy/move worked
        exists_source = await storage.exists(source_key)
        exists_copy = await storage.exists(copy_key)
        exists_moved = await storage.exists(move_key)
        
        print(f"  Source exists: {exists_source}")
        print(f"  Copy exists: {exists_copy}")
        print(f"  Moved exists: {exists_moved}")
        
        # Cleanup
        print("\nCleaning up:")
        keys_to_delete = []
        async for page in storage.list_objects_paginated(prefix="demo/"):
            keys_to_delete.extend([obj.key for obj in page.objects])
        
        if keys_to_delete:
            await storage.delete_many(keys_to_delete)
            print(f"✓ Deleted {len(keys_to_delete)} objects")
        
        await storage.close()

# Run the demonstration
asyncio.run(demonstrate_storage_operations())
```

#### Advanced Storage Features
```python
from commons_cloud.storage import (
    StorageClient, MultipartUpload, LifecyclePolicy, 
    ReplicationConfig, VersioningConfig
)
import aiofiles
import hashlib
from pathlib import Path

async def advanced_storage_features():
    """Demonstrate advanced storage features."""
    
    storage = StorageClient(configs["aws"])  # Using AWS config from above
    
    # 1. Multipart Upload for Large Files
    print("=== Multipart Upload ===")
    
    # Create a large test file (simulated)
    large_file_path = "large_test_file.dat"
    chunk_size = 5 * 1024 * 1024  # 5MB chunks
    total_size = 50 * 1024 * 1024  # 50MB total
    
    # Simulate multipart upload
    upload_key = "uploads/large_file.dat"
    
    async with storage.multipart_upload(upload_key) as upload:
        for i in range(0, total_size, chunk_size):
            # Create chunk data (simulated)
            chunk_data = b'X' * min(chunk_size, total_size - i)
            part_number = (i // chunk_size) + 1
            
            etag = await upload.upload_part(part_number, chunk_data)
            print(f"✓ Uploaded part {part_number}, ETag: {etag}")
    
    print(f"✓ Completed multipart upload for {upload_key}")
    
    # 2. Lifecycle Management
    print("\n=== Lifecycle Policies ===")
    
    lifecycle_policy = LifecyclePolicy(
        rules=[
            {
                "id": "archive-logs",
                "status": "Enabled",
                "filter": {"prefix": "logs/"},
                "transitions": [
                    {
                        "days": 30,
                        "storage_class": "STANDARD_IA"  # Infrequent Access
                    },
                    {
                        "days": 90,
                        "storage_class": "GLACIER"
                    },
                    {
                        "days": 365,
                        "storage_class": "DEEP_ARCHIVE"
                    }
                ],
                "expiration": {"days": 2555}  # 7 years
            },
            {
                "id": "cleanup-temp",
                "status": "Enabled",
                "filter": {"prefix": "temp/"},
                "expiration": {"days": 7}  # Delete after 7 days
            },
            {
                "id": "abort-incomplete-uploads",
                "status": "Enabled",
                "abort_incomplete_multipart_upload": {"days": 1}
            }
        ]
    )
    
    await storage.set_lifecycle_policy(lifecycle_policy)
    print("✓ Applied lifecycle policy")
    
    # 3. Versioning and Object Locking
    print("\n=== Versioning Configuration ===")
    
    versioning_config = VersioningConfig(
        status="Enabled"  # or "Suspended"
    )
    
    await storage.set_versioning(versioning_config)
    print("✓ Enabled versioning")
    
    # Upload multiple versions of the same file
    test_key = "versioned/config.json"
    
    for version in range(1, 4):
        config_data = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "settings": {"debug": version % 2 == 0}
        }
        
        await storage.upload(
            key=test_key,
            data=json.dumps(config_data, indent=2).encode(),
            content_type="application/json"
        )
        print(f"✓ Uploaded version {version} of {test_key}")
    
    # List object versions
    versions = await storage.list_object_versions(test_key)
    print(f"Found {len(versions)} versions:")
    for version in versions:
        print(f"  - Version ID: {version.version_id}, Modified: {version.last_modified}")
    
    # 4. Cross-Region Replication
    print("\n=== Cross-Region Replication ===")
    
    replication_config = ReplicationConfig(
        role_arn="arn:aws:iam::123456789012:role/replication-role",
        rules=[
            {
                "id": "backup-critical-data",
                "status": "Enabled",
                "priority": 1,
                "filter": {"prefix": "critical/"},
                "destination": {
                    "bucket": "my-backup-bucket",
                    "region": "us-west-2",
                    "storage_class": "STANDARD_IA"
                }
            }
        ]
    )
    
    await storage.set_replication(replication_config)
    print("✓ Configured cross-region replication")
    
    # 5. Server-Side Encryption
    print("\n=== Server-Side Encryption ===")
    
    # Upload with different encryption options
    encryption_examples = [
        {
            "key": "encrypted/aes256.txt",
            "data": b"AES256 encrypted content",
            "encryption": {"method": "AES256"}
        },
        {
            "key": "encrypted/kms.txt",
            "data": b"KMS encrypted content",
            "encryption": {
                "method": "aws:kms",
                "kms_key_id": "arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012"
            }
        }
    ]
    
    for example in encryption_examples:
        await storage.upload(
            key=example["key"],
            data=example["data"],
            encryption=example["encryption"]
        )
        print(f"✓ Uploaded {example['key']} with {example['encryption']['method']} encryption")
    
    # 6. Event Notifications
    print("\n=== Event Notifications ===")
    
    notification_config = {
        "lambda_configurations": [
            {
                "id": "process-uploads",
                "lambda_function_arn": "arn:aws:lambda:us-east-1:123456789012:function:ProcessUploads",
                "events": ["s3:ObjectCreated:*"],
                "filter": {
                    "key": {
                        "filter_rules": [
                            {"name": "prefix", "value": "uploads/"},
                            {"name": "suffix", "value": ".jpg"}
                        ]
                    }
                }
            }
        ],
        "topic_configurations": [
            {
                "id": "notify-deletions",
                "topic_arn": "arn:aws:sns:us-east-1:123456789012:object-deletions",
                "events": ["s3:ObjectRemoved:*"]
            }
        ]
    }
    
    await storage.set_notification_configuration(notification_config)
    print("✓ Configured event notifications")
    
    await storage.close()

# Advanced batch operations
async def batch_storage_operations():
    """Demonstrate batch operations for efficiency."""
    
    storage = StorageClient(configs["aws"])
    
    print("=== Batch Operations ===")
    
    # Batch upload
    files_to_upload = [
        {"key": f"batch/file_{i:03d}.txt", "data": f"Content for file {i}".encode()}
        for i in range(100)
    ]
    
    # Upload in batches of 10
    batch_size = 10
    for i in range(0, len(files_to_upload), batch_size):
        batch = files_to_upload[i:i + batch_size]
        
        # Upload batch concurrently
        upload_tasks = [
            storage.upload(item["key"], item["data"]) 
            for item in batch
        ]
        
        await asyncio.gather(*upload_tasks)
        print(f"✓ Uploaded batch {i//batch_size + 1} ({len(batch)} files)")
    
    # Batch download
    keys_to_download = [f"batch/file_{i:03d}.txt" for i in range(0, 20)]
    
    download_tasks = [storage.download(key) for key in keys_to_download]
    downloaded_contents = await asyncio.gather(*download_tasks)
    
    print(f"✓ Downloaded {len(downloaded_contents)} files in batch")
    
    # Batch delete
    keys_to_delete = [item["key"] for item in files_to_upload]
    await storage.delete_many(keys_to_delete)
    print(f"✓ Deleted {len(keys_to_delete)} files in batch")
    
    await storage.close()

# Run advanced features
asyncio.run(advanced_storage_features())
asyncio.run(batch_storage_operations())
```

### Cloud Compute Management

#### Virtual Machine Operations
```python
import asyncio
from commons_cloud.compute import ComputeClient, InstanceConfig, NetworkConfig
from commons_cloud.compute.types import InstanceState, InstanceType
from datetime import datetime, timedelta
import time

# Configure compute clients for different providers
async def demonstrate_compute_operations():
    """Demonstrate VM management across cloud providers."""
    
    # AWS EC2 Configuration
    aws_compute = ComputeClient(
        provider="aws",
        region="us-east-1",
        credentials={
            "access_key_id": "AKIA...",
            "secret_access_key": "..."
        }
    )
    
    # GCP Compute Engine Configuration
    gcp_compute = ComputeClient(
        provider="gcp",
        project_id="my-project",
        zone="us-central1-a",
        credentials={
            "service_account_path": "/path/to/service-account.json"
        }
    )
    
    # Azure Virtual Machines Configuration
    azure_compute = ComputeClient(
        provider="azure",
        subscription_id="12345678-1234-1234-1234-123456789012",
        region="eastus",
        credentials={
            "tenant_id": "...",
            "client_id": "...",
            "client_secret": "..."
        }
    )
    
    providers = {
        "AWS": aws_compute,
        "GCP": gcp_compute,
        "Azure": azure_compute
    }
    
    for provider_name, compute in providers.items():
        print(f"\n=== {provider_name} Compute Operations ===")
        
        # 1. List existing instances
        print("\nExisting instances:")
        instances = await compute.list_instances()
        
        for instance in instances:
            print(f"  - {instance.name} ({instance.id})")
            print(f"    State: {instance.state}")
            print(f"    Type: {instance.instance_type}")
            print(f"    IP: {instance.public_ip or 'N/A'}")
            print(f"    Created: {instance.created_at}")
        
        # 2. Create new instance
        print("\nCreating new instance...")
        
        # Network configuration
        network_config = NetworkConfig(
            vpc_id="vpc-12345678" if provider_name == "AWS" else None,
            subnet_id="subnet-12345678" if provider_name == "AWS" else None,
            security_groups=["sg-web-servers"] if provider_name == "AWS" else [],
            assign_public_ip=True,
            firewall_rules=[
                {
                    "name": "allow-http",
                    "protocol": "tcp",
                    "ports": [80, 443],
                    "source_ranges": ["0.0.0.0/0"]
                },
                {
                    "name": "allow-ssh",
                    "protocol": "tcp",
                    "ports": [22],
                    "source_ranges": ["10.0.0.0/8"]
                }
            ]
        )
        
        # Instance configuration
        instance_config = InstanceConfig(
            name=f"web-server-{int(time.time())}",
            image_id={
                "AWS": "ami-0abcdef1234567890",  # Amazon Linux 2
                "GCP": "projects/ubuntu-os-cloud/global/images/family/ubuntu-2004-lts",
                "Azure": "Canonical:0001-com-ubuntu-server-focal:20_04-lts-gen2:latest"
            }[provider_name],
            instance_type={
                "AWS": "t3.micro",
                "GCP": "e2-micro",
                "Azure": "Standard_B1s"
            }[provider_name],
            key_name="my-keypair",
            network_config=network_config,
            user_data="""
#!/bin/bash
apt-get update
apt-get install -y nginx
systemctl start nginx
systemctl enable nginx

# Create simple index page
echo "<h1>Hello from Commons Cloud!</h1>" > /var/www/html/index.html
echo "<p>Provider: {}</p>" >> /var/www/html/index.html
echo "<p>Instance ID: $(curl -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo 'N/A')</p>" >> /var/www/html/index.html
echo "<p>Started: $(date)</p>" >> /var/www/html/index.html
""".format(provider_name),
            tags={
                "Environment": "demo",
                "Project": "commons-cloud",
                "Owner": "demo-user",
                "CreatedBy": "commons-cloud-demo"
            },
            disk_size_gb=20,
            enable_monitoring=True,
            backup_enabled=True
        )
        
        # Create instance
        new_instance = await compute.create_instance(instance_config)
        print(f"✓ Created instance: {new_instance.name} ({new_instance.id})")
        
        # Wait for instance to be running
        print("Waiting for instance to start...")
        instance = await compute.wait_for_state(
            new_instance.id, 
            InstanceState.RUNNING, 
            timeout=300
        )
        print(f"✓ Instance is now {instance.state}")
        print(f"  Public IP: {instance.public_ip}")
        print(f"  Private IP: {instance.private_ip}")
        
        # 3. Instance management operations
        print("\nInstance management:")
        
        # Get instance details
        instance_details = await compute.get_instance(new_instance.id)
        print(f"CPU Utilization: {instance_details.cpu_utilization}%")
        print(f"Memory Usage: {instance_details.memory_usage}%")
        print(f"Network In: {instance_details.network_in_bytes} bytes")
        print(f"Network Out: {instance_details.network_out_bytes} bytes")
        
        # Resize instance (if supported)
        if provider_name in ["AWS", "GCP"]:
            print("\nResizing instance...")
            new_type = {
                "AWS": "t3.small",
                "GCP": "e2-small"
            }[provider_name]
            
            await compute.resize_instance(new_instance.id, new_type)
            print(f"✓ Resized to {new_type}")
        
        # Create snapshot/backup
        print("\nCreating snapshot...")
        snapshot = await compute.create_snapshot(
            instance_id=new_instance.id,
            name=f"snapshot-{new_instance.name}-{int(time.time())}",
            description="Demo snapshot created by commons-cloud"
        )
        print(f"✓ Created snapshot: {snapshot.id}")
        
        # 4. Load balancing setup
        if provider_name == "AWS":  # Example for AWS
            print("\nSetting up load balancer...")
            
            lb_config = {
                "name": f"demo-lb-{int(time.time())}",
                "scheme": "internet-facing",
                "type": "application",
                "subnets": ["subnet-12345678", "subnet-87654321"],
                "security_groups": ["sg-lb-security"],
                "listeners": [
                    {
                        "port": 80,
                        "protocol": "HTTP",
                        "target_group": {
                            "name": f"demo-targets-{int(time.time())}",
                            "port": 80,
                            "protocol": "HTTP",
                            "health_check": {
                                "path": "/",
                                "healthy_threshold": 2,
                                "unhealthy_threshold": 3,
                                "timeout": 5,
                                "interval": 30
                            }
                        }
                    }
                ]
            }
            
            load_balancer = await compute.create_load_balancer(lb_config)
            print(f"✓ Created load balancer: {load_balancer.dns_name}")
            
            # Register instance with load balancer
            await compute.register_targets(
                load_balancer.target_group_arn,
                [new_instance.id]
            )
            print(f"✓ Registered instance with load balancer")
        
        # 5. Auto Scaling Group (AWS example)
        if provider_name == "AWS":
            print("\nCreating Auto Scaling Group...")
            
            # First create launch template
            launch_template = await compute.create_launch_template({
                "name": f"demo-template-{int(time.time())}",
                "image_id": "ami-0abcdef1234567890",
                "instance_type": "t3.micro",
                "key_name": "my-keypair",
                "security_groups": ["sg-web-servers"],
                "user_data": instance_config.user_data,
                "tags": instance_config.tags
            })
            
            # Create Auto Scaling Group
            asg_config = {
                "name": f"demo-asg-{int(time.time())}",
                "launch_template_id": launch_template.id,
                "min_size": 1,
                "max_size": 5,
                "desired_capacity": 2,
                "vpc_zone_identifiers": ["subnet-12345678", "subnet-87654321"],
                "target_group_arns": [load_balancer.target_group_arn],
                "health_check_type": "ELB",
                "health_check_grace_period": 300,
                "policies": [
                    {
                        "name": "scale-up",
                        "policy_type": "TargetTrackingScaling",
                        "target_value": 70.0,
                        "metric_type": "ASGAverageCPUUtilization"
                    }
                ]
            }
            
            auto_scaling_group = await compute.create_auto_scaling_group(asg_config)
            print(f"✓ Created Auto Scaling Group: {auto_scaling_group.name}")
        
        # 6. Monitoring and alerting
        print("\nSetting up monitoring...")
        
        # Enable detailed monitoring
        await compute.enable_detailed_monitoring(new_instance.id)
        print("✓ Enabled detailed monitoring")
        
        # Create CloudWatch alarms (AWS example)
        if provider_name == "AWS":
            alarm_configs = [
                {
                    "name": f"high-cpu-{new_instance.id}",
                    "description": "High CPU utilization alert",
                    "metric_name": "CPUUtilization",
                    "namespace": "AWS/EC2",
                    "statistic": "Average",
                    "period": 300,
                    "evaluation_periods": 2,
                    "threshold": 80.0,
                    "comparison_operator": "GreaterThanThreshold",
                    "dimensions": {"InstanceId": new_instance.id},
                    "alarm_actions": ["arn:aws:sns:us-east-1:123456789012:high-cpu-alerts"]
                },
                {
                    "name": f"low-disk-space-{new_instance.id}",
                    "description": "Low disk space alert",
                    "metric_name": "DiskSpaceUtilization",
                    "namespace": "CWAgent",
                    "statistic": "Average",
                    "period": 300,
                    "evaluation_periods": 1,
                    "threshold": 90.0,
                    "comparison_operator": "GreaterThanThreshold",
                    "dimensions": {"InstanceId": new_instance.id}
                }
            ]
            
            for alarm_config in alarm_configs:
                alarm = await compute.create_alarm(alarm_config)
                print(f"✓ Created alarm: {alarm.name}")
        
        # 7. Instance lifecycle management
        print("\nTesting instance lifecycle...")
        
        # Stop instance
        await compute.stop_instance(new_instance.id)
        print("✓ Stopped instance")
        
        # Wait for stopped state
        await compute.wait_for_state(new_instance.id, InstanceState.STOPPED, timeout=120)
        print("✓ Instance is stopped")
        
        # Start instance again
        await compute.start_instance(new_instance.id)
        print("✓ Started instance")
        
        # Wait for running state
        await compute.wait_for_state(new_instance.id, InstanceState.RUNNING, timeout=120)
        print("✓ Instance is running again")
        
        # Reboot instance
        await compute.reboot_instance(new_instance.id)
        print("✓ Rebooted instance")
        
        # 8. Cleanup (commented out for safety)
        # print("\nCleaning up resources...")
        # await compute.terminate_instance(new_instance.id)
        # print("✓ Terminated instance")
        
        await compute.close()

# Container and Serverless Compute
async def demonstrate_container_operations():
    """Demonstrate container and serverless operations."""
    
    print("=== Container and Serverless Operations ===")
    
    # Kubernetes cluster management (GKE example)
    from commons_cloud.compute import KubernetesClient
    
    k8s_client = KubernetesClient(
        provider="gcp",
        project_id="my-project",
        region="us-central1"
    )
    
    # Create GKE cluster
    cluster_config = {
        "name": "demo-cluster",
        "node_pools": [
            {
                "name": "default-pool",
                "node_count": 3,
                "machine_type": "e2-medium",
                "disk_size_gb": 100,
                "oauth_scopes": [
                    "https://www.googleapis.com/auth/cloud-platform"
                ]
            }
        ],
        "network": "default",
        "subnetwork": "default",
        "enable_autoscaling": True,
        "min_node_count": 1,
        "max_node_count": 10
    }
    
    cluster = await k8s_client.create_cluster(cluster_config)
    print(f"✓ Created Kubernetes cluster: {cluster.name}")
    
    # Deploy application to cluster
    deployment_config = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "demo-app"},
        "spec": {
            "replicas": 3,
            "selector": {"matchLabels": {"app": "demo"}},
            "template": {
                "metadata": {"labels": {"app": "demo"}},
                "spec": {
                    "containers": [
                        {
                            "name": "demo-container",
                            "image": "nginx:latest",
                            "ports": [{"containerPort": 80}]
                        }
                    ]
                }
            }
        }
    }
    
    await k8s_client.apply_manifest(deployment_config)
    print("✓ Deployed application to cluster")
    
    # Serverless functions (AWS Lambda example)
    from commons_cloud.compute import ServerlessClient
    
    serverless = ServerlessClient(provider="aws", region="us-east-1")
    
    function_config = {
        "name": "demo-function",
        "runtime": "python3.9",
        "handler": "lambda_function.lambda_handler",
        "code": {
            "zip_file": b"""\nimport json

def lambda_handler(event, context):
    return {
        'statusCode': 200,
        'body': json.dumps({'message': 'Hello from Commons Cloud Lambda!'})
    }
"""
        },
        "environment": {
            "variables": {
                "ENV": "demo",
                "LOG_LEVEL": "INFO"
            }
        },
        "timeout": 30,
        "memory_size": 128
    }
    
    function = await serverless.create_function(function_config)
    print(f"✓ Created serverless function: {function.arn}")
    
    # Test function
    response = await serverless.invoke_function(
        function.name,
        payload={"test": "data"}
    )
    print(f"✓ Function response: {response['body']}")
    
    await k8s_client.close()
    await serverless.close()

# Run compute demonstrations
asyncio.run(demonstrate_compute_operations())
asyncio.run(demonstrate_container_operations())
```

### Secrets Management

#### Multi-Provider Secret Operations
```python
import asyncio
import json
from commons_cloud.secrets import SecretManager, RotationConfig, AccessPolicy
from datetime import datetime, timedelta
import base64

async def demonstrate_secrets_management():
    """Demonstrate secrets management across cloud providers."""
    
    # Configure secret managers for different providers
    secret_managers = {
        "aws": SecretManager(
            provider="aws",
            region="us-east-1",
            credentials={
                "access_key_id": "AKIA...",
                "secret_access_key": "..."
            }
        ),
        "gcp": SecretManager(
            provider="gcp",
            project_id="my-project",
            credentials={
                "service_account_path": "/path/to/service-account.json"
            }
        ),
        "azure": SecretManager(
            provider="azure",
            vault_url="https://my-vault.vault.azure.net/",
            credentials={
                "tenant_id": "...",
                "client_id": "...",
                "client_secret": "..."
            }
        )
    }
    
    for provider_name, secrets in secret_managers.items():
        print(f"\n=== {provider_name.upper()} Secrets Management ===")
        
        # 1. Create different types of secrets
        secret_examples = [
            {
                "name": "database-connection",
                "value": json.dumps({
                    "host": "db.example.com",
                    "port": 5432,
                    "database": "production",
                    "username": "app_user",
                    "password": "super_secure_password_123!"
                }),
                "description": "Production database connection details",
                "tags": {"Environment": "production", "Type": "database"},
                "content_type": "application/json"
            },
            {
                "name": "api-keys",
                "value": json.dumps({
                    "stripe_key": "sk_live_...",
                    "sendgrid_key": "SG.abc123...",
                    "google_maps_key": "AIza..."
                }),
                "description": "External API keys",
                "tags": {"Environment": "production", "Type": "api-keys"},
                "content_type": "application/json"
            },
            {
                "name": "ssl-certificate",
                "value": "-----BEGIN CERTIFICATE-----\nMIIC...\n-----END CERTIFICATE-----",
                "description": "SSL certificate for production domain",
                "tags": {"Environment": "production", "Type": "certificate"},
                "content_type": "text/plain"
            },
            {
                "name": "jwt-signing-key",
                "value": base64.b64encode(b"super_secret_jwt_key_256_bits_long!").decode(),
                "description": "JWT signing key for authentication",
                "tags": {"Environment": "production", "Type": "encryption"},
                "content_type": "text/plain"
            }
        ]
        
        print("\nCreating secrets:")
        for secret_data in secret_examples:
            try:
                secret = await secrets.create_secret(
                    name=secret_data["name"],
                    value=secret_data["value"],
                    description=secret_data["description"],
                    tags=secret_data["tags"],
                    content_type=secret_data.get("content_type")
                )
                print(f"✓ Created secret: {secret.name} (ARN: {secret.arn})")
            except Exception as e:
                print(f"⚠ Failed to create {secret_data['name']}: {e}")
        
        # 2. Retrieve and verify secrets
        print("\nRetrieving secrets:")
        for secret_data in secret_examples:
            try:
                retrieved_secret = await secrets.get_secret(
                    name=secret_data["name"],
                    version="AWSCURRENT"  # or version ID
                )
                
                # Verify content matches
                if retrieved_secret.value == secret_data["value"]:
                    print(f"✓ Retrieved and verified: {secret_data['name']}")
                else:
                    print(f"⚠ Content mismatch for: {secret_data['name']}")
                
                # Show metadata
                print(f"  Created: {retrieved_secret.created_date}")
                print(f"  Modified: {retrieved_secret.last_changed_date}")
                print(f"  Version: {retrieved_secret.version_id}")
                
            except Exception as e:
                print(f"⚠ Failed to retrieve {secret_data['name']}: {e}")
        
        # 3. Secret versioning
        print("\nTesting secret versioning:")
        
        # Update database password
        new_db_config = json.dumps({
            "host": "db.example.com",
            "port": 5432,
            "database": "production",
            "username": "app_user",
            "password": "new_super_secure_password_456!"
        })
        
        try:
            await secrets.update_secret(
                name="database-connection",
                value=new_db_config,
                description="Updated production database connection"
            )
            print("✓ Updated database-connection secret")
            
            # List all versions
            versions = await secrets.list_secret_versions("database-connection")
            print(f"  Total versions: {len(versions)}")
            
            for version in versions:
                print(f"    Version {version.version_id}: Created {version.created_date}")
                
        except Exception as e:
            print(f"⚠ Failed to update secret: {e}")
        
        # 4. Automatic rotation setup
        if provider_name == "aws":  # AWS Secrets Manager supports automatic rotation
            print("\nSetting up automatic rotation:")
            
            try:
                rotation_config = RotationConfig(
                    rotation_lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:rotate-db-password",
                    rotation_rules={
                        "automatically_after_days": 30  # Rotate every 30 days
                    }
                )
                
                await secrets.enable_rotation(
                    name="database-connection",
                    config=rotation_config
                )
                print("✓ Enabled automatic rotation for database-connection")
                
            except Exception as e:
                print(f"⚠ Failed to enable rotation: {e}")
        
        # 5. Access policies and permissions
        print("\nManaging access policies:")
        
        try:
            # Create access policy for application role
            access_policy = AccessPolicy({
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {
                            "AWS": "arn:aws:iam::123456789012:role/application-role"
                        },
                        "Action": "secretsmanager:GetSecretValue",
                        "Resource": "*",
                        "Condition": {
                            "StringEquals": {
                                "secretsmanager:ResourceTag/Environment": "production"
                            }
                        }
                    }
                ]
            })
            
            await secrets.set_resource_policy(
                name="database-connection",
                policy=access_policy
            )
            print("✓ Set resource policy for database-connection")
            
        except Exception as e:
            print(f"⚠ Failed to set resource policy: {e}")
        
        # 6. Bulk operations
        print("\nBulk secret operations:")
        
        # Create multiple environment-specific secrets
        environments = ["development", "staging", "production"]
        
        for env in environments:
            secret_name = f"app-config-{env}"
            config_value = json.dumps({
                "database_url": f"postgres://user:pass@{env}-db.example.com/app",
                "redis_url": f"redis://{env}-cache.example.com:6379",
                "api_base_url": f"https://api-{env}.example.com",
                "debug": env != "production",
                "log_level": "INFO" if env == "production" else "DEBUG"
            })
            
            try:
                await secrets.create_secret(
                    name=secret_name,
                    value=config_value,
                    description=f"Application configuration for {env} environment",
                    tags={
                        "Environment": env,
                        "Type": "application-config",
                        "ManagedBy": "commons-cloud"
                    }
                )
                print(f"✓ Created {secret_name}")
                
            except Exception as e:
                print(f"⚠ Failed to create {secret_name}: {e}")
        
        # 7. Secret monitoring and auditing
        print("\nSecret usage monitoring:")
        
        try:
            # Get secret access logs (AWS CloudTrail example)
            if provider_name == "aws":
                access_logs = await secrets.get_access_logs(
                    secret_name="database-connection",
                    start_time=datetime.now() - timedelta(days=7),
                    end_time=datetime.now()
                )
                
                print(f"Access events in last 7 days: {len(access_logs)}")
                for log in access_logs[:5]:  # Show first 5
                    print(f"  {log.timestamp}: {log.event_name} by {log.user_identity}")
            
            # List secrets by tags
            production_secrets = await secrets.list_secrets(
                filters={
                    "tag-key": "Environment",
                    "tag-value": "production"
                }
            )
            
            print(f"\nProduction secrets: {len(production_secrets)}")
            for secret in production_secrets:
                print(f"  - {secret.name}: {secret.description}")
                
        except Exception as e:
            print(f"⚠ Failed to get monitoring data: {e}")
        
        # 8. Secret backup and disaster recovery
        print("\nBackup and disaster recovery:")
        
        try:
            # Export secrets for backup (be careful with this in production!)
            backup_data = await secrets.export_secrets(
                filters={
                    "tag-key": "Environment",
                    "tag-value": "production"
                },
                include_values=False  # Only metadata for security
            )
            
            print(f"✓ Exported {len(backup_data)} secrets (metadata only)")
            
            # Cross-region replication setup (AWS example)
            if provider_name == "aws":
                replication_config = {
                    "replica_regions": [
                        {
                            "region": "us-west-2",
                            "kms_key_id": "arn:aws:kms:us-west-2:123456789012:key/..."
                        },
                        {
                            "region": "eu-west-1",
                            "kms_key_id": "arn:aws:kms:eu-west-1:123456789012:key/..."
                        }
                    ]
                }
                
                await secrets.enable_replication(
                    name="database-connection",
                    config=replication_config
                )
                print("✓ Enabled cross-region replication")
                
        except Exception as e:
            print(f"⚠ Failed backup/replication setup: {e}")
        
        await secrets.close()

# Secret injection for applications
async def demonstrate_secret_injection():
    """Demonstrate secure secret injection patterns."""
    
    print("\n=== Secret Injection Patterns ===")
    
    secrets = SecretManager(provider="aws", region="us-east-1")
    
    # 1. Environment variable injection
    print("\nEnvironment variable injection:")
    
    app_secrets = await secrets.get_secrets_by_prefix("app-config-production")
    
    # Safely inject into environment
    import os
    env_mapping = {
        "DATABASE_URL": "database-connection",
        "REDIS_URL": "cache-connection",
        "API_KEY": "external-api-key"
    }
    
    for env_var, secret_name in env_mapping.items():
        try:
            secret_value = await secrets.get_secret(secret_name)
            # In production, use secure injection mechanisms
            # os.environ[env_var] = secret_value.value
            print(f"✓ Would inject {env_var} from {secret_name}")
        except Exception as e:
            print(f"⚠ Failed to inject {env_var}: {e}")
    
    # 2. Configuration file generation
    print("\nConfiguration file generation:")
    
    config_template = {
        "database": {
            "host": "${DB_HOST}",
            "port": "${DB_PORT}",
            "username": "${DB_USER}",
            "password": "${DB_PASSWORD}"
        },
        "redis": {
            "url": "${REDIS_URL}"
        },
        "external_apis": {
            "stripe_key": "${STRIPE_KEY}",
            "sendgrid_key": "${SENDGRID_KEY}"
        }
    }
    
    # Replace placeholders with actual secret values
    db_secret = json.loads(await secrets.get_secret("database-connection").value)
    api_secrets = json.loads(await secrets.get_secret("api-keys").value)
    
    resolved_config = {
        "database": {
            "host": db_secret["host"],
            "port": db_secret["port"],
            "username": db_secret["username"],
            "password": db_secret["password"]
        },
        "redis": {
            "url": "redis://cache.example.com:6379"
        },
        "external_apis": {
            "stripe_key": api_secrets["stripe_key"],
            "sendgrid_key": api_secrets["sendgrid_key"]
        }
    }
    
    print("✓ Generated secure configuration file")
    
    # 3. Runtime secret fetching with caching
    print("\nRuntime secret fetching with caching:")
    
    from commons_cloud.secrets import SecretCache
    
    secret_cache = SecretCache(
        secret_manager=secrets,
        ttl_seconds=300,  # 5 minutes
        max_size=100
    )
    
    # Cached secret access
    for i in range(3):
        start_time = time.time()
        cached_secret = await secret_cache.get_secret("database-connection")
        fetch_time = time.time() - start_time
        
        cache_status = "HIT" if i > 0 else "MISS"
        print(f"  Fetch {i+1}: {fetch_time:.3f}s ({cache_status})")
    
    await secrets.close()
    await secret_cache.close()

# Run secrets demonstrations
asyncio.run(demonstrate_secrets_management())
asyncio.run(demonstrate_secret_injection())
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