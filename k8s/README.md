# Commons K8s

Kubernetes utilities and patterns for Python applications.

## Features

- **Client Management**: Multi-cluster support with authentication handling
- **Resource Management**: CRUD operations for Kubernetes resources
- **Custom Resources**: CRD management and operator patterns
- **Argo Integration**: Workflow creation and management
- **Async Support**: Full async/await support for all operations

## Installation

```bash
# Basic installation
pip install commons-k8s

# With Argo Workflows support
pip install commons-k8s[argo]
```

## Usage

### Client Management

```python
from commons_k8s import K8sClient

# Create client with current context
client = await K8sClient.from_config()

# Create client with specific kubeconfig
client = await K8sClient.from_config(config_file="~/.kube/config", context="prod")

# Create in-cluster client
client = await K8sClient.in_cluster()

# Switch context
await client.switch_context("staging")
```

### Resource Management

```python
from commons_k8s.resources import Deployment, Service, ConfigMap

# List resources
deployments = await client.list_deployments(namespace="default")

# Get specific resource
deployment = await client.get_deployment("my-app", namespace="default")

# Create resource
deployment = Deployment(
    name="my-app",
    namespace="default",
    image="myapp:latest",
    replicas=3,
    port=8080,
)
await client.create_deployment(deployment)

# Update resource
deployment.spec.replicas = 5
await client.update_deployment(deployment)

# Patch resource
await client.patch_deployment(
    "my-app",
    namespace="default",
    patch={"spec": {"replicas": 10}}
)

# Delete resource
await client.delete_deployment("my-app", namespace="default")

# Watch resources
async for event in client.watch_deployments(namespace="default"):
    print(f"{event.type}: {event.object.metadata.name}")
```

### Custom Resources

```python
from commons_k8s.operators import CustomResource, Operator

# Define custom resource
@CustomResource(
    group="example.com",
    version="v1",
    plural="myresources",
    singular="myresource",
    kind="MyResource",
)
class MyResource:
    def __init__(self, name: str, spec: dict):
        self.name = name
        self.spec = spec

# Create operator
class MyOperator(Operator):
    async def reconcile(self, resource: MyResource, **kwargs):
        # Reconciliation logic
        logger.info(f"Reconciling {resource.name}")
        
        # Create owned resources
        deployment = self.create_deployment(resource)
        await self.client.create_deployment(deployment)
        
        # Update status
        await self.update_status(resource, {"phase": "Running"})

# Run operator
operator = MyOperator()
await operator.run()
```

### Argo Workflows

```python
from commons_k8s.argo import Workflow, DAG, Task

# Create workflow
workflow = Workflow("data-processing")

# Define DAG
dag = DAG()

# Add tasks
extract = Task(
    name="extract",
    image="extractor:latest",
    command=["python", "extract.py"],
)

transform = Task(
    name="transform",
    image="transformer:latest",
    command=["python", "transform.py"],
    dependencies=["extract"],
)

load = Task(
    name="load",
    image="loader:latest",
    command=["python", "load.py"],
    dependencies=["transform"],
)

# Build workflow
dag.add_tasks([extract, transform, load])
workflow.spec = dag.to_dict()

# Submit workflow
await client.create_workflow(workflow)

# Monitor workflow
status = await client.get_workflow_status("data-processing")
print(f"Workflow status: {status.phase}")
```

### Advanced Features

#### Resource Templates

```python
from commons_k8s.templates import ResourceTemplate

# Create template
template = ResourceTemplate("deployment.yaml.j2")

# Render with variables
deployment_yaml = template.render(
    name="my-app",
    image="myapp:latest",
    replicas=3,
    env_vars={"API_KEY": "secret"},
)

# Apply to cluster
await client.apply_yaml(deployment_yaml)
```

#### Batch Operations

```python
# Batch create
resources = [deployment1, deployment2, service1, service2]
await client.create_many(resources)

# Batch delete
await client.delete_many(
    resource_type="deployment",
    names=["app1", "app2"],
    namespace="default",
)

# Batch patch
patches = [
    {"name": "app1", "patch": {"spec": {"replicas": 5}}},
    {"name": "app2", "patch": {"spec": {"replicas": 3}}},
]
await client.patch_many("deployment", patches, namespace="default")
```

#### Health Checks

```python
from commons_k8s.health import HealthChecker

# Create health checker
checker = HealthChecker(client)

# Check deployment health
health = await checker.check_deployment("my-app", namespace="default")
print(f"Deployment health: {health.status}")
print(f"Ready replicas: {health.ready_replicas}/{health.desired_replicas}")

# Check cluster health
cluster_health = await checker.check_cluster()
for component, status in cluster_health.items():
    print(f"{component}: {status}")
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev,argo]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=commons_k8s
```