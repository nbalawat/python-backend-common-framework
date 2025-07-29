# Commons K8s

Comprehensive Kubernetes utilities and patterns for Python applications, providing async-first APIs for cluster management, resource operations, custom operators, and workflow orchestration.

## Installation

```bash
pip install commons-k8s
```

## Features

- **Multi-Cluster Management**: Connect and manage multiple Kubernetes clusters
- **Resource Operations**: Full CRUD operations for all Kubernetes resources  
- **Custom Resources & Operators**: CRD management and controller patterns
- **Workflow Orchestration**: Argo Workflows integration
- **Health Monitoring**: Comprehensive cluster and application health checks
- **Template Management**: Helm-style templating with Jinja2
- **Event Streaming**: Real-time resource event monitoring
- **Batch Operations**: Efficient bulk resource operations
- **Security**: RBAC management and security scanning

## Quick Start

```python
import asyncio
from commons_k8s import K8sClient
from commons_k8s.resources import Deployment

async def main():
    # Connect to cluster
    client = await K8sClient.from_config()
    
    # Create a simple deployment
    deployment = Deployment(
        name="hello-world",
        namespace="default",
        image="nginx:latest",
        replicas=3,
        port=80
    )
    
    # Deploy to cluster
    await client.create_deployment(deployment)
    print("Deployment created successfully!")
    
    # Check status
    status = await client.get_deployment_status("hello-world")
    print(f"Ready replicas: {status.ready_replicas}/{status.replicas}")
    
    await client.close()

asyncio.run(main())
```

## Detailed Usage Examples

### Multi-Cluster Client Management

#### Cluster Connection and Configuration
```python
import asyncio
from commons_k8s import K8sClient, ClusterConfig
from commons_k8s.auth import ServiceAccountAuth, CertificateAuth
from pathlib import Path
import yaml

async def demonstrate_cluster_management():
    """Demonstrate multi-cluster connection and management."""
    
    print("=== Kubernetes Cluster Management ===")
    
    # 1. Connection methods
    clients = {}
    
    # Method 1: Default kubeconfig (current context)
    try:
        clients["default"] = await K8sClient.from_config()
        print("✓ Connected using default kubeconfig")
    except Exception as e:
        print(f"⚠ Failed to connect with default config: {e}")
    
    # Method 2: Specific kubeconfig file and context
    try:
        clients["production"] = await K8sClient.from_config(
            config_file="~/.kube/prod-config",
            context="production-cluster"
        )
        print("✓ Connected to production cluster")
    except Exception as e:
        print(f"⚠ Failed to connect to production: {e}")
    
    # Method 3: In-cluster authentication (for pods running in K8s)
    try:
        clients["in_cluster"] = await K8sClient.in_cluster()
        print("✓ Connected using in-cluster authentication")
    except Exception as e:
        print(f"⚠ In-cluster auth not available: {e}")
    
    # Method 4: Manual cluster configuration
    cluster_config = ClusterConfig(
        server="https://k8s-api.example.com:6443",
        ca_cert_path="/path/to/ca.crt",
        auth=CertificateAuth(
            client_cert_path="/path/to/client.crt",
            client_key_path="/path/to/client.key"
        )
    )
    
    try:
        clients["manual"] = await K8sClient.from_cluster_config(cluster_config)
        print("✓ Connected using manual configuration")
    except Exception as e:
        print(f"⚠ Manual connection failed: {e}")
    
    # Method 5: Service account authentication
    sa_auth = ServiceAccountAuth(
        token_path="/var/run/secrets/kubernetes.io/serviceaccount/token",
        ca_cert_path="/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
    )
    
    service_account_config = ClusterConfig(
        server="https://kubernetes.default.svc",
        auth=sa_auth
    )
    
    try:
        clients["service_account"] = await K8sClient.from_cluster_config(service_account_config)
        print("✓ Connected using service account")
    except Exception as e:
        print(f"⚠ Service account auth failed: {e}")
    
    # 2. Cluster information and capabilities
    for name, client in clients.items():
        if client:
            print(f"\n--- {name.upper()} CLUSTER INFO ---")
            
            try:
                # Get cluster version
                version_info = await client.get_version()
                print(f"Kubernetes Version: {version_info.git_version}")
                print(f"Platform: {version_info.platform}")
                
                # Get cluster nodes
                nodes = await client.list_nodes()
                print(f"Nodes: {len(nodes)}")
                
                for node in nodes[:3]:  # Show first 3 nodes
                    print(f"  - {node.metadata.name}:")
                    print(f"    Status: {node.status.conditions[-1].type}")
                    print(f"    Kubelet: {node.status.node_info.kubelet_version}")
                    print(f"    OS: {node.status.node_info.os_image}")
                
                # Get namespaces
                namespaces = await client.list_namespaces()
                print(f"Namespaces: {len(namespaces)}")
                for ns in namespaces[:5]:  # Show first 5
                    print(f"  - {ns.metadata.name}")
                
                # Test permissions
                print("Permission checks:")
                permissions = [
                    ("pods", "list", "default"),
                    ("deployments", "create", "default"),
                    ("services", "delete", "kube-system"),
                    ("nodes", "get", "")
                ]
                
                for resource, verb, namespace in permissions:
                    try:
                        can_perform = await client.can_i(verb, resource, namespace)
                        status = "✓" if can_perform else "✗"
                        ns_info = f" in {namespace}" if namespace else " cluster-wide"
                        print(f"    {status} {verb} {resource}{ns_info}")
                    except Exception as e:
                        print(f"    ⚠ Failed to check {verb} {resource}: {e}")
                
            except Exception as e:
                print(f"⚠ Failed to get cluster info: {e}")
    
    # 3. Context switching and management
    print("\n=== Context Management ===")
    
    if "default" in clients and clients["default"]:
        client = clients["default"]
        
        try:
            # List available contexts
            contexts = await client.list_contexts()
            print(f"Available contexts: {len(contexts)}")
            
            for context in contexts:
                current = " (current)" if context.is_current else ""
                print(f"  - {context.name}: cluster={context.cluster}, user={context.user}{current}")
            
            # Switch context (if multiple available)
            if len(contexts) > 1:
                new_context = contexts[1].name  # Switch to second context
                await client.switch_context(new_context)
                print(f"✓ Switched to context: {new_context}")
                
                # Verify switch worked
                current_context = await client.get_current_context()
                print(f"Current context: {current_context.name}")
                
        except Exception as e:
            print(f"⚠ Context management failed: {e}")
    
    # 4. Multi-cluster operations
    print("\n=== Multi-Cluster Operations ===")
    
    from commons_k8s import MultiClusterClient
    
    # Create multi-cluster client
    multi_client = MultiClusterClient()
    
    # Add clusters
    for name, client in clients.items():
        if client:
            await multi_client.add_cluster(name, client)
    
    try:
        # List pods across all clusters
        all_pods = await multi_client.list_pods_all_clusters(namespace="default")
        
        print(f"Total pods across all clusters: {len(all_pods)}")
        for cluster_name, pods in all_pods.items():
            print(f"  {cluster_name}: {len(pods)} pods")
            for pod in pods[:2]:  # Show first 2 pods per cluster
                print(f"    - {pod.metadata.name} ({pod.status.phase})")
        
        # Execute command across clusters
        results = await multi_client.execute_all_clusters(
            lambda client: client.get_cluster_resource_usage()
        )
        
        print("\nResource usage across clusters:")
        for cluster_name, usage in results.items():
            if usage and not isinstance(usage, Exception):
                print(f"  {cluster_name}:")
                print(f"    CPU: {usage.cpu_usage:.1f}%")
                print(f"    Memory: {usage.memory_usage:.1f}%")
                print(f"    Storage: {usage.storage_usage:.1f}%")
        
    except Exception as e:
        print(f"⚠ Multi-cluster operations failed: {e}")
    
    # Cleanup
    for client in clients.values():
        if client:
            await client.close()
    
    await multi_client.close()

# Connection pooling and optimization
async def demonstrate_connection_optimization():
    """Demonstrate connection pooling and optimization techniques."""
    
    print("\n=== Connection Optimization ===")
    
    from commons_k8s import ConnectionPool, ClientConfig
    
    # Configure optimized client
    client_config = ClientConfig(
        connection_pool_size=20,
        max_connections_per_host=10,
        request_timeout=30.0,
        connection_timeout=10.0,
        read_timeout=60.0,
        retry_attempts=3,
        retry_backoff_factor=2.0,
        enable_compression=True,
        enable_keepalive=True,
        keepalive_timeout=30.0
    )
    
    client = await K8sClient.from_config(config=client_config)
    
    # Demonstrate efficient batch operations
    print("\nPerforming optimized batch operations...")
    
    import time
    
    # Batch resource fetching
    start_time = time.time()
    
    # Instead of individual API calls
    batch_tasks = [
        client.list_pods(namespace="default"),
        client.list_services(namespace="default"),
        client.list_deployments(namespace="default"),
        client.list_configmaps(namespace="default"),
        client.list_secrets(namespace="default")
    ]
    
    results = await asyncio.gather(*batch_tasks)
    batch_time = time.time() - start_time
    
    pods, services, deployments, configmaps, secrets = results
    
    print(f"Batch fetch completed in {batch_time:.2f}s:")
    print(f"  - Pods: {len(pods)}")
    print(f"  - Services: {len(services)}")
    print(f"  - Deployments: {len(deployments)}")
    print(f"  - ConfigMaps: {len(configmaps)}")
    print(f"  - Secrets: {len(secrets)}")
    
    # Connection pooling demonstration
    pool = ConnectionPool(max_size=50, max_connections_per_host=10)
    
    # Multiple clients sharing the same pool
    pooled_clients = []
    for i in range(5):
        pooled_client = await K8sClient.from_config(
            connection_pool=pool
        )
        pooled_clients.append(pooled_client)
    
    print(f"✓ Created {len(pooled_clients)} clients sharing connection pool")
    
    # Demonstrate concurrent operations
    concurrent_tasks = []
    for i, client in enumerate(pooled_clients):
        task = client.list_pods(namespace="default")
        concurrent_tasks.append(task)
    
    start_time = time.time()
    concurrent_results = await asyncio.gather(*concurrent_tasks)
    concurrent_time = time.time() - start_time
    
    print(f"Concurrent operations completed in {concurrent_time:.2f}s")
    print(f"Total pods across all clients: {sum(len(pods) for pods in concurrent_results)}")
    
    # Cleanup
    for client in pooled_clients:
        await client.close()
    
    await pool.close()
    await client.close()

# Run demonstrations
asyncio.run(demonstrate_cluster_management())
asyncio.run(demonstrate_connection_optimization())
```

### Comprehensive Resource Management

#### Core Resource Operations
```python
import asyncio
from commons_k8s.resources import (
    Deployment, Service, ConfigMap, Secret, Ingress,
    Pod, PersistentVolumeClaim, ServiceAccount, NetworkPolicy
)
from commons_k8s import K8sClient
from datetime import datetime
import yaml
import json

async def demonstrate_resource_management():
    """Demonstrate comprehensive Kubernetes resource operations."""
    
    client = await K8sClient.from_config()
    
    print("=== Kubernetes Resource Management ===")
    
    # 1. Application Deployment with Full Stack
    print("\nDeploying complete application stack...")
    
    namespace = "demo-app"
    
    # Create namespace
    await client.create_namespace(namespace, labels={
        "app": "demo",
        "environment": "development",
        "managed-by": "commons-k8s"
    })
    print(f"✓ Created namespace: {namespace}")
    
    # 1.1 ConfigMap for application configuration
    config_data = {
        "app.yaml": yaml.dump({
            "database": {
                "host": "postgres-service",
                "port": 5432,
                "name": "myapp"
            },
            "redis": {
                "host": "redis-service",
                "port": 6379
            },
            "logging": {
                "level": "INFO",
                "format": "json"
            },
            "features": {
                "enable_metrics": True,
                "enable_tracing": True,
                "rate_limiting": {
                    "enabled": True,
                    "requests_per_minute": 1000
                }
            }
        }),
        "nginx.conf": """
server {
    listen 80;
    server_name _;
    
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /health {
        access_log off;
        return 200 "healthy\n";
    }
}
"""
    }
    
    app_config = ConfigMap(
        name="app-config",
        namespace=namespace,
        data=config_data,
        labels={"app": "demo", "component": "config"}
    )
    
    await client.create_configmap(app_config)
    print("✓ Created application ConfigMap")
    
    # 1.2 Secret for sensitive data
    secret_data = {
        "database-password": "c3VwZXJfc2VjcmV0X3Bhc3N3b3Jk",  # base64 encoded
        "api-key": "YWJjZGVmZ2hpams=",
        "jwt-secret": "dGhpc19pc19hX3Zlcnlfc2VjcmV0X2tleQ=="
    }
    
    app_secret = Secret(
        name="app-secrets",
        namespace=namespace,
        data=secret_data,
        type="Opaque",
        labels={"app": "demo", "component": "secrets"}
    )
    
    await client.create_secret(app_secret)
    print("✓ Created application Secret")
    
    # 1.3 PersistentVolumeClaim for data storage
    pvc = PersistentVolumeClaim(
        name="app-data",
        namespace=namespace,
        size="10Gi",
        access_modes=["ReadWriteOnce"],
        storage_class="fast-ssd",
        labels={"app": "demo", "component": "storage"}
    )
    
    await client.create_pvc(pvc)
    print("✓ Created PersistentVolumeClaim")
    
    # 1.4 ServiceAccount with RBAC
    service_account = ServiceAccount(
        name="app-service-account",
        namespace=namespace,
        labels={"app": "demo"}
    )
    
    await client.create_service_account(service_account)
    print("✓ Created ServiceAccount")
    
    # Create RBAC rules
    from commons_k8s.rbac import Role, RoleBinding
    
    app_role = Role(
        name="app-role",
        namespace=namespace,
        rules=[
            {
                "api_groups": [""],
                "resources": ["configmaps", "secrets"],
                "verbs": ["get", "list"]
            },
            {
                "api_groups": ["apps"],
                "resources": ["deployments"],
                "verbs": ["get", "list", "watch"]
            }
        ]
    )
    
    await client.create_role(app_role)
    
    role_binding = RoleBinding(
        name="app-role-binding",
        namespace=namespace,
        role_ref={
            "kind": "Role",
            "name": "app-role",
            "api_group": "rbac.authorization.k8s.io"
        },
        subjects=[
            {
                "kind": "ServiceAccount",
                "name": "app-service-account",
                "namespace": namespace
            }
        ]
    )
    
    await client.create_role_binding(role_binding)
    print("✓ Created RBAC rules")
    
    # 1.5 Main application deployment
    app_deployment = Deployment(
        name="demo-app",
        namespace=namespace,
        replicas=3,
        labels={"app": "demo", "component": "web"},
        selector={"app": "demo", "component": "web"},
        template={
            "metadata": {
                "labels": {"app": "demo", "component": "web"}
            },
            "spec": {
                "service_account_name": "app-service-account",
                "containers": [
                    {
                        "name": "app",
                        "image": "myapp:v1.2.3",
                        "ports": [{"container_port": 8080, "name": "http"}],
                        "env": [
                            {
                                "name": "DATABASE_PASSWORD",
                                "value_from": {
                                    "secret_key_ref": {
                                        "name": "app-secrets",
                                        "key": "database-password"
                                    }
                                }
                            },
                            {
                                "name": "API_KEY",
                                "value_from": {
                                    "secret_key_ref": {
                                        "name": "app-secrets",
                                        "key": "api-key"
                                    }
                                }
                            }
                        ],
                        "volume_mounts": [
                            {
                                "name": "config",
                                "mount_path": "/etc/config",
                                "read_only": True
                            },
                            {
                                "name": "data",
                                "mount_path": "/var/lib/app"
                            }
                        ],
                        "resources": {
                            "requests": {"cpu": "100m", "memory": "128Mi"},
                            "limits": {"cpu": "500m", "memory": "512Mi"}
                        },
                        "liveness_probe": {
                            "http_get": {"path": "/health", "port": 8080},
                            "initial_delay_seconds": 30,
                            "period_seconds": 10
                        },
                        "readiness_probe": {
                            "http_get": {"path": "/ready", "port": 8080},
                            "initial_delay_seconds": 5,
                            "period_seconds": 5
                        }
                    },
                    {
                        "name": "nginx-proxy",
                        "image": "nginx:1.21-alpine",
                        "ports": [{"container_port": 80, "name": "proxy"}],
                        "volume_mounts": [
                            {
                                "name": "nginx-config",
                                "mount_path": "/etc/nginx/conf.d",
                                "read_only": True
                            }
                        ],
                        "resources": {
                            "requests": {"cpu": "50m", "memory": "64Mi"},
                            "limits": {"cpu": "100m", "memory": "128Mi"}
                        }
                    }
                ],
                "volumes": [
                    {
                        "name": "config",
                        "config_map": {"name": "app-config"}
                    },
                    {
                        "name": "nginx-config",
                        "config_map": {
                            "name": "app-config",
                            "items": [{"key": "nginx.conf", "path": "default.conf"}]
                        }
                    },
                    {
                        "name": "data",
                        "persistent_volume_claim": {"claim_name": "app-data"}
                    }
                ],
                "node_selector": {"node-type": "application"},
                "affinity": {
                    "pod_anti_affinity": {
                        "preferred_during_scheduling_ignored_during_execution": [
                            {
                                "weight": 100,
                                "pod_affinity_term": {
                                    "label_selector": {
                                        "match_expressions": [
                                            {
                                                "key": "app",
                                                "operator": "In",
                                                "values": ["demo"]
                                            }
                                        ]
                                    },
                                    "topology_key": "kubernetes.io/hostname"
                                }
                            }
                        ]
                    }
                }
            }
        }
    )
    
    await client.create_deployment(app_deployment)
    print("✓ Created main application Deployment")
    
    # 1.6 Service to expose the application
    app_service = Service(
        name="demo-app-service",
        namespace=namespace,
        selector={"app": "demo", "component": "web"},
        ports=[
            {"name": "http", "port": 80, "target_port": 80, "protocol": "TCP"}
        ],
        type="ClusterIP",
        labels={"app": "demo", "component": "service"}
    )
    
    await client.create_service(app_service)
    print("✓ Created Service")
    
    # 1.7 Ingress for external access
    app_ingress = Ingress(
        name="demo-app-ingress",
        namespace=namespace,
        rules=[
            {
                "host": "demo.example.com",
                "http": {
                    "paths": [
                        {
                            "path": "/",
                            "path_type": "Prefix",
                            "backend": {
                                "service": {
                                    "name": "demo-app-service",
                                    "port": {"number": 80}
                                }
                            }
                        }
                    ]
                }
            }
        ],
        tls=[
            {
                "hosts": ["demo.example.com"],
                "secret_name": "demo-app-tls"
            }
        ],
        annotations={
            "kubernetes.io/ingress.class": "nginx",
            "cert-manager.io/cluster-issuer": "letsencrypt-prod",
            "nginx.ingress.kubernetes.io/rate-limit": "100",
            "nginx.ingress.kubernetes.io/ssl-redirect": "true"
        },
        labels={"app": "demo", "component": "ingress"}
    )
    
    await client.create_ingress(app_ingress)
    print("✓ Created Ingress")
    
    # 1.8 HorizontalPodAutoscaler for auto-scaling
    from commons_k8s.autoscaling import HorizontalPodAutoscaler
    
    hpa = HorizontalPodAutoscaler(
        name="demo-app-hpa",
        namespace=namespace,
        target_ref={
            "api_version": "apps/v1",
            "kind": "Deployment",
            "name": "demo-app"
        },
        min_replicas=3,
        max_replicas=10,
        metrics=[
            {
                "type": "Resource",
                "resource": {
                    "name": "cpu",
                    "target": {
                        "type": "Utilization",
                        "average_utilization": 70
                    }
                }
            },
            {
                "type": "Resource",
                "resource": {
                    "name": "memory",
                    "target": {
                        "type": "Utilization",
                        "average_utilization": 80
                    }
                }
            }
        ],
        behavior={
            "scale_up": {
                "stabilization_window_seconds": 60,
                "policies": [
                    {
                        "type": "Percent",
                        "value": 100,
                        "period_seconds": 15
                    }
                ]
            },
            "scale_down": {
                "stabilization_window_seconds": 300,
                "policies": [
                    {
                        "type": "Percent",
                        "value": 10,
                        "period_seconds": 60
                    }
                ]
            }
        }
    )
    
    await client.create_hpa(hpa)
    print("✓ Created HorizontalPodAutoscaler")
    
    # 2. Resource Monitoring and Management
    print("\n=== Resource Monitoring ===")
    
    # Wait for deployment to be ready
    print("Waiting for deployment to be ready...")
    await client.wait_for_deployment_ready(
        "demo-app", 
        namespace=namespace, 
        timeout=300
    )
    print("✓ Deployment is ready")
    
    # Get deployment status
    deployment_status = await client.get_deployment_status(
        "demo-app", 
        namespace=namespace
    )
    
    print(f"Deployment status:")
    print(f"  Replicas: {deployment_status.ready_replicas}/{deployment_status.replicas}")
    print(f"  Updated: {deployment_status.updated_replicas}")
    print(f"  Available: {deployment_status.available_replicas}")
    
    # List all pods
    pods = await client.list_pods(
        namespace=namespace,
        label_selector="app=demo,component=web"
    )
    
    print(f"\nPods ({len(pods)}):")
    for pod in pods:
        print(f"  - {pod.metadata.name}:")
        print(f"    Status: {pod.status.phase}")
        print(f"    Node: {pod.spec.node_name}")
        print(f"    IP: {pod.status.pod_ip}")
        print(f"    Started: {pod.status.start_time}")
        
        # Show container statuses
        if pod.status.container_statuses:
            for container in pod.status.container_statuses:
                print(f"    Container {container.name}: {container.state}")
    
    # 3. Resource Updates and Patches
    print("\n=== Resource Updates ===")
    
    # Rolling update - change image version
    print("Performing rolling update...")
    
    patch = {
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "app",
                            "image": "myapp:v1.2.4"  # New version
                        }
                    ]
                }
            }
        }
    }
    
    await client.patch_deployment(
        "demo-app",
        namespace=namespace,
        patch=patch
    )
    print("✓ Started rolling update")
    
    # Monitor rollout status
    rollout_status = await client.get_rollout_status(
        "demo-app",
        namespace=namespace
    )
    
    print(f"Rollout status: {rollout_status.phase}")
    
    # Scale deployment
    print("\nScaling deployment...")
    
    await client.scale_deployment(
        "demo-app",
        namespace=namespace,
        replicas=5
    )
    print("✓ Scaled deployment to 5 replicas")
    
    # Update ConfigMap
    print("\nUpdating configuration...")
    
    updated_config_data = config_data.copy()
    updated_config_data["app.yaml"] = yaml.dump({
        "database": {
            "host": "postgres-service",
            "port": 5432,
            "name": "myapp",
            "pool_size": 20  # New setting
        },
        "redis": {
            "host": "redis-service",
            "port": 6379,
            "max_connections": 100  # New setting
        },
        "logging": {
            "level": "DEBUG",  # Changed from INFO
            "format": "json"
        },
        "features": {
            "enable_metrics": True,
            "enable_tracing": True,
            "rate_limiting": {
                "enabled": True,
                "requests_per_minute": 1500  # Increased
            },
            "new_feature": True  # New feature flag
        }
    })
    
    await client.patch_configmap(
        "app-config",
        namespace=namespace,
        patch={"data": updated_config_data}
    )
    print("✓ Updated ConfigMap")
    
    # Restart deployment to pick up config changes
    await client.restart_deployment("demo-app", namespace=namespace)
    print("✓ Restarted deployment to pick up config changes")
    
    await client.close()

# Advanced resource operations
async def demonstrate_advanced_operations():
    """Demonstrate advanced resource operations and patterns."""
    
    client = await K8sClient.from_config()
    
    print("\n=== Advanced Resource Operations ===")
    
    namespace = "advanced-demo"
    await client.create_namespace(namespace)
    
    # 1. Batch operations
    print("\nBatch resource creation...")
    
    # Create multiple resources in one batch
    resources = []
    
    # Multiple deployments
    for i in range(3):
        deployment = Deployment(
            name=f"worker-{i}",
            namespace=namespace,
            replicas=2,
            image=f"worker:v1.0.{i}",
            labels={"app": "worker", "instance": str(i)}
        )
        resources.append(deployment)
    
    # Multiple services
    for i in range(3):
        service = Service(
            name=f"worker-{i}-service",
            namespace=namespace,
            selector={"app": "worker", "instance": str(i)},
            ports=[{"port": 8080, "target_port": 8080}]
        )
        resources.append(service)
    
    # Batch create
    created_resources = await client.create_many(resources)
    print(f"✓ Created {len(created_resources)} resources in batch")
    
    # 2. Resource watching and event handling
    print("\nStarting resource watch...")
    
    from commons_k8s.events import EventHandler
    
    class DeploymentEventHandler(EventHandler):
        def __init__(self):
            self.events_received = 0
        
        async def handle_added(self, deployment):
            self.events_received += 1
            print(f"  ✓ Deployment added: {deployment.metadata.name}")
        
        async def handle_modified(self, deployment):
            self.events_received += 1
            print(f"  ✓ Deployment modified: {deployment.metadata.name}")
        
        async def handle_deleted(self, deployment):
            self.events_received += 1
            print(f"  ✓ Deployment deleted: {deployment.metadata.name}")
    
    handler = DeploymentEventHandler()
    
    # Start watching in background
    watch_task = asyncio.create_task(
        client.watch_deployments(
            namespace=namespace,
            handler=handler,
            timeout=30  # Watch for 30 seconds
        )
    )
    
    # Perform operations that will trigger events
    await asyncio.sleep(2)  # Let watch start
    
    # Scale a deployment (will trigger MODIFIED event)
    await client.scale_deployment(
        "worker-0",
        namespace=namespace,
        replicas=3
    )
    
    # Delete a deployment (will trigger DELETED event)
    await client.delete_deployment("worker-2", namespace=namespace)
    
    # Wait for watch to complete
    try:
        await asyncio.wait_for(watch_task, timeout=35)
    except asyncio.TimeoutError:
        watch_task.cancel()
    
    print(f"Total events received: {handler.events_received}")
    
    # 3. Resource queries and filtering
    print("\n=== Resource Queries ===")
    
    # Complex label selector queries
    queries = [
        "app=worker",
        "app=worker,instance!=1",
        "app in (worker,api)",
        "instance notin (0,2)"
    ]
    
    for query in queries:
        try:
            deployments = await client.list_deployments(
                namespace=namespace,
                label_selector=query
            )
            print(f"Query '{query}': {len(deployments)} deployments")
        except Exception as e:
            print(f"Query '{query}' failed: {e}")
    
    # Field selector queries
    field_queries = [
        "status.phase=Running",
        "spec.nodeName=node-1"
    ]
    
    for query in field_queries:
        try:
            pods = await client.list_pods(
                namespace=namespace,
                field_selector=query
            )
            print(f"Field query '{query}': {len(pods)} pods")
        except Exception as e:
            print(f"Field query '{query}' failed: {e}")
    
    # 4. Resource ownership and garbage collection
    print("\n=== Ownership and Garbage Collection ===")
    
    # Create parent resource
    parent_deployment = await client.get_deployment("worker-0", namespace=namespace)
    
    # Create child resource with owner reference
    child_configmap = ConfigMap(
        name="worker-0-config",
        namespace=namespace,
        data={"config.yaml": "debug: true"},
        owner_references=[
            {
                "api_version": parent_deployment.api_version,
                "kind": parent_deployment.kind,
                "name": parent_deployment.metadata.name,
                "uid": parent_deployment.metadata.uid,
                "controller": True,
                "block_owner_deletion": True
            }
        ]
    )
    
    await client.create_configmap(child_configmap)
    print("✓ Created child resource with owner reference")
    
    # Verify ownership
    created_configmap = await client.get_configmap(
        "worker-0-config", 
        namespace=namespace
    )
    
    if created_configmap.metadata.owner_references:
        owner = created_configmap.metadata.owner_references[0]
        print(f"ConfigMap owned by: {owner.kind}/{owner.name}")
    
    # Delete parent (child should be garbage collected)
    await client.delete_deployment("worker-0", namespace=namespace)
    print("✓ Deleted parent deployment")
    
    # Wait and check if child was deleted
    await asyncio.sleep(5)
    
    try:
        await client.get_configmap("worker-0-config", namespace=namespace)
        print("⚠ Child ConfigMap still exists (may take time for GC)")
    except Exception:
        print("✓ Child ConfigMap was garbage collected")
    
    await client.close()

# Run resource management demonstrations
asyncio.run(demonstrate_resource_management())
asyncio.run(demonstrate_advanced_operations())
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