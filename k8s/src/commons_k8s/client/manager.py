"""Kubernetes client manager implementation."""

from typing import Optional, Dict, Any, List
from pathlib import Path
import asyncio
from kubernetes_asyncio import client, config
from kubernetes_asyncio.client import ApiClient, CoreV1Api, AppsV1Api
from dataclasses import dataclass

from commons_core.logging import get_logger
from commons_core.errors import ConfigError

logger = get_logger(__name__)


@dataclass
class ClientConfig:
    """Kubernetes client configuration."""
    
    kubeconfig_path: Optional[str] = None
    context: Optional[str] = None
    in_cluster: bool = False
    verify_ssl: bool = True
    ssl_ca_cert: Optional[str] = None
    api_key: Optional[str] = None
    host: Optional[str] = None


class K8sClient:
    """Synchronous Kubernetes client (legacy compatibility)."""
    
    def __init__(self, config: Optional[ClientConfig] = None) -> None:
        self.config = config or ClientConfig()
        self._setup_client()
        
    def _setup_client(self) -> None:
        """Setup Kubernetes client."""
        # This would use the sync kubernetes library
        # Simplified for example
        pass
        
    @classmethod
    def from_config(
        cls,
        config_file: Optional[str] = None,
        context: Optional[str] = None,
    ) -> "K8sClient":
        """Create client from kubeconfig."""
        config = ClientConfig(
            kubeconfig_path=config_file,
            context=context,
        )
        return cls(config)
        
    @classmethod
    def in_cluster(cls) -> "K8sClient":
        """Create in-cluster client."""
        config = ClientConfig(in_cluster=True)
        return cls(config)


class K8sAsyncClient:
    """Asynchronous Kubernetes client."""
    
    def __init__(self, config: Optional[ClientConfig] = None) -> None:
        self.config = config or ClientConfig()
        self._api_client: Optional[ApiClient] = None
        self._core_v1: Optional[CoreV1Api] = None
        self._apps_v1: Optional[AppsV1Api] = None
        
    async def __aenter__(self) -> "K8sAsyncClient":
        """Async context manager entry."""
        await self._setup_client()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
        
    async def _setup_client(self) -> None:
        """Setup Kubernetes client."""
        try:
            if self.config.in_cluster:
                config.load_incluster_config()
            else:
                await config.load_kube_config(
                    config_file=self.config.kubeconfig_path,
                    context=self.config.context,
                )
                
            configuration = client.Configuration.get_default_copy()
            
            if self.config.host:
                configuration.host = self.config.host
            if self.config.api_key:
                configuration.api_key = {"authorization": self.config.api_key}
            if self.config.ssl_ca_cert:
                configuration.ssl_ca_cert = self.config.ssl_ca_cert
            configuration.verify_ssl = self.config.verify_ssl
            
            self._api_client = ApiClient(configuration=configuration)
            self._core_v1 = CoreV1Api(self._api_client)
            self._apps_v1 = AppsV1Api(self._api_client)
            
            logger.info("Kubernetes client initialized", context=self.config.context)
            
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")
            raise ConfigError(f"Failed to initialize Kubernetes client: {e}")
            
    async def close(self) -> None:
        """Close client connections."""
        if self._api_client:
            await self._api_client.close()
            
    @classmethod
    async def from_config(
        cls,
        config_file: Optional[str] = None,
        context: Optional[str] = None,
    ) -> "K8sAsyncClient":
        """Create client from kubeconfig."""
        config = ClientConfig(
            kubeconfig_path=config_file,
            context=context,
        )
        client = cls(config)
        await client._setup_client()
        return client
        
    @classmethod
    async def in_cluster(cls) -> "K8sAsyncClient":
        """Create in-cluster client."""
        config = ClientConfig(in_cluster=True)
        client = cls(config)
        await client._setup_client()
        return client
        
    async def switch_context(self, context: str) -> None:
        """Switch to different context."""
        self.config.context = context
        await self._setup_client()
        
    # Deployment operations
    async def list_deployments(
        self,
        namespace: str = "default",
        label_selector: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List deployments in namespace."""
        try:
            if label_selector:
                deployments = await self._apps_v1.list_namespaced_deployment(
                    namespace=namespace,
                    label_selector=label_selector,
                )
            else:
                deployments = await self._apps_v1.list_namespaced_deployment(
                    namespace=namespace,
                )
                
            return [self._deployment_to_dict(d) for d in deployments.items]
            
        except Exception as e:
            logger.error(f"Failed to list deployments: {e}")
            raise
            
    async def get_deployment(self, name: str, namespace: str = "default") -> Dict[str, Any]:
        """Get deployment by name."""
        try:
            deployment = await self._apps_v1.read_namespaced_deployment(
                name=name,
                namespace=namespace,
            )
            return self._deployment_to_dict(deployment)
            
        except Exception as e:
            logger.error(f"Failed to get deployment {name}: {e}")
            raise
            
    async def create_deployment(
        self,
        deployment: Dict[str, Any],
        namespace: str = "default",
    ) -> Dict[str, Any]:
        """Create deployment."""
        try:
            created = await self._apps_v1.create_namespaced_deployment(
                namespace=namespace,
                body=deployment,
            )
            logger.info(f"Created deployment {created.metadata.name}")
            return self._deployment_to_dict(created)
            
        except Exception as e:
            logger.error(f"Failed to create deployment: {e}")
            raise
            
    async def update_deployment(
        self,
        name: str,
        deployment: Dict[str, Any],
        namespace: str = "default",
    ) -> Dict[str, Any]:
        """Update deployment."""
        try:
            updated = await self._apps_v1.patch_namespaced_deployment(
                name=name,
                namespace=namespace,
                body=deployment,
            )
            logger.info(f"Updated deployment {name}")
            return self._deployment_to_dict(updated)
            
        except Exception as e:
            logger.error(f"Failed to update deployment {name}: {e}")
            raise
            
    async def delete_deployment(
        self,
        name: str,
        namespace: str = "default",
        grace_period: int = 30,
    ) -> None:
        """Delete deployment."""
        try:
            await self._apps_v1.delete_namespaced_deployment(
                name=name,
                namespace=namespace,
                grace_period_seconds=grace_period,
            )
            logger.info(f"Deleted deployment {name}")
            
        except Exception as e:
            logger.error(f"Failed to delete deployment {name}: {e}")
            raise
            
    async def scale_deployment(
        self,
        name: str,
        replicas: int,
        namespace: str = "default",
    ) -> None:
        """Scale deployment to specified replicas."""
        try:
            # Get current deployment
            deployment = await self._apps_v1.read_namespaced_deployment(
                name=name,
                namespace=namespace,
            )
            
            # Update replicas
            deployment.spec.replicas = replicas
            
            await self._apps_v1.patch_namespaced_deployment(
                name=name,
                namespace=namespace,
                body=deployment,
            )
            
            logger.info(f"Scaled deployment {name} to {replicas} replicas")
            
        except Exception as e:
            logger.error(f"Failed to scale deployment {name}: {e}")
            raise
            
    # Pod operations
    async def list_pods(
        self,
        namespace: str = "default",
        label_selector: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List pods in namespace."""
        try:
            if label_selector:
                pods = await self._core_v1.list_namespaced_pod(
                    namespace=namespace,
                    label_selector=label_selector,
                )
            else:
                pods = await self._core_v1.list_namespaced_pod(
                    namespace=namespace,
                )
                
            return [self._pod_to_dict(p) for p in pods.items]
            
        except Exception as e:
            logger.error(f"Failed to list pods: {e}")
            raise
            
    def _deployment_to_dict(self, deployment) -> Dict[str, Any]:
        """Convert deployment object to dict."""
        return {
            "name": deployment.metadata.name,
            "namespace": deployment.metadata.namespace,
            "replicas": deployment.spec.replicas,
            "available_replicas": deployment.status.available_replicas or 0,
            "labels": deployment.metadata.labels or {},
            "creation_timestamp": deployment.metadata.creation_timestamp,
        }
        
    def _pod_to_dict(self, pod) -> Dict[str, Any]:
        """Convert pod object to dict."""
        return {
            "name": pod.metadata.name,
            "namespace": pod.metadata.namespace,
            "phase": pod.status.phase,
            "node": pod.spec.node_name,
            "labels": pod.metadata.labels or {},
            "creation_timestamp": pod.metadata.creation_timestamp,
        }