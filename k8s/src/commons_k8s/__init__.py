"""Commons K8s - Kubernetes abstractions."""

from .types import ResourceSpec, ResourceStatus, PodSpec
from .resources import Deployment, Service, Pod, ConfigMap, Secret

__version__ = "0.1.0"

__all__ = [
    "ResourceSpec",
    "ResourceStatus", 
    "PodSpec",
    "Deployment",
    "Service",
    "Pod", 
    "ConfigMap",
    "Secret",
]
