"""Deployment resource."""

from ..types import ResourceSpec

class Deployment:
    """Kubernetes Deployment resource."""
    
    def __init__(self, name: str, namespace: str = "default"):
        self.name = name
        self.namespace = namespace
        self.spec = ResourceSpec(
            api_version="apps/v1",
            kind="Deployment",
            metadata={"name": name, "namespace": namespace}
        )
