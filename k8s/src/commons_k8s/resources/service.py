"""Service resource."""

from ..types import ResourceSpec

class Service:
    """Kubernetes Service resource."""
    
    def __init__(self, name: str, namespace: str = "default"):
        self.name = name
        self.namespace = namespace
        self.spec = ResourceSpec(
            api_version="v1",
            kind="Service", 
            metadata={"name": name, "namespace": namespace}
        )
