"""Pod resource."""

from ..types import ResourceSpec

class Pod:
    """Kubernetes Pod resource."""
    
    def __init__(self, name: str, namespace: str = "default"):
        self.name = name
        self.namespace = namespace
        self.spec = ResourceSpec(
            api_version="v1",
            kind="Pod",
            metadata={"name": name, "namespace": namespace}
        )
