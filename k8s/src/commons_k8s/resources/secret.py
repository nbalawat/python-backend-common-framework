"""Secret resource."""

from ..types import ResourceSpec

class Secret:
    """Kubernetes Secret resource."""
    
    def __init__(self, name: str, namespace: str = "default"):
        self.name = name
        self.namespace = namespace
        self.spec = ResourceSpec(
            api_version="v1",
            kind="Secret",
            metadata={"name": name, "namespace": namespace}
        )
