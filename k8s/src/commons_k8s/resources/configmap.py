"""ConfigMap resource."""

from ..types import ResourceSpec

class ConfigMap:
    """Kubernetes ConfigMap resource."""
    
    def __init__(self, name: str, namespace: str = "default"):
        self.name = name
        self.namespace = namespace
        self.spec = ResourceSpec(
            api_version="v1",
            kind="ConfigMap",
            metadata={"name": name, "namespace": namespace}
        )
