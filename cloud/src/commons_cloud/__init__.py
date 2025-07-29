"""Commons Cloud - Multi-cloud provider abstractions."""

from .abstractions import (
    StorageProvider,
    StorageClient,
    ComputeProvider,
    SecretsProvider,
    SecretManager,
)
from .factory import (
    StorageFactory,
    ComputeFactory,
    SecretsFactory,
    CloudProvider,
)

__version__ = "0.1.0"

__all__ = [
    "StorageProvider",
    "StorageClient",
    "ComputeProvider",
    "SecretsProvider",
    "SecretManager",
    "StorageFactory",
    "ComputeFactory",
    "SecretsFactory",
    "CloudProvider",
]