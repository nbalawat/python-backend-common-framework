"""Cloud provider abstractions."""

from .storage import StorageProvider, StorageObject, UploadOptions, SignedUrlOptions, StorageClient
from .compute import ComputeProvider, Instance, InstanceState, CreateInstanceOptions
from .secrets import SecretsProvider, SecretManager, Secret, SecretVersion, RotationConfig

__all__ = [
    # Storage
    "StorageProvider",
    "StorageClient", 
    "StorageObject",
    "UploadOptions",
    "SignedUrlOptions",
    # Compute
    "ComputeProvider",
    "Instance",
    "InstanceState",
    "CreateInstanceOptions",
    # Secrets
    "SecretsProvider",
    "SecretManager",
    "Secret",
    "SecretVersion",
    "RotationConfig",
]