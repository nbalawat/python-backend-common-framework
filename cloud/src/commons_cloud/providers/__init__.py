"""Cloud provider implementations."""

from .aws import S3Storage, EC2Compute, SecretsManagerSecrets
from .gcp import GCSStorage, ComputeEngineCompute, SecretManagerSecrets
from .azure import BlobStorage, VirtualMachineCompute, KeyVaultSecrets

__all__ = [
    # AWS
    "S3Storage",
    "EC2Compute",
    "SecretsManagerSecrets",
    # GCP
    "GCSStorage",
    "ComputeEngineCompute",
    "SecretManagerSecrets",
    # Azure
    "BlobStorage",
    "VirtualMachineCompute",
    "KeyVaultSecrets",
]