"""Azure provider implementations."""

from .blob import BlobStorage
from .vm import VirtualMachineCompute
from .key_vault import KeyVaultSecrets

__all__ = [
    "BlobStorage",
    "VirtualMachineCompute",
    "KeyVaultSecrets",
]