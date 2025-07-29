"""Factory classes for creating cloud provider instances."""

from typing import Any, Dict, Optional, Type, Union
from enum import Enum

from commons_core.errors import ValidationError
from .abstractions import StorageProvider, ComputeProvider, SecretsProvider


class CloudProviderType(Enum):
    """Supported cloud providers."""
    
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"


class StorageFactory:
    """Factory for creating storage provider instances."""
    
    _providers: Dict[str, Type[StorageProvider]] = {}
    
    @classmethod
    def register(cls, provider: str, provider_class: Type[StorageProvider]) -> None:
        """Register a storage provider."""
        cls._providers[provider.lower()] = provider_class
        
    @classmethod
    async def create(
        cls,
        provider: Union[str, CloudProviderType],
        **kwargs: Any,
    ) -> StorageProvider:
        """Create a storage provider instance."""
        if isinstance(provider, CloudProviderType):
            provider = provider.value
            
        provider = provider.lower()
        
        if provider not in cls._providers:
            # Lazy import providers
            if provider == "aws":
                from .providers.aws import S3Storage
                cls.register("aws", S3Storage)
            elif provider == "gcp":
                from .providers.gcp import GCSStorage
                cls.register("gcp", GCSStorage)
            elif provider == "azure":
                from .providers.azure import BlobStorage
                cls.register("azure", BlobStorage)
            else:
                raise ValidationError(f"Unknown storage provider: {provider}")
                
        provider_class = cls._providers[provider]
        return provider_class(**kwargs)


class ComputeFactory:
    """Factory for creating compute provider instances."""
    
    _providers: Dict[str, Type[ComputeProvider]] = {}
    
    @classmethod
    def register(cls, provider: str, provider_class: Type[ComputeProvider]) -> None:
        """Register a compute provider."""
        cls._providers[provider.lower()] = provider_class
        
    @classmethod
    async def create(
        cls,
        provider: Union[str, CloudProviderType],
        **kwargs: Any,
    ) -> ComputeProvider:
        """Create a compute provider instance."""
        if isinstance(provider, CloudProviderType):
            provider = provider.value
            
        provider = provider.lower()
        
        if provider not in cls._providers:
            # Lazy import providers
            if provider == "aws":
                from .providers.aws import EC2Compute
                cls.register("aws", EC2Compute)
            elif provider == "gcp":
                from .providers.gcp import ComputeEngineCompute
                cls.register("gcp", ComputeEngineCompute)
            elif provider == "azure":
                from .providers.azure import VirtualMachineCompute
                cls.register("azure", VirtualMachineCompute)
            else:
                raise ValidationError(f"Unknown compute provider: {provider}")
                
        provider_class = cls._providers[provider]
        return provider_class(**kwargs)


class SecretsFactory:
    """Factory for creating secrets provider instances."""
    
    _providers: Dict[str, Type[SecretsProvider]] = {}
    
    @classmethod
    def register(cls, provider: str, provider_class: Type[SecretsProvider]) -> None:
        """Register a secrets provider."""
        cls._providers[provider.lower()] = provider_class
        
    @classmethod
    async def create(
        cls,
        provider: Union[str, CloudProviderType],
        **kwargs: Any,
    ) -> SecretsProvider:
        """Create a secrets provider instance."""
        if isinstance(provider, CloudProviderType):
            provider = provider.value
            
        provider = provider.lower()
        
        if provider not in cls._providers:
            # Lazy import providers
            if provider == "aws":
                from .providers.aws import SecretsManagerSecrets
                cls.register("aws", SecretsManagerSecrets)
            elif provider == "gcp":
                from .providers.gcp import SecretManagerSecrets
                cls.register("gcp", SecretManagerSecrets)
            elif provider == "azure":
                from .providers.azure import KeyVaultSecrets
                cls.register("azure", KeyVaultSecrets)
            else:
                raise ValidationError(f"Unknown secrets provider: {provider}")
                
        provider_class = cls._providers[provider]
        return provider_class(**kwargs)


class CloudProvider:
    """Unified cloud provider interface."""
    
    def __init__(
        self,
        provider: Union[str, CloudProviderType],
        credentials: Optional[Dict[str, Any]] = None,
        region: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(provider, str):
            provider = CloudProviderType(provider.lower())
            
        self.provider = provider
        self.name = provider.value
        self.credentials = credentials or {}
        self.region = region
        self.kwargs = kwargs
        
    async def get_storage(self, bucket: str, **kwargs: Any) -> StorageProvider:
        """Get storage provider instance."""
        params = {
            "bucket": bucket,
            "region": self.region,
            **self.credentials,
            **self.kwargs,
            **kwargs,
        }
        return await StorageFactory.create(self.provider, **params)
        
    async def get_compute(self, **kwargs: Any) -> ComputeProvider:
        """Get compute provider instance."""
        params = {
            "region": self.region,
            **self.credentials,
            **self.kwargs,
            **kwargs,
        }
        return await ComputeFactory.create(self.provider, **params)
        
    async def get_secrets(self, **kwargs: Any) -> SecretsProvider:
        """Get secrets provider instance."""
        params = {
            "region": self.region,
            **self.credentials,
            **self.kwargs,
            **kwargs,
        }
        return await SecretsFactory.create(self.provider, **params)