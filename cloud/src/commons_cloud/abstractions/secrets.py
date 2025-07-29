"""Secrets provider abstraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from commons_core.types import BaseModel


class SecretManager:
    """High-level secret manager client."""
    
    def __init__(self, provider: str = "aws"):
        self.provider = provider
    
    async def get_secret(self, name: str) -> Dict[str, Any]:
        """Get secret value."""
        return {"value": "mock-secret-value"}
    
    async def set_secret(self, name: str, value: str) -> bool:
        """Set secret value."""
        return True
    
    async def delete_secret(self, name: str) -> bool:
        """Delete secret."""
        return True
    
    async def list_secrets(self) -> List[str]:
        """List all secrets."""
        return ["secret1", "secret2"]


@dataclass
class Secret:
    """Represents a secret."""
    
    name: str
    arn: Optional[str] = None
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    version_count: int = 0
    rotation_enabled: bool = False
    last_rotation: Optional[datetime] = None
    tags: Optional[Dict[str, str]] = None


@dataclass
class SecretVersion:
    """Represents a version of a secret."""
    
    version_id: str
    version_stage: List[str]
    created_at: datetime
    value: Optional[Union[str, Dict[str, Any]]] = None


@dataclass
class RotationConfig:
    """Configuration for secret rotation."""
    
    enabled: bool
    rotation_lambda_arn: Optional[str] = None
    rotation_interval_days: int = 30
    immediately_rotate: bool = False


class SecretsProvider(ABC):
    """Abstract base class for secrets providers."""
    
    def __init__(self, region: str) -> None:
        self.region = region
        
    @abstractmethod
    async def create_secret(
        self,
        name: str,
        value: Union[str, Dict[str, Any]],
        description: Optional[str] = None,
        kms_key_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Secret:
        """Create a new secret."""
        pass
        
    @abstractmethod
    async def get_secret(
        self,
        name: str,
        version_id: Optional[str] = None,
        version_stage: Optional[str] = None,
    ) -> Union[str, Dict[str, Any]]:
        """Retrieve secret value."""
        pass
        
    @abstractmethod
    async def get_secret_metadata(self, name: str) -> Secret:
        """Get secret metadata without the value."""
        pass
        
    @abstractmethod
    async def update_secret(
        self,
        name: str,
        value: Union[str, Dict[str, Any]],
        description: Optional[str] = None,
    ) -> SecretVersion:
        """Update secret value."""
        pass
        
    @abstractmethod
    async def delete_secret(
        self,
        name: str,
        force: bool = False,
        recovery_days: int = 30,
    ) -> None:
        """Delete a secret."""
        pass
        
    @abstractmethod
    async def restore_secret(self, name: str) -> None:
        """Restore a deleted secret."""
        pass
        
    @abstractmethod
    async def list_secrets(
        self,
        filters: Optional[Dict[str, Any]] = None,
        max_results: Optional[int] = None,
    ) -> List[Secret]:
        """List all secrets."""
        pass
        
    @abstractmethod
    async def list_secret_versions(
        self,
        name: str,
        include_deprecated: bool = False,
    ) -> List[SecretVersion]:
        """List all versions of a secret."""
        pass
        
    @abstractmethod
    async def get_secret_version(
        self,
        name: str,
        version_id: str,
    ) -> SecretVersion:
        """Get a specific version of a secret."""
        pass
        
    @abstractmethod
    async def tag_secret(
        self,
        name: str,
        tags: Dict[str, str],
    ) -> None:
        """Add tags to a secret."""
        pass
        
    @abstractmethod
    async def untag_secret(
        self,
        name: str,
        tag_keys: List[str],
    ) -> None:
        """Remove tags from a secret."""
        pass
        
    @abstractmethod
    async def enable_rotation(
        self,
        name: str,
        rotation_config: RotationConfig,
    ) -> None:
        """Enable automatic rotation for a secret."""
        pass
        
    @abstractmethod
    async def disable_rotation(self, name: str) -> None:
        """Disable automatic rotation."""
        pass
        
    @abstractmethod
    async def rotate_secret(self, name: str) -> None:
        """Manually trigger secret rotation."""
        pass
        
    @abstractmethod
    async def put_secret_version(
        self,
        name: str,
        value: Union[str, Dict[str, Any]],
        version_stages: Optional[List[str]] = None,
    ) -> SecretVersion:
        """Put a new version of a secret."""
        pass
        
    @abstractmethod
    async def update_version_stage(
        self,
        name: str,
        version_stage: str,
        move_to_version_id: str,
        remove_from_version_id: Optional[str] = None,
    ) -> None:
        """Update version stages."""
        pass