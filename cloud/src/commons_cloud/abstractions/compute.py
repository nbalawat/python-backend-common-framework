"""Compute provider abstraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from commons_core.types import BaseModel


class InstanceState(Enum):
    """Instance states across providers."""
    
    PENDING = "pending"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    TERMINATED = "terminated"
    UNKNOWN = "unknown"


@dataclass
class Instance:
    """Represents a compute instance."""
    
    id: str
    name: str
    state: InstanceState
    instance_type: str
    image_id: str
    public_ip: Optional[str] = None
    private_ip: Optional[str] = None
    created_at: Optional[datetime] = None
    tags: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CreateInstanceOptions:
    """Options for creating instances."""
    
    name: str
    image_id: str
    instance_type: str
    key_name: Optional[str] = None
    security_groups: Optional[List[str]] = None
    subnet_id: Optional[str] = None
    user_data: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, str]] = None
    disk_size_gb: Optional[int] = None
    disk_type: Optional[str] = None
    public_ip: bool = True
    monitoring: bool = False
    spot_instance: bool = False
    spot_price: Optional[float] = None


class ComputeProvider(ABC):
    """Abstract base class for compute providers."""
    
    def __init__(self, region: str, zone: Optional[str] = None) -> None:
        self.region = region
        self.zone = zone
        
    @abstractmethod
    async def list_instances(
        self,
        filters: Optional[Dict[str, Any]] = None,
        max_results: Optional[int] = None,
    ) -> List[Instance]:
        """List compute instances."""
        pass
        
    @abstractmethod
    async def get_instance(self, instance_id: str) -> Instance:
        """Get instance details."""
        pass
        
    @abstractmethod
    async def create_instance(
        self,
        options: CreateInstanceOptions,
    ) -> Instance:
        """Create a new instance."""
        pass
        
    @abstractmethod
    async def start_instance(self, instance_id: str) -> None:
        """Start a stopped instance."""
        pass
        
    @abstractmethod
    async def stop_instance(self, instance_id: str, force: bool = False) -> None:
        """Stop a running instance."""
        pass
        
    @abstractmethod
    async def restart_instance(self, instance_id: str) -> None:
        """Restart an instance."""
        pass
        
    @abstractmethod
    async def terminate_instance(self, instance_id: str) -> None:
        """Terminate an instance."""
        pass
        
    @abstractmethod
    async def resize_instance(
        self,
        instance_id: str,
        new_instance_type: str,
    ) -> None:
        """Resize an instance."""
        pass
        
    @abstractmethod
    async def get_console_output(self, instance_id: str) -> str:
        """Get console output from instance."""
        pass
        
    @abstractmethod
    async def get_console_screenshot(self, instance_id: str) -> bytes:
        """Get console screenshot."""
        pass
        
    @abstractmethod
    async def attach_disk(
        self,
        instance_id: str,
        disk_id: str,
        device_name: Optional[str] = None,
    ) -> None:
        """Attach a disk to instance."""
        pass
        
    @abstractmethod
    async def detach_disk(
        self,
        instance_id: str,
        disk_id: str,
    ) -> None:
        """Detach a disk from instance."""
        pass
        
    @abstractmethod
    async def create_snapshot(
        self,
        instance_id: str,
        snapshot_name: str,
        description: Optional[str] = None,
    ) -> str:
        """Create a snapshot of instance."""
        pass
        
    @abstractmethod
    async def list_snapshots(
        self,
        instance_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List snapshots."""
        pass
        
    @abstractmethod
    async def delete_snapshot(self, snapshot_id: str) -> None:
        """Delete a snapshot."""
        pass
        
    @abstractmethod
    async def create_image(
        self,
        instance_id: str,
        image_name: str,
        description: Optional[str] = None,
    ) -> str:
        """Create an image from instance."""
        pass
        
    @abstractmethod
    async def list_images(
        self,
        owned_by_me: bool = False,
    ) -> List[Dict[str, Any]]:
        """List available images."""
        pass
        
    @abstractmethod
    async def delete_image(self, image_id: str) -> None:
        """Delete an image."""
        pass
        
    @abstractmethod
    async def update_tags(
        self,
        instance_id: str,
        tags: Dict[str, str],
    ) -> None:
        """Update instance tags."""
        pass
        
    @abstractmethod
    async def get_instance_types(self) -> List[Dict[str, Any]]:
        """Get available instance types."""
        pass