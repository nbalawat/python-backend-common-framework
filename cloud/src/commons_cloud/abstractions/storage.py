"""Storage provider abstraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncIterator, Dict, List, Optional, Set, Union
from pathlib import Path
import io

from commons_core.types import BaseModel


class StorageClient:
    """High-level storage client."""
    
    def __init__(self, bucket: str, provider: Optional[str] = None):
        self.bucket = bucket
        self.provider = provider or "s3"
    
    async def upload(self, key: str, data: Union[bytes, str, io.IOBase]) -> bool:
        """Upload data to storage."""
        return True
    
    async def download(self, key: str) -> bytes:
        """Download data from storage."""
        return b"mock data"
    
    async def exists(self, key: str) -> bool:
        """Check if object exists."""
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete object."""
        return True


@dataclass
class StorageObject:
    """Represents a storage object."""
    
    key: str
    size: int
    last_modified: datetime
    etag: str
    storage_class: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None
    content_type: Optional[str] = None


@dataclass
class UploadOptions:
    """Options for uploading objects."""
    
    content_type: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None
    storage_class: Optional[str] = None
    encryption: Optional[str] = None
    cache_control: Optional[str] = None
    content_disposition: Optional[str] = None
    content_encoding: Optional[str] = None
    expires: Optional[datetime] = None
    acl: Optional[str] = None


@dataclass
class SignedUrlOptions:
    """Options for generating signed URLs."""
    
    expires_in: int = 3600  # seconds
    method: str = "GET"
    content_type: Optional[str] = None
    response_headers: Optional[Dict[str, str]] = None
    version: Optional[str] = None


class StorageProvider(ABC):
    """Abstract base class for storage providers."""
    
    def __init__(self, bucket: str, region: Optional[str] = None) -> None:
        self.bucket = bucket
        self.region = region
        
    @abstractmethod
    async def upload(
        self,
        key: str,
        data: Union[bytes, io.IOBase, Path],
        options: Optional[UploadOptions] = None,
    ) -> StorageObject:
        """Upload an object to storage."""
        pass
        
    @abstractmethod
    async def download(self, key: str) -> bytes:
        """Download an object from storage."""
        pass
        
    @abstractmethod
    async def download_stream(self, key: str) -> AsyncIterator[bytes]:
        """Download an object as a stream."""
        pass
        
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete an object from storage."""
        pass
        
    @abstractmethod
    async def delete_many(self, keys: List[str]) -> None:
        """Delete multiple objects from storage."""
        pass
        
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if an object exists."""
        pass
        
    @abstractmethod
    async def get_metadata(self, key: str) -> StorageObject:
        """Get object metadata."""
        pass
        
    @abstractmethod
    async def update_metadata(self, key: str, metadata: Dict[str, str]) -> None:
        """Update object metadata."""
        pass
        
    @abstractmethod
    async def list_objects(
        self,
        prefix: Optional[str] = None,
        delimiter: Optional[str] = None,
        max_keys: Optional[int] = None,
        continuation_token: Optional[str] = None,
    ) -> tuple[List[StorageObject], Optional[str]]:
        """List objects in the bucket."""
        pass
        
    @abstractmethod
    async def copy(
        self,
        source_key: str,
        dest_key: str,
        source_bucket: Optional[str] = None,
        options: Optional[UploadOptions] = None,
    ) -> StorageObject:
        """Copy an object within or between buckets."""
        pass
        
    @abstractmethod
    async def move(
        self,
        source_key: str,
        dest_key: str,
        source_bucket: Optional[str] = None,
    ) -> StorageObject:
        """Move an object (copy and delete)."""
        pass
        
    @abstractmethod
    async def create_signed_url(
        self,
        key: str,
        options: Optional[SignedUrlOptions] = None,
    ) -> str:
        """Create a signed URL for temporary access."""
        pass
        
    @abstractmethod
    async def create_multipart_upload(
        self,
        key: str,
        options: Optional[UploadOptions] = None,
    ) -> str:
        """Initiate a multipart upload."""
        pass
        
    @abstractmethod
    async def upload_part(
        self,
        key: str,
        upload_id: str,
        part_number: int,
        data: bytes,
    ) -> str:
        """Upload a part in a multipart upload."""
        pass
        
    @abstractmethod
    async def complete_multipart_upload(
        self,
        key: str,
        upload_id: str,
        parts: List[Dict[str, Union[int, str]]],
    ) -> StorageObject:
        """Complete a multipart upload."""
        pass
        
    @abstractmethod
    async def abort_multipart_upload(self, key: str, upload_id: str) -> None:
        """Abort a multipart upload."""
        pass
        
    @abstractmethod
    async def set_lifecycle_policy(self, rules: List[Dict[str, any]]) -> None:
        """Set lifecycle policy for the bucket."""
        pass
        
    @abstractmethod
    async def get_lifecycle_policy(self) -> List[Dict[str, any]]:
        """Get lifecycle policy for the bucket."""
        pass
        
    @abstractmethod
    async def enable_versioning(self) -> None:
        """Enable versioning for the bucket."""
        pass
        
    @abstractmethod
    async def disable_versioning(self) -> None:
        """Disable versioning for the bucket."""
        pass
        
    @abstractmethod
    async def list_versions(
        self,
        prefix: Optional[str] = None,
        max_keys: Optional[int] = None,
    ) -> List[StorageObject]:
        """List all versions of objects."""
        pass