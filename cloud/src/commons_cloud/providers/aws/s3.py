"""AWS S3 storage implementation."""

from datetime import datetime, timedelta
from typing import AsyncIterator, Dict, List, Optional, Set, Union
from pathlib import Path
import io
import aioboto3
from botocore.exceptions import ClientError

from commons_core.logging import get_logger
from ...abstractions.storage import (
    StorageProvider,
    StorageObject,
    UploadOptions,
    SignedUrlOptions,
)

logger = get_logger(__name__)


class S3Storage(StorageProvider):
    """AWS S3 storage provider."""
    
    def __init__(
        self,
        bucket: str,
        region: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        session_token: Optional[str] = None,
    ) -> None:
        super().__init__(bucket, region)
        self.session = aioboto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            aws_session_token=session_token,
            region_name=region,
        )
        
    async def upload(
        self,
        key: str,
        data: Union[bytes, io.IOBase, Path],
        options: Optional[UploadOptions] = None,
    ) -> StorageObject:
        """Upload an object to S3."""
        options = options or UploadOptions()
        
        async with self.session.client("s3") as s3:
            extra_args = {}
            
            if options.content_type:
                extra_args["ContentType"] = options.content_type
            if options.metadata:
                extra_args["Metadata"] = options.metadata
            if options.storage_class:
                extra_args["StorageClass"] = options.storage_class
            if options.encryption:
                extra_args["ServerSideEncryption"] = options.encryption
            if options.cache_control:
                extra_args["CacheControl"] = options.cache_control
            if options.content_disposition:
                extra_args["ContentDisposition"] = options.content_disposition
            if options.content_encoding:
                extra_args["ContentEncoding"] = options.content_encoding
            if options.expires:
                extra_args["Expires"] = options.expires
            if options.acl:
                extra_args["ACL"] = options.acl
                
            # Handle different data types
            if isinstance(data, bytes):
                await s3.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=data,
                    **extra_args
                )
            elif isinstance(data, Path):
                with open(data, "rb") as f:
                    await s3.upload_fileobj(f, self.bucket, key, ExtraArgs=extra_args)
            else:
                await s3.upload_fileobj(data, self.bucket, key, ExtraArgs=extra_args)
                
            # Get object metadata
            response = await s3.head_object(Bucket=self.bucket, Key=key)
            
            return StorageObject(
                key=key,
                size=response["ContentLength"],
                last_modified=response["LastModified"],
                etag=response["ETag"].strip('"'),
                storage_class=response.get("StorageClass"),
                metadata=response.get("Metadata"),
                content_type=response.get("ContentType"),
            )
            
    async def download(self, key: str) -> bytes:
        """Download an object from S3."""
        async with self.session.client("s3") as s3:
            try:
                response = await s3.get_object(Bucket=self.bucket, Key=key)
                return await response["Body"].read()
            except ClientError as e:
                if e.response["Error"]["Code"] == "NoSuchKey":
                    raise FileNotFoundError(f"Object not found: {key}")
                raise
                
    async def download_stream(self, key: str) -> AsyncIterator[bytes]:
        """Download an object as a stream."""
        async with self.session.client("s3") as s3:
            try:
                response = await s3.get_object(Bucket=self.bucket, Key=key)
                async with response["Body"] as stream:
                    async for chunk in stream.iter_chunks():
                        yield chunk
            except ClientError as e:
                if e.response["Error"]["Code"] == "NoSuchKey":
                    raise FileNotFoundError(f"Object not found: {key}")
                raise
                
    async def delete(self, key: str) -> None:
        """Delete an object from S3."""
        async with self.session.client("s3") as s3:
            await s3.delete_object(Bucket=self.bucket, Key=key)
            
    async def delete_many(self, keys: List[str]) -> None:
        """Delete multiple objects from S3."""
        async with self.session.client("s3") as s3:
            # S3 allows max 1000 objects per delete request
            for i in range(0, len(keys), 1000):
                batch = keys[i:i + 1000]
                objects = [{"Key": key} for key in batch]
                
                await s3.delete_objects(
                    Bucket=self.bucket,
                    Delete={"Objects": objects}
                )
                
    async def exists(self, key: str) -> bool:
        """Check if an object exists in S3."""
        async with self.session.client("s3") as s3:
            try:
                await s3.head_object(Bucket=self.bucket, Key=key)
                return True
            except ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    return False
                raise
                
    async def get_metadata(self, key: str) -> StorageObject:
        """Get object metadata."""
        async with self.session.client("s3") as s3:
            try:
                response = await s3.head_object(Bucket=self.bucket, Key=key)
                
                return StorageObject(
                    key=key,
                    size=response["ContentLength"],
                    last_modified=response["LastModified"],
                    etag=response["ETag"].strip('"'),
                    storage_class=response.get("StorageClass"),
                    metadata=response.get("Metadata"),
                    content_type=response.get("ContentType"),
                )
            except ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    raise FileNotFoundError(f"Object not found: {key}")
                raise
                
    async def update_metadata(self, key: str, metadata: Dict[str, str]) -> None:
        """Update object metadata."""
        async with self.session.client("s3") as s3:
            # S3 requires copying the object to update metadata
            await s3.copy_object(
                Bucket=self.bucket,
                Key=key,
                CopySource={"Bucket": self.bucket, "Key": key},
                Metadata=metadata,
                MetadataDirective="REPLACE",
            )
            
    async def list_objects(
        self,
        prefix: Optional[str] = None,
        delimiter: Optional[str] = None,
        max_keys: Optional[int] = None,
        continuation_token: Optional[str] = None,
    ) -> tuple[List[StorageObject], Optional[str]]:
        """List objects in the bucket."""
        async with self.session.client("s3") as s3:
            params = {
                "Bucket": self.bucket,
                "MaxKeys": max_keys or 1000,
            }
            
            if prefix:
                params["Prefix"] = prefix
            if delimiter:
                params["Delimiter"] = delimiter
            if continuation_token:
                params["ContinuationToken"] = continuation_token
                
            response = await s3.list_objects_v2(**params)
            
            objects = []
            for obj in response.get("Contents", []):
                objects.append(
                    StorageObject(
                        key=obj["Key"],
                        size=obj["Size"],
                        last_modified=obj["LastModified"],
                        etag=obj["ETag"].strip('"'),
                        storage_class=obj.get("StorageClass"),
                    )
                )
                
            next_token = response.get("NextContinuationToken")
            return objects, next_token
            
    async def copy(
        self,
        source_key: str,
        dest_key: str,
        source_bucket: Optional[str] = None,
        options: Optional[UploadOptions] = None,
    ) -> StorageObject:
        """Copy an object within or between buckets."""
        options = options or UploadOptions()
        source_bucket = source_bucket or self.bucket
        
        async with self.session.client("s3") as s3:
            copy_source = {"Bucket": source_bucket, "Key": source_key}
            
            extra_args = {}
            if options.metadata:
                extra_args["Metadata"] = options.metadata
                extra_args["MetadataDirective"] = "REPLACE"
            if options.storage_class:
                extra_args["StorageClass"] = options.storage_class
            if options.encryption:
                extra_args["ServerSideEncryption"] = options.encryption
                
            await s3.copy_object(
                Bucket=self.bucket,
                Key=dest_key,
                CopySource=copy_source,
                **extra_args
            )
            
            return await self.get_metadata(dest_key)
            
    async def move(
        self,
        source_key: str,
        dest_key: str,
        source_bucket: Optional[str] = None,
    ) -> StorageObject:
        """Move an object (copy and delete)."""
        # Copy first
        result = await self.copy(source_key, dest_key, source_bucket)
        
        # Then delete source
        if source_bucket and source_bucket != self.bucket:
            async with self.session.client("s3") as s3:
                await s3.delete_object(Bucket=source_bucket, Key=source_key)
        else:
            await self.delete(source_key)
            
        return result
        
    async def create_signed_url(
        self,
        key: str,
        options: Optional[SignedUrlOptions] = None,
    ) -> str:
        """Create a signed URL for temporary access."""
        options = options or SignedUrlOptions()
        
        async with self.session.client("s3") as s3:
            params = {
                "Bucket": self.bucket,
                "Key": key,
            }
            
            if options.response_headers:
                for header, value in options.response_headers.items():
                    params[header] = value
                    
            url = await s3.generate_presigned_url(
                ClientMethod=f"{options.method.lower()}_object",
                Params=params,
                ExpiresIn=options.expires_in,
            )
            
            return url
            
    async def create_multipart_upload(
        self,
        key: str,
        options: Optional[UploadOptions] = None,
    ) -> str:
        """Initiate a multipart upload."""
        options = options or UploadOptions()
        
        async with self.session.client("s3") as s3:
            extra_args = {}
            
            if options.content_type:
                extra_args["ContentType"] = options.content_type
            if options.metadata:
                extra_args["Metadata"] = options.metadata
            if options.storage_class:
                extra_args["StorageClass"] = options.storage_class
                
            response = await s3.create_multipart_upload(
                Bucket=self.bucket,
                Key=key,
                **extra_args
            )
            
            return response["UploadId"]
            
    async def upload_part(
        self,
        key: str,
        upload_id: str,
        part_number: int,
        data: bytes,
    ) -> str:
        """Upload a part in a multipart upload."""
        async with self.session.client("s3") as s3:
            response = await s3.upload_part(
                Bucket=self.bucket,
                Key=key,
                UploadId=upload_id,
                PartNumber=part_number,
                Body=data,
            )
            
            return response["ETag"]
            
    async def complete_multipart_upload(
        self,
        key: str,
        upload_id: str,
        parts: List[Dict[str, Union[int, str]]],
    ) -> StorageObject:
        """Complete a multipart upload."""
        async with self.session.client("s3") as s3:
            await s3.complete_multipart_upload(
                Bucket=self.bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
            )
            
            return await self.get_metadata(key)
            
    async def abort_multipart_upload(self, key: str, upload_id: str) -> None:
        """Abort a multipart upload."""
        async with self.session.client("s3") as s3:
            await s3.abort_multipart_upload(
                Bucket=self.bucket,
                Key=key,
                UploadId=upload_id,
            )
            
    async def set_lifecycle_policy(self, rules: List[Dict[str, any]]) -> None:
        """Set lifecycle policy for the bucket."""
        async with self.session.client("s3") as s3:
            await s3.put_bucket_lifecycle_configuration(
                Bucket=self.bucket,
                LifecycleConfiguration={"Rules": rules},
            )
            
    async def get_lifecycle_policy(self) -> List[Dict[str, any]]:
        """Get lifecycle policy for the bucket."""
        async with self.session.client("s3") as s3:
            try:
                response = await s3.get_bucket_lifecycle_configuration(
                    Bucket=self.bucket
                )
                return response.get("Rules", [])
            except ClientError as e:
                if e.response["Error"]["Code"] == "NoSuchLifecycleConfiguration":
                    return []
                raise
                
    async def enable_versioning(self) -> None:
        """Enable versioning for the bucket."""
        async with self.session.client("s3") as s3:
            await s3.put_bucket_versioning(
                Bucket=self.bucket,
                VersioningConfiguration={"Status": "Enabled"},
            )
            
    async def disable_versioning(self) -> None:
        """Disable versioning for the bucket."""
        async with self.session.client("s3") as s3:
            await s3.put_bucket_versioning(
                Bucket=self.bucket,
                VersioningConfiguration={"Status": "Suspended"},
            )
            
    async def list_versions(
        self,
        prefix: Optional[str] = None,
        max_keys: Optional[int] = None,
    ) -> List[StorageObject]:
        """List all versions of objects."""
        async with self.session.client("s3") as s3:
            params = {
                "Bucket": self.bucket,
                "MaxKeys": max_keys or 1000,
            }
            
            if prefix:
                params["Prefix"] = prefix
                
            response = await s3.list_object_versions(**params)
            
            objects = []
            for version in response.get("Versions", []):
                objects.append(
                    StorageObject(
                        key=version["Key"],
                        size=version["Size"],
                        last_modified=version["LastModified"],
                        etag=version["ETag"].strip('"'),
                        storage_class=version.get("StorageClass"),
                        metadata={"VersionId": version["VersionId"]},
                    )
                )
                
            return objects