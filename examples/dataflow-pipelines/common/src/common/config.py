"""
Configuration management for Dataflow pipelines.

Provides Pydantic-based configuration models for different pipeline types
with environment-specific overrides and validation.
"""

try:
    from pydantic_settings import BaseSettings
    from pydantic import Field, field_validator
except ImportError:
    from pydantic import BaseSettings, Field
    from pydantic import validator as field_validator
from typing import Dict, Any, Optional, List, Literal
from enum import Enum
import os
import json
from pathlib import Path


class PipelineType(str, Enum):
    """Pipeline type enumeration."""
    BATCH = "batch"
    STREAMING = "streaming"


class EnvironmentType(str, Enum):
    """Environment type enumeration."""
    LOCAL = "local"
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class BaseConfig(BaseSettings):
    """Base configuration for all pipelines."""
    
    # GCP settings
    project_id: str = Field(..., description="GCP project ID")
    region: str = Field(default="us-central1", description="GCP region")
    
    # Pipeline settings
    pipeline_name: str = Field(..., description="Pipeline name")
    pipeline_type: PipelineType = Field(..., description="Pipeline type")
    environment: EnvironmentType = Field(default=EnvironmentType.LOCAL, description="Environment")
    
    # Common settings
    temp_location: str = Field(..., description="Temporary storage location")
    staging_location: str = Field(..., description="Staging location")
    
    # Performance settings
    max_num_workers: int = Field(default=10, description="Maximum number of workers")
    disk_size_gb: int = Field(default=100, description="Worker disk size in GB")
    machine_type: str = Field(default="n1-standard-2", description="Worker machine type")
    
    # Monitoring
    enable_monitoring: bool = Field(default=True, description="Enable monitoring")
    monitoring_namespace: str = Field(default="dataflow", description="Monitoring namespace")
    
    class Config:
        env_prefix = "DATAFLOW_"
        case_sensitive = False
        use_enum_values = True

    @field_validator("temp_location", "staging_location")
    @classmethod
    def validate_gcs_path(cls, v):
        """Validate GCS paths."""
        if not v.startswith("gs://"):
            raise ValueError("Location must be a valid GCS path starting with gs://")
        return v


class BatchConfig(BaseConfig):
    """Configuration for batch processing pipelines."""
    
    pipeline_type: Literal[PipelineType.BATCH] = Field(default=PipelineType.BATCH)
    
    # Input settings
    input_bucket: str = Field(..., description="Input GCS bucket")
    input_prefix: str = Field(default="", description="Input file prefix")
    file_pattern: str = Field(default="*.json", description="Input file pattern")
    
    # Processing settings
    batch_size: int = Field(default=1000, description="Batch size for processing")
    max_files_per_batch: int = Field(default=100, description="Maximum files per batch")
    
    # Schema settings
    schema_file: Optional[str] = Field(None, description="Path to schema file")
    enforce_schema: bool = Field(default=False, description="Enforce schema validation")
    
    # BigTable settings
    bigtable_instance: str = Field(..., description="BigTable instance ID")
    bigtable_table: str = Field(..., description="BigTable table ID")
    bigtable_column_family: str = Field(default="cf", description="BigTable column family")
    
    # BigQuery settings (optional)
    bigquery_dataset: Optional[str] = Field(None, description="BigQuery dataset")
    bigquery_table: Optional[str] = Field(None, description="BigQuery table")
    
    @field_validator("input_bucket")
    @classmethod
    def validate_bucket_name(cls, v):
        """Validate GCS bucket name."""
        if v.startswith("gs://"):
            v = v[5:]  # Remove gs:// prefix
        return v


class StreamingConfig(BaseConfig):
    """Configuration for streaming processing pipelines."""
    
    pipeline_type: Literal[PipelineType.STREAMING] = Field(default=PipelineType.STREAMING)
    
    # Pub/Sub settings
    subscription: str = Field(..., description="Pub/Sub subscription")
    topic: Optional[str] = Field(None, description="Pub/Sub topic (for publishing)")
    dead_letter_topic: Optional[str] = Field(None, description="Dead letter topic")
    
    # Streaming settings
    streaming: Literal[True] = Field(default=True)
    enable_streaming_engine: bool = Field(default=True, description="Enable Streaming Engine")
    
    # Windowing settings
    window_size_seconds: int = Field(default=300, description="Window size in seconds")
    allowed_lateness_seconds: int = Field(default=3600, description="Allowed lateness in seconds")
    trigger_frequency_seconds: int = Field(default=60, description="Trigger frequency in seconds")
    
    # BigTable streaming settings
    bigtable_instance: str = Field(..., description="BigTable instance ID")
    bigtable_table: str = Field(..., description="BigTable table ID")
    bigtable_column_family: str = Field(default="cf", description="BigTable column family")
    bigtable_batch_size: int = Field(default=100, description="BigTable batch size")
    bigtable_flush_interval_seconds: int = Field(default=10, description="BigTable flush interval")
    
    # Error handling
    max_retry_attempts: int = Field(default=3, description="Maximum retry attempts")
    retry_delay_seconds: int = Field(default=5, description="Retry delay in seconds")
    
    @field_validator("subscription")
    @classmethod
    def validate_subscription(cls, v):
        """Validate Pub/Sub subscription format."""
        if not v.startswith("projects/"):
            raise ValueError("Subscription must be in format: projects/{project}/subscriptions/{name}")
        return v


class PipelineConfig:
    """Factory for creating pipeline configurations."""
    
    @staticmethod
    def create_batch_config(
        pipeline_name: str,
        project_id: str,
        input_bucket: str,
        bigtable_instance: str,
        bigtable_table: str,
        environment: EnvironmentType = EnvironmentType.LOCAL,
        **kwargs
    ) -> BatchConfig:
        """Create a batch pipeline configuration."""
        
        # Set default locations based on project
        temp_location = kwargs.get("temp_location", f"gs://{project_id}-dataflow-temp/")
        staging_location = kwargs.get("staging_location", f"gs://{project_id}-dataflow-staging/")
        
        config_data = {
            "pipeline_name": pipeline_name,
            "project_id": project_id,
            "environment": environment,
            "temp_location": temp_location,
            "staging_location": staging_location,
            "input_bucket": input_bucket,
            "bigtable_instance": bigtable_instance,
            "bigtable_table": bigtable_table,
            **kwargs
        }
        
        return BatchConfig(**config_data)
    
    @staticmethod
    def create_streaming_config(
        pipeline_name: str,
        project_id: str,
        subscription: str,
        bigtable_instance: str,
        bigtable_table: str,
        environment: EnvironmentType = EnvironmentType.LOCAL,
        **kwargs
    ) -> StreamingConfig:
        """Create a streaming pipeline configuration."""
        
        # Set default locations based on project
        temp_location = kwargs.get("temp_location", f"gs://{project_id}-dataflow-temp/")
        staging_location = kwargs.get("staging_location", f"gs://{project_id}-dataflow-staging/")
        
        config_data = {
            "pipeline_name": pipeline_name,
            "project_id": project_id,
            "environment": environment,
            "temp_location": temp_location,
            "staging_location": staging_location,
            "subscription": subscription,
            "bigtable_instance": bigtable_instance,
            "bigtable_table": bigtable_table,
            **kwargs
        }
        
        return StreamingConfig(**config_data)
    
    @staticmethod
    def load_from_file(config_file: str) -> BaseConfig:
        """Load configuration from JSON file."""
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_path) as f:
            config_data = json.load(f)
        
        pipeline_type = config_data.get("pipeline_type", "batch")
        
        if pipeline_type == "batch":
            return BatchConfig(**config_data)
        elif pipeline_type == "streaming":
            return StreamingConfig(**config_data)
        else:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}")


def get_pipeline_config(
    config_file: Optional[str] = None,
    pipeline_name: Optional[str] = None,
    environment: Optional[str] = None
) -> BaseConfig:
    """
    Get pipeline configuration from various sources.
    
    Priority order:
    1. Explicitly provided config file
    2. Environment-specific config file
    3. Environment variables
    4. Default values
    """
    
    # Try to load from config file
    if config_file:
        return PipelineConfig.load_from_file(config_file)
    
    # Try environment-specific config
    if pipeline_name and environment:
        env_config_file = f"deployment/configs/{environment}/{pipeline_name}.json"
        if os.path.exists(env_config_file):
            return PipelineConfig.load_from_file(env_config_file)
    
    # Try general environment config
    if environment:
        general_config_file = f"deployment/configs/{environment}.json"
        if os.path.exists(general_config_file):
            return PipelineConfig.load_from_file(general_config_file)
    
    # Fall back to environment variables and defaults
    # This will raise validation errors if required fields are missing
    try:
        return BatchConfig()
    except Exception:
        return StreamingConfig()


def create_dataflow_options(config: BaseConfig) -> Dict[str, Any]:
    """Create Dataflow pipeline options from configuration."""
    
    options = {
        "project": config.project_id,
        "region": config.region,
        "temp_location": config.temp_location,
        "staging_location": config.staging_location,
        "max_num_workers": config.max_num_workers,
        "disk_size_gb": config.disk_size_gb,
        "machine_type": config.machine_type,
        "job_name": f"{config.pipeline_name}-{config.environment}",
        "setup_file": "./setup.py",
        "save_main_session": True,
    }
    
    if isinstance(config, StreamingConfig):
        options.update({
            "streaming": True,
            "enable_streaming_engine": config.enable_streaming_engine,
        })
        
        # Add autoscaling for streaming
        options["autoscaling_algorithm"] = "THROUGHPUT_BASED"
    
    # Environment-specific settings
    if config.environment != EnvironmentType.LOCAL:
        options["runner"] = "DataflowRunner"
    else:
        options["runner"] = "DirectRunner"
    
    return options