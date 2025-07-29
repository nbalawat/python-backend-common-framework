"""Kubernetes types and models."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class ResourceSpec(BaseModel):
    """Kubernetes resource specification."""
    api_version: str = Field(..., description="API version")
    kind: str = Field(..., description="Resource kind")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Resource metadata")
    spec: Dict[str, Any] = Field(default_factory=dict, description="Resource spec")

class ResourceCondition(BaseModel):
    """Kubernetes resource condition."""
    type: str = Field(..., description="Condition type")
    status: str = Field(..., description="Condition status")
    reason: Optional[str] = Field(None, description="Condition reason")
    message: Optional[str] = Field(None, description="Condition message")
    last_transition_time: Optional[datetime] = Field(None, description="Last transition time")

class ResourceStatus(BaseModel):
    """Kubernetes resource status."""
    phase: str = Field(..., description="Resource phase")
    conditions: List[ResourceCondition] = Field(default_factory=list, description="Resource conditions")
    ready: bool = Field(False, description="Resource ready status")
    message: Optional[str] = Field(None, description="Status message")
    reason: Optional[str] = Field(None, description="Status reason")

class PodSpec(BaseModel):
    """Pod specification."""
    containers: List[Dict[str, Any]] = Field(..., description="Container specifications")
    restart_policy: str = Field("Always", description="Restart policy")
    service_account: Optional[str] = Field(None, description="Service account")
    node_selector: Dict[str, str] = Field(default_factory=dict, description="Node selector")
    tolerations: List[Dict[str, Any]] = Field(default_factory=list, description="Tolerations")
    affinity: Optional[Dict[str, Any]] = Field(None, description="Pod affinity")

class ServiceSpec(BaseModel):
    """Service specification."""
    selector: Dict[str, str] = Field(..., description="Label selector")
    ports: List[Dict[str, Any]] = Field(..., description="Service ports")
    type: str = Field("ClusterIP", description="Service type")
    external_ips: List[str] = Field(default_factory=list, description="External IPs")

class DeploymentSpec(BaseModel):
    """Deployment specification."""
    replicas: int = Field(1, description="Number of replicas")
    selector: Dict[str, Any] = Field(..., description="Label selector")
    template: Dict[str, Any] = Field(..., description="Pod template")
    strategy: Dict[str, Any] = Field(default_factory=dict, description="Deployment strategy")

class ConfigMapData(BaseModel):
    """ConfigMap data."""
    data: Dict[str, str] = Field(default_factory=dict, description="ConfigMap data")
    binary_data: Dict[str, bytes] = Field(default_factory=dict, description="Binary data")

class SecretData(BaseModel):
    """Secret data."""
    data: Dict[str, str] = Field(default_factory=dict, description="Secret data (base64 encoded)")
    string_data: Dict[str, str] = Field(default_factory=dict, description="String data")
    type: str = Field("Opaque", description="Secret type")