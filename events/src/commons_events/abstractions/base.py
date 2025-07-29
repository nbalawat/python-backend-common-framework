"""Base event abstractions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Set, Type, TypeVar
from uuid import uuid4

from commons_core.types import BaseModel

T = TypeVar("T", bound="Event")


@dataclass
class EventMetadata:
    """Event metadata."""
    
    event_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: Optional[str] = None
    event_version: str = "1.0.0"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    source: Optional[str] = None
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "event_version": self.event_version,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "source": self.source,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
        }


class Event(BaseModel):
    """Base event class."""
    
    event_type: str
    data: Dict[str, Any]
    source: str
    metadata: Optional[Dict[str, Any]] = None
    
    def __init__(self, event_type: str, data: Dict[str, Any], source: str, **kwargs):
        super().__init__(
            event_type=event_type,
            data=data,
            source=source,
            metadata=kwargs.get('metadata', {}),
            **kwargs
        )
            
    @classmethod
    def get_event_type(cls) -> str:
        """Get event type name."""
        return cls.__name__
        
    @classmethod
    def get_event_version(cls) -> str:
        """Get event version."""
        return getattr(cls, "__event_version__", "1.0.0")
        
    def serialize(self, format: str = "json") -> bytes:
        """Serialize event."""
        if format == "json":
            return self.model_dump_json().encode()
        elif format == "avro":
            # Avro serialization would go here
            raise NotImplementedError("Avro serialization not implemented")
        elif format == "protobuf":
            # Protobuf serialization would go here
            raise NotImplementedError("Protobuf serialization not implemented")
        else:
            raise ValueError(f"Unknown format: {format}")
            
    @classmethod
    def deserialize(cls: Type[T], data: bytes, format: str = "json") -> T:
        """Deserialize event."""
        if format == "json":
            return cls.model_validate_json(data.decode())
        elif format == "avro":
            # Avro deserialization would go here
            raise NotImplementedError("Avro deserialization not implemented")
        elif format == "protobuf":
            # Protobuf deserialization would go here
            raise NotImplementedError("Protobuf deserialization not implemented")
        else:
            raise ValueError(f"Unknown format: {format}")


def EventSchema(
    name: str,
    version: str = "1.0.0",
    schema_type: str = "json",
    schema: Optional[Dict[str, Any]] = None,
):
    """Decorator to define event schema."""
    def decorator(cls: Type[Event]) -> Type[Event]:
        cls.__event_name__ = name
        cls.__event_version__ = version
        cls.__schema_type__ = schema_type
        cls.__schema__ = schema
        return cls
    return decorator


class EventHandler(ABC):
    """Base event handler interface."""
    
    @abstractmethod
    async def handle(self, event: Event) -> None:
        """Handle event."""
        pass
        
    @abstractmethod
    def can_handle(self, event: Event) -> bool:
        """Check if handler can handle event."""
        pass