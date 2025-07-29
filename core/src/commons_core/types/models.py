"""Base Pydantic models with enhanced functionality."""

from datetime import datetime
from typing import Any, Dict, Optional, Type, TypeVar
from pydantic import BaseModel as PydanticBaseModel, ConfigDict, Field, SecretStr
import orjson

T = TypeVar("T", bound="BaseModel")


class BaseModel(PydanticBaseModel):
    """Enhanced base model with additional functionality."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        arbitrary_types_allowed=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            SecretStr: lambda v: "***MASKED***",
        },
    )
    
    def model_dump_json(self, **kwargs: Any) -> str:
        """Dump model to JSON using orjson for performance."""
        data = self.model_dump(**kwargs)
        return orjson.dumps(data).decode()
        
    @classmethod
    def model_validate_json(cls: Type[T], json_data: str, **kwargs: Any) -> T:
        """Load model from JSON using orjson."""
        data = orjson.loads(json_data)
        return cls.model_validate(data, **kwargs)
        
    def to_msgpack(self) -> bytes:
        """Serialize to MessagePack format."""
        try:
            import msgpack
            return msgpack.packb(self.model_dump())
        except ImportError:
            raise ImportError("msgpack is required for MessagePack serialization")
            
    @classmethod
    def from_msgpack(cls: Type[T], data: bytes) -> T:
        """Deserialize from MessagePack format."""
        try:
            import msgpack
            return cls.model_validate(msgpack.unpackb(data, raw=False))
        except ImportError:
            raise ImportError("msgpack is required for MessagePack deserialization")
            
    def to_yaml(self) -> str:
        """Serialize to YAML format."""
        try:
            import yaml
            return yaml.dump(self.model_dump(), default_flow_style=False)
        except ImportError:
            raise ImportError("PyYAML is required for YAML serialization")
            
    @classmethod
    def from_yaml(cls: Type[T], data: str) -> T:
        """Deserialize from YAML format."""
        try:
            import yaml
            return cls.model_validate(yaml.safe_load(data))
        except ImportError:
            raise ImportError("PyYAML is required for YAML deserialization")
            
    def diff(self, other: "BaseModel") -> Dict[str, Any]:
        """Get differences between this model and another."""
        if not isinstance(other, self.__class__):
            raise TypeError(f"Cannot diff with {type(other)}")
            
        self_dict = self.model_dump()
        other_dict = other.model_dump()
        
        diff = {}
        all_keys = set(self_dict.keys()) | set(other_dict.keys())
        
        for key in all_keys:
            self_val = self_dict.get(key)
            other_val = other_dict.get(key)
            
            if self_val != other_val:
                diff[key] = {"old": self_val, "new": other_val}
                
        return diff


class ImmutableModel(BaseModel):
    """Immutable base model (frozen)."""
    
    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        use_enum_values=True,
        arbitrary_types_allowed=True,
    )
    
    def copy_with(self, **kwargs: Any) -> "ImmutableModel":
        """Create a copy with updated fields."""
        data = self.model_dump()
        data.update(kwargs)
        return self.__class__.model_validate(data)