"""Serialization utilities for various formats."""

from typing import Any, Dict, Optional, Type, TypeVar, Union
import orjson
import json
from datetime import datetime, date
from decimal import Decimal
from pathlib import Path

T = TypeVar("T")


def default_json_encoder(obj: Any) -> Any:
    """Default JSON encoder for common types."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, Path):
        return str(obj)
    elif hasattr(obj, "model_dump"):
        return obj.model_dump()
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def to_json(
    obj: Any,
    pretty: bool = False,
    sort_keys: bool = False,
    default: Any = default_json_encoder,
) -> str:
    """Serialize object to JSON string."""
    options = 0
    if pretty:
        options |= orjson.OPT_INDENT_2
    if sort_keys:
        options |= orjson.OPT_SORT_KEYS
        
    return orjson.dumps(obj, default=default, option=options).decode()


def from_json(data: Union[str, bytes], model: Optional[Type[T]] = None) -> Union[Any, T]:
    """Deserialize JSON string to object."""
    parsed = orjson.loads(data)
    
    if model and hasattr(model, "model_validate"):
        return model.model_validate(parsed)
    return parsed


def to_msgpack(obj: Any) -> bytes:
    """Serialize object to MessagePack bytes."""
    try:
        import msgpack
        
        def encoder(obj):
            if isinstance(obj, (datetime, date)):
                return {"__datetime__": obj.isoformat()}
            elif isinstance(obj, Decimal):
                return {"__decimal__": str(obj)}
            elif hasattr(obj, "model_dump"):
                return obj.model_dump()
            return obj
            
        return msgpack.packb(obj, default=encoder, use_bin_type=True)
    except ImportError:
        raise ImportError("msgpack is required for MessagePack serialization")


def from_msgpack(data: bytes, model: Optional[Type[T]] = None) -> Union[Any, T]:
    """Deserialize MessagePack bytes to object."""
    try:
        import msgpack
        
        def decoder(obj):
            if "__datetime__" in obj:
                return datetime.fromisoformat(obj["__datetime__"])
            elif "__decimal__" in obj:
                return Decimal(obj["__decimal__"])
            return obj
            
        parsed = msgpack.unpackb(data, object_hook=decoder, raw=False)
        
        if model and hasattr(model, "model_validate"):
            return model.model_validate(parsed)
        return parsed
    except ImportError:
        raise ImportError("msgpack is required for MessagePack deserialization")


def to_yaml(obj: Any) -> str:
    """Serialize object to YAML string."""
    try:
        import yaml
        
        def representer(dumper, data):
            if hasattr(data, "model_dump"):
                return dumper.represent_dict(data.model_dump())
            elif isinstance(data, (datetime, date)):
                return dumper.represent_str(data.isoformat())
            elif isinstance(data, Decimal):
                return dumper.represent_float(float(data))
            return dumper.represent_data(data)
            
        yaml.add_multi_representer(object, representer)
        return yaml.dump(obj, default_flow_style=False, sort_keys=False)
    except ImportError:
        raise ImportError("PyYAML is required for YAML serialization")


def from_yaml(data: str, model: Optional[Type[T]] = None) -> Union[Any, T]:
    """Deserialize YAML string to object."""
    try:
        import yaml
        parsed = yaml.safe_load(data)
        
        if model and hasattr(model, "model_validate"):
            return model.model_validate(parsed)
        return parsed
    except ImportError:
        raise ImportError("PyYAML is required for YAML deserialization")