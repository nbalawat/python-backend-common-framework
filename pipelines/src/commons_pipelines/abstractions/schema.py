"""Schema definitions for pipelines."""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass


class DataType(Enum):
    """Supported data types."""
    
    # Primitive types
    BOOLEAN = "boolean"
    INT = "int"
    LONG = "long"
    FLOAT = "float"
    DOUBLE = "double"
    DECIMAL = "decimal"
    STRING = "string"
    BINARY = "binary"
    
    # Date/time types
    DATE = "date"
    TIMESTAMP = "timestamp"
    TIME = "time"
    
    # Complex types
    ARRAY = "array"
    MAP = "map"
    STRUCT = "struct"
    
    # Special types
    NULL = "null"
    UNKNOWN = "unknown"
    
    @classmethod
    def from_python_type(cls, py_type: type) -> "DataType":
        """Convert Python type to DataType."""
        mapping = {
            bool: cls.BOOLEAN,
            int: cls.LONG,
            float: cls.DOUBLE,
            str: cls.STRING,
            bytes: cls.BINARY,
            list: cls.ARRAY,
            dict: cls.MAP,
            type(None): cls.NULL,
        }
        return mapping.get(py_type, cls.UNKNOWN)


@dataclass
class Field:
    """Schema field definition."""
    
    name: str
    data_type: Union[DataType, "StructType", "ArrayType", "MapType"]
    nullable: bool = True
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
            
    def with_metadata(self, key: str, value: Any) -> "Field":
        """Add metadata to field."""
        self.metadata[key] = value
        return self
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "type": self._type_to_dict(self.data_type),
            "nullable": self.nullable,
            "metadata": self.metadata,
        }
        
    def _type_to_dict(self, dtype: Any) -> Any:
        """Convert data type to dict."""
        if isinstance(dtype, DataType):
            return dtype.value
        elif hasattr(dtype, "to_dict"):
            return dtype.to_dict()
        return str(dtype)


@dataclass
class StructType:
    """Struct type with named fields."""
    
    fields: List[Field]
    
    def add(self, field: Field) -> "StructType":
        """Add field to struct."""
        self.fields.append(field)
        return self
        
    def field_names(self) -> List[str]:
        """Get field names."""
        return [f.name for f in self.fields]
        
    def field_map(self) -> Dict[str, Field]:
        """Get field map."""
        return {f.name: f for f in self.fields}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": "struct",
            "fields": [f.to_dict() for f in self.fields],
        }


@dataclass
class ArrayType:
    """Array type with element type."""
    
    element_type: Union[DataType, StructType, "ArrayType", "MapType"]
    contains_null: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": "array",
            "element_type": self._type_to_dict(self.element_type),
            "contains_null": self.contains_null,
        }
        
    def _type_to_dict(self, dtype: Any) -> Any:
        """Convert data type to dict."""
        if isinstance(dtype, DataType):
            return dtype.value
        elif hasattr(dtype, "to_dict"):
            return dtype.to_dict()
        return str(dtype)


@dataclass
class MapType:
    """Map type with key and value types."""
    
    key_type: DataType
    value_type: Union[DataType, StructType, ArrayType, "MapType"]
    value_contains_null: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": "map",
            "key_type": self.key_type.value,
            "value_type": self._type_to_dict(self.value_type),
            "value_contains_null": self.value_contains_null,
        }
        
    def _type_to_dict(self, dtype: Any) -> Any:
        """Convert data type to dict."""
        if isinstance(dtype, DataType):
            return dtype.value
        elif hasattr(dtype, "to_dict"):
            return dtype.to_dict()
        return str(dtype)


class Schema:
    """Data schema definition."""
    
    def __init__(self, fields: Union[List[Field], List[tuple], StructType]) -> None:
        if isinstance(fields, StructType):
            self.struct = fields
        elif isinstance(fields, list) and fields and isinstance(fields[0], tuple):
            # Convert tuples to fields
            self.struct = StructType([
                Field(name, dtype) if isinstance(dtype, DataType) else Field(name, DataType.STRING)
                for name, dtype in fields
            ])
        else:
            self.struct = StructType(fields)
            
    @property
    def fields(self) -> List[Field]:
        """Get schema fields."""
        return self.struct.fields
        
    @property
    def field_names(self) -> List[str]:
        """Get field names."""
        return self.struct.field_names()
        
    def add_column(
        self,
        name: str,
        data_type: Union[DataType, StructType, ArrayType, MapType],
        nullable: bool = True,
    ) -> "Schema":
        """Add column to schema."""
        new_fields = self.fields.copy()
        new_fields.append(Field(name, data_type, nullable))
        return Schema(new_fields)
        
    def drop_column(self, name: str) -> "Schema":
        """Drop column from schema."""
        new_fields = [f for f in self.fields if f.name != name]
        return Schema(new_fields)
        
    def rename_column(self, old_name: str, new_name: str) -> "Schema":
        """Rename column."""
        new_fields = []
        for field in self.fields:
            if field.name == old_name:
                new_field = Field(new_name, field.data_type, field.nullable, field.metadata)
                new_fields.append(new_field)
            else:
                new_fields.append(field)
        return Schema(new_fields)
        
    def select(self, *columns: str) -> "Schema":
        """Select subset of columns."""
        field_map = self.struct.field_map()
        new_fields = [field_map[col] for col in columns if col in field_map]
        return Schema(new_fields)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.struct.to_dict()
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Schema":
        """Create schema from dictionary."""
        fields = []
        for field_dict in data.get("fields", []):
            field = Field(
                name=field_dict["name"],
                data_type=cls._parse_type(field_dict["type"]),
                nullable=field_dict.get("nullable", True),
                metadata=field_dict.get("metadata", {}),
            )
            fields.append(field)
        return cls(fields)
        
    @classmethod
    def _parse_type(cls, type_def: Union[str, Dict[str, Any]]) -> Any:
        """Parse type definition."""
        if isinstance(type_def, str):
            return DataType(type_def)
        elif isinstance(type_def, dict):
            type_name = type_def["type"]
            if type_name == "struct":
                fields = []
                for f in type_def["fields"]:
                    fields.append(Field(
                        f["name"],
                        cls._parse_type(f["type"]),
                        f.get("nullable", True),
                        f.get("metadata", {}),
                    ))
                return StructType(fields)
            elif type_name == "array":
                return ArrayType(
                    cls._parse_type(type_def["element_type"]),
                    type_def.get("contains_null", True),
                )
            elif type_name == "map":
                return MapType(
                    DataType(type_def["key_type"]),
                    cls._parse_type(type_def["value_type"]),
                    type_def.get("value_contains_null", True),
                )
        return DataType.UNKNOWN