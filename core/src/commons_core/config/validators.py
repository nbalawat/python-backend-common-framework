"""Configuration validation utilities."""

from typing import Any, Dict, Optional, Type, TypeVar
from pydantic import BaseModel as PydanticBaseModel, Field, validator
from pydantic_settings import BaseSettings

from ..errors import ValidationError


class BaseConfig(BaseSettings):
    """Base configuration model with validation."""
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_default = True
        extra = "ignore"
        
    def validate_complete(self) -> None:
        """Validate that all required fields are present."""
        errors = []
        for field_name, field_info in self.model_fields.items():
            if field_info.is_required and getattr(self, field_name) is None:
                errors.append(f"Missing required field: {field_name}")
        
        if errors:
            raise ValidationError(f"Configuration validation failed: {', '.join(errors)}")


class ConfigValidator:
    """Configuration validator for schema validation."""
    
    def __init__(self, schema: Dict[str, Any]) -> None:
        self.schema = schema
        
    def validate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration against schema."""
        validated = {}
        errors = []
        
        for key, rules in self.schema.items():
            value = config.get(key)
            
            # Check required fields
            if rules.get("required", False) and value is None:
                errors.append(f"Missing required field: {key}")
                continue
                
            # Skip optional fields with no value
            if value is None:
                if "default" in rules:
                    validated[key] = rules["default"]
                continue
                
            # Type validation
            expected_type = rules.get("type")
            if expected_type and not self._check_type(value, expected_type):
                errors.append(f"Invalid type for {key}: expected {expected_type}, got {type(value).__name__}")
                continue
                
            # Range validation for numbers
            if isinstance(value, (int, float)):
                if "min" in rules and value < rules["min"]:
                    errors.append(f"Value for {key} below minimum: {value} < {rules['min']}")
                if "max" in rules and value > rules["max"]:
                    errors.append(f"Value for {key} above maximum: {value} > {rules['max']}")
                    
            # Length validation for strings
            if isinstance(value, str):
                if "min_length" in rules and len(value) < rules["min_length"]:
                    errors.append(f"Value for {key} too short: {len(value)} < {rules['min_length']}")
                if "max_length" in rules and len(value) > rules["max_length"]:
                    errors.append(f"Value for {key} too long: {len(value)} > {rules['max_length']}")
                    
            # Pattern validation
            if "pattern" in rules and isinstance(value, str):
                import re
                if not re.match(rules["pattern"], value):
                    errors.append(f"Value for {key} doesn't match pattern: {rules['pattern']}")
                    
            # Enum validation
            if "enum" in rules and value not in rules["enum"]:
                errors.append(f"Value for {key} not in allowed values: {rules['enum']}")
                
            validated[key] = value
            
        if errors:
            raise ValidationError(f"Configuration validation failed: {'; '.join(errors)}")
            
        return validated
        
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_map = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
        }
        
        expected_python_type = type_map.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
            
        return True