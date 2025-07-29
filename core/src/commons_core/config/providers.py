"""Configuration providers for different sources."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Set
import os
import json
import yaml
from dotenv import dotenv_values

from ..errors import ConfigError


class ConfigProvider(ABC):
    """Base configuration provider interface."""
    
    @abstractmethod
    def load(self) -> Dict[str, Any]:
        """Load configuration from source."""
        pass


class EnvProvider(ConfigProvider):
    """Environment variable configuration provider."""
    
    def __init__(self, prefix: str = "", delimiter: str = "__") -> None:
        self.prefix = prefix.upper() if prefix else ""
        self.delimiter = delimiter
        
    def load(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        
        # Load from .env file if exists
        if Path(".env").exists():
            env_vars = dotenv_values(".env")
            for key, value in env_vars.items():
                if self._should_include(key):
                    self._set_nested_value(config, self._parse_key(key), self._parse_value(value))
        
        # Load from actual environment
        for key, value in os.environ.items():
            if self._should_include(key):
                self._set_nested_value(config, self._parse_key(key), self._parse_value(value))
                
        return config
        
    def _should_include(self, key: str) -> bool:
        """Check if environment variable should be included."""
        return not self.prefix or key.startswith(self.prefix)
        
    def _parse_key(self, key: str) -> str:
        """Parse environment variable key to nested path."""
        if self.prefix and key.startswith(self.prefix):
            key = key[len(self.prefix):].lstrip("_")
        return key.lower().replace(self.delimiter, ".")
        
    def _parse_value(self, value: Optional[str]) -> Any:
        """Parse environment variable value."""
        if value is None:
            return None
            
        # Try to parse as JSON
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            pass
            
        # Check for boolean values
        if value.lower() in ("true", "yes", "1", "on"):
            return True
        elif value.lower() in ("false", "no", "0", "off"):
            return False
            
        # Check for numeric values
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
            
        return value
        
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any) -> None:
        """Set nested value in config dictionary."""
        keys = path.split(".")
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                return  # Skip if path conflicts with existing value
            current = current[key]
            
        current[keys[-1]] = value


class FileProvider(ConfigProvider):
    """File-based configuration provider (YAML/JSON)."""
    
    def __init__(self, path: Path, format: Optional[str] = None) -> None:
        self.path = path
        self.format = format or self._detect_format()
        
    def load(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if not self.path.exists():
            raise ConfigError(f"Configuration file not found: {self.path}")
            
        try:
            with open(self.path, "r") as f:
                if self.format == "yaml":
                    return yaml.safe_load(f) or {}
                elif self.format == "json":
                    return json.load(f)
                else:
                    raise ConfigError(f"Unsupported file format: {self.format}")
        except Exception as e:
            raise ConfigError(f"Failed to load configuration from {self.path}: {e}")
            
    def _detect_format(self) -> str:
        """Detect file format from extension."""
        suffix = self.path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            return "yaml"
        elif suffix == ".json":
            return "json"
        else:
            raise ConfigError(f"Cannot detect format for file: {self.path}")


class DictProvider(ConfigProvider):
    """Dictionary-based configuration provider."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        
    def load(self) -> Dict[str, Any]:
        """Return the dictionary configuration."""
        return self.config.copy()