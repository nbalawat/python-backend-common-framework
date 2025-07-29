"""Configuration manager implementation."""

from typing import Any, Dict, List, Optional, Set, Type, TypeVar, Union
from pathlib import Path
import os
from functools import reduce

from ..errors import ConfigError
from .providers import ConfigProvider, EnvProvider, FileProvider, DictProvider
from .validators import BaseConfig

T = TypeVar("T", bound=BaseConfig)


class ConfigManager:
    """Hierarchical configuration manager with multi-source support."""
    
    def __init__(self) -> None:
        self._providers: List[ConfigProvider] = []
        self._config_cache: Dict[str, Any] = {}
        self._secrets: set[str] = set()
        
    def load_from_env(self, prefix: str = "") -> None:
        """Load configuration from environment variables."""
        provider = EnvProvider(prefix=prefix)
        self._providers.append(provider)
        self._merge_config(provider.load())
        
    def load_from_file(self, path: Union[str, Path], format: Optional[str] = None) -> None:
        """Load configuration from a file (YAML/JSON)."""
        provider = FileProvider(path=Path(path), format=format)
        self._providers.append(provider)
        self._merge_config(provider.load())
        
    def load_from_dict(self, config: Dict[str, Any]) -> None:
        """Load configuration from a dictionary."""
        provider = DictProvider(config=config)
        self._providers.append(provider)
        self._merge_config(provider.load())
        
    def get(
        self, 
        key: str, 
        default: Any = None, 
        secret: bool = False,
        required: bool = False
    ) -> Any:
        """Get configuration value by dot-notation key."""
        try:
            value = self._get_nested(self._config_cache, key)
            if secret:
                self._secrets.add(key)
            return value
        except KeyError:
            if required:
                raise ConfigError(f"Required configuration key not found: {key}")
            return default
            
    def set(self, key: str, value: Any, secret: bool = False) -> None:
        """Set configuration value."""
        self._set_nested(self._config_cache, key, value)
        if secret:
            self._secrets.add(key)
            
    def load_model(self, model_class: Type[T], prefix: str = "") -> T:
        """Load configuration into a Pydantic model."""
        config_dict = self._get_prefix_config(prefix)
        try:
            return model_class(**config_dict)
        except Exception as e:
            raise ConfigError(f"Failed to load config model {model_class.__name__}: {e}")
            
    def reload(self) -> None:
        """Reload configuration from all providers."""
        self._config_cache.clear()
        for provider in self._providers:
            self._merge_config(provider.load())
            
    def to_dict(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        if include_secrets:
            return self._config_cache.copy()
        return self._mask_secrets(self._config_cache)
        
    def _merge_config(self, new_config: Dict[str, Any]) -> None:
        """Merge new configuration with existing."""
        self._config_cache = self._deep_merge(self._config_cache, new_config)
        
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
        
    def _get_nested(self, data: Dict[str, Any], key: str) -> Any:
        """Get nested value using dot notation."""
        keys = key.split(".")
        return reduce(lambda d, k: d[k], keys, data)
        
    def _set_nested(self, data: Dict[str, Any], key: str, value: Any) -> None:
        """Set nested value using dot notation."""
        keys = key.split(".")
        for k in keys[:-1]:
            data = data.setdefault(k, {})
        data[keys[-1]] = value
        
    def _get_prefix_config(self, prefix: str) -> Dict[str, Any]:
        """Get configuration subset by prefix."""
        if not prefix:
            return self._config_cache
        try:
            return self._get_nested(self._config_cache, prefix)
        except KeyError:
            return {}
            
    def _mask_secrets(self, config: Dict[str, Any], path: str = "") -> Dict[str, Any]:
        """Mask secret values in configuration."""
        result = {}
        for key, value in config.items():
            current_path = f"{path}.{key}" if path else key
            if current_path in self._secrets:
                result[key] = "***MASKED***"
            elif isinstance(value, dict):
                result[key] = self._mask_secrets(value, current_path)
            else:
                result[key] = value
        return result