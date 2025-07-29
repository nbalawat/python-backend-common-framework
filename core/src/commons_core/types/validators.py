"""Custom validators for Pydantic models."""

from typing import Any, Callable, Optional
from pydantic import field_validator, model_validator
from pydantic.functional_validators import BeforeValidator, AfterValidator
import re

# Re-export pydantic validators for convenience
validator = field_validator
root_validator = model_validator


def email_validator(v: str) -> str:
    """Validate email address."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, v):
        raise ValueError(f"Invalid email address: {v}")
    return v.lower()


def url_validator(v: str) -> str:
    """Validate URL."""
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    if not re.match(pattern, v):
        raise ValueError(f"Invalid URL: {v}")
    return v


def phone_validator(v: str) -> str:
    """Validate phone number (E.164 format)."""
    pattern = r'^\+[1-9]\d{1,14}$'
    cleaned = re.sub(r'[\s\-\(\)]', '', v)
    if not re.match(pattern, cleaned):
        raise ValueError(f"Invalid phone number: {v}")
    return cleaned


def uuid_validator(v: str) -> str:
    """Validate UUID."""
    pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    if not re.match(pattern, v.lower()):
        raise ValueError(f"Invalid UUID: {v}")
    return v.lower()


def ip_address_validator(v: str) -> str:
    """Validate IP address (v4 or v6)."""
    # IPv4 pattern
    ipv4_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    # IPv6 pattern (simplified)
    ipv6_pattern = r'^([0-9a-fA-F]{0,4}:){7}[0-9a-fA-F]{0,4}$'
    
    if re.match(ipv4_pattern, v):
        # Validate IPv4 octets
        octets = v.split('.')
        if all(0 <= int(octet) <= 255 for octet in octets):
            return v
    elif re.match(ipv6_pattern, v):
        return v.lower()
        
    raise ValueError(f"Invalid IP address: {v}")


def port_validator(v: int) -> int:
    """Validate port number."""
    if not 1 <= v <= 65535:
        raise ValueError(f"Port must be between 1 and 65535, got {v}")
    return v


def percentage_validator(v: float) -> float:
    """Validate percentage (0-100)."""
    if not 0 <= v <= 100:
        raise ValueError(f"Percentage must be between 0 and 100, got {v}")
    return v


def positive_validator(v: float) -> float:
    """Validate positive number."""
    if v <= 0:
        raise ValueError(f"Value must be positive, got {v}")
    return v


def non_negative_validator(v: float) -> float:
    """Validate non-negative number."""
    if v < 0:
        raise ValueError(f"Value must be non-negative, got {v}")
    return v


def length_validator(min_length: Optional[int] = None, max_length: Optional[int] = None) -> Callable:
    """Create a length validator."""
    def validate(v: str) -> str:
        if min_length is not None and len(v) < min_length:
            raise ValueError(f"Length must be at least {min_length}, got {len(v)}")
        if max_length is not None and len(v) > max_length:
            raise ValueError(f"Length must be at most {max_length}, got {len(v)}")
        return v
    return validate


def regex_validator(pattern: str, flags: int = 0) -> Callable:
    """Create a regex validator."""
    compiled = re.compile(pattern, flags)
    
    def validate(v: str) -> str:
        if not compiled.match(v):
            raise ValueError(f"Value does not match pattern {pattern}: {v}")
        return v
    return validate


def enum_validator(allowed_values: list) -> Callable:
    """Create an enum validator."""
    def validate(v: Any) -> Any:
        if v not in allowed_values:
            raise ValueError(f"Value must be one of {allowed_values}, got {v}")
        return v
    return validate