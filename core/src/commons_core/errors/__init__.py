"""Error handling module with retry and circuit breaker patterns."""

from .exceptions import (
    CommonsError,
    ConfigError,
    ValidationError,
    RetryableError,
    CircuitBreakerError,
)
from .handlers import ErrorHandler, GlobalErrorHandler
from .retry import retry, retry_async, RetryConfig

try:
    from .circuit_breaker import CircuitBreaker
except ImportError:
    # pybreaker might not be installed
    CircuitBreaker = None

__all__ = [
    "CommonsError",
    "ConfigError",
    "ValidationError",
    "RetryableError",
    "CircuitBreakerError",
    "ErrorHandler",
    "GlobalErrorHandler",
    "retry",
    "retry_async",
    "RetryConfig",
    "CircuitBreaker",
]