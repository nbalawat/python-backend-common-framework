"""Custom exception classes."""

from typing import Any, Dict, Optional


class CommonsError(Exception):
    """Base exception for all Commons errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary."""
        return {
            "error": self.error_code,
            "message": self.message,
            "context": self.context,
        }


class ConfigError(CommonsError):
    """Configuration-related errors."""
    
    def __init__(self, message: str, key: Optional[str] = None) -> None:
        context = {"key": key} if key else {}
        super().__init__(message, "CONFIG_ERROR", context)


class ValidationError(CommonsError):
    """Validation errors."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
    ) -> None:
        context = {}
        if field:
            context["field"] = field
        if value is not None:
            context["value"] = value
        super().__init__(message, "VALIDATION_ERROR", context)


class RetryableError(CommonsError):
    """Error that can be retried."""
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        context = {}
        if retry_after is not None:
            context["retry_after"] = retry_after
        if max_retries is not None:
            context["max_retries"] = max_retries
        super().__init__(message, "RETRYABLE_ERROR", context)


class CircuitBreakerError(CommonsError):
    """Circuit breaker is open."""
    
    def __init__(self, message: str, service: str, reset_timeout: float) -> None:
        super().__init__(
            message,
            "CIRCUIT_BREAKER_OPEN",
            {"service": service, "reset_timeout": reset_timeout}
        )