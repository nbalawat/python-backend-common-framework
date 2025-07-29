"""Logger implementation with structured logging support."""

import logging
import sys
from contextlib import contextmanager
from datetime import datetime
from time import time
from typing import Any, Dict, List, Optional, Union
import structlog
from structlog.types import FilteringBoundLogger, WrappedLogger

from .formatters import JsonFormatter
from .handlers import get_handler


# Global logger cache
_loggers: Dict[str, "Logger"] = {}
_configured = False


class Logger:
    """Enhanced logger with structured logging capabilities."""
    
    def __init__(self, name: str, context: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.context = context or {}
        self._logger = structlog.get_logger(name).bind(**self.context)
        
    def bind(self, **kwargs: Any) -> "Logger":
        """Create new logger with additional context."""
        new_context = {**self.context, **kwargs}
        return Logger(self.name, new_context)
        
    def unbind(self, *keys: str) -> "Logger":
        """Create new logger with context keys removed."""
        new_context = {k: v for k, v in self.context.items() if k not in keys}
        return Logger(self.name, new_context)
        
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._logger.debug(message, **self._merge_context(kwargs))
        
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._logger.info(message, **self._merge_context(kwargs))
        
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._logger.warning(message, **self._merge_context(kwargs))
        
    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self._logger.error(message, **self._merge_context(kwargs))
        
    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        self._logger.critical(message, **self._merge_context(kwargs))
        
    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        self._logger.exception(message, **self._merge_context(kwargs))
        
    @contextmanager
    def timer(self, operation: str, **kwargs: Any):
        """Context manager for timing operations."""
        start_time = time()
        self.info(f"{operation} started", operation=operation, **kwargs)
        
        try:
            yield
        except Exception as e:
            self.error(
                f"{operation} failed",
                operation=operation,
                duration_ms=int((time() - start_time) * 1000),
                error=str(e),
                **kwargs
            )
            raise
        else:
            self.info(
                f"{operation} completed",
                operation=operation,
                duration_ms=int((time() - start_time) * 1000),
                **kwargs
            )
            
    def _merge_context(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge logger context with provided kwargs."""
        return {**self.context, **kwargs}


def configure_logging(
    level: Union[str, int] = "INFO",
    format: str = "json",
    handlers: Optional[List[str]] = None,
    context: Optional[Dict[str, Any]] = None,
    correlation_id_header: str = "X-Correlation-ID",
) -> None:
    """Configure global logging settings."""
    global _configured
    
    if _configured:
        return
        
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            _add_global_context(context or {}),
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                ],
            ),
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer() if format == "json" else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Add handlers
    if handlers is None:
        handlers = ["console"]
        
    for handler_name in handlers:
        handler = get_handler(handler_name, format=format)
        if handler:
            root_logger.addHandler(handler)
            
    _configured = True


def _add_global_context(context: Dict[str, Any]):
    """Processor to add global context to all log entries."""
    def processor(logger: WrappedLogger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        event_dict.update(context)
        return event_dict
    return processor


def get_logger(name: str, **context: Any) -> Logger:
    """Get or create a logger instance."""
    if not _configured:
        configure_logging()
        
    cache_key = f"{name}:{hash(frozenset(context.items()))}"
    
    if cache_key not in _loggers:
        _loggers[cache_key] = Logger(name, context)
        
    return _loggers[cache_key]