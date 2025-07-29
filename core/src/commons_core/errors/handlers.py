"""Error handlers for structured error management."""

import sys
import traceback
from typing import Any, Callable, Dict, Optional, Type, Tuple
from functools import wraps
import logging

from .exceptions import CommonsError

logger = logging.getLogger(__name__)


class ErrorHandler:
    """Handle errors with context and logging."""
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        include_traceback: bool = True,
        reraise: bool = True,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.include_traceback = include_traceback
        self.reraise = reraise
        
    def handle(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        operation: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Handle an error with logging and context."""
        error_dict = self._error_to_dict(error, context, operation)
        
        # Log the error
        self.logger.error(
            f"Error in {operation or 'operation'}: {error}",
            extra=error_dict,
            exc_info=self.include_traceback
        )
        
        if self.reraise:
            raise
            
        return error_dict
        
    def _error_to_dict(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        operation: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convert error to dictionary format."""
        error_dict = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "operation": operation,
        }
        
        if isinstance(error, CommonsError):
            error_dict.update(error.to_dict())
            
        if context:
            error_dict["context"] = context
            
        if self.include_traceback:
            error_dict["traceback"] = traceback.format_exc()
            
        return error_dict
        
    def __call__(
        self,
        operation: Optional[str] = None,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
        context_func: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> Callable:
        """Decorator for error handling."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    context = {}
                    if context_func:
                        context = context_func(*args, **kwargs)
                    
                    op_name = operation or func.__name__
                    self.handle(e, context, op_name)
                    
            return wrapper
        return decorator


class GlobalErrorHandler:
    """Global error handler for uncaught exceptions."""
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        exit_on_error: bool = True,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.exit_on_error = exit_on_error
        self._original_excepthook = None
        
    def install(self) -> None:
        """Install global error handler."""
        self._original_excepthook = sys.excepthook
        sys.excepthook = self._handle_exception
        
    def uninstall(self) -> None:
        """Uninstall global error handler."""
        if self._original_excepthook:
            sys.excepthook = self._original_excepthook
            
    def _handle_exception(
        self,
        exc_type: Type[BaseException],
        exc_value: BaseException,
        exc_traceback: Any,
    ) -> None:
        """Handle uncaught exceptions."""
        if issubclass(exc_type, KeyboardInterrupt):
            # Allow keyboard interrupt to work normally
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
            
        self.logger.critical(
            f"Uncaught exception: {exc_value}",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
        
        if self.exit_on_error:
            sys.exit(1)