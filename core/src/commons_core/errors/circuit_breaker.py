"""Circuit breaker implementation for fault tolerance."""

from typing import Any, Callable, Dict, List, Optional, Type

try:
    from pybreaker import CircuitBreaker as PyBreakerCircuitBreaker
    from pybreaker import STATE_CLOSED, STATE_OPEN, STATE_HALF_OPEN
    
    class CircuitBreaker(PyBreakerCircuitBreaker):
        """Enhanced circuit breaker with additional features."""
        
        def __init__(
            self,
            failure_threshold: int = 5,
            recovery_timeout: float = 60.0,
            expected_exception: type = Exception,
            name: Optional[str] = None,
            exclude: Optional[List[Type[Exception]]] = None,
        ) -> None:
            super().__init__(
                fail_max=failure_threshold,
                reset_timeout=recovery_timeout,
                exclude=exclude or [],
                name=name,
            )
            self.expected_exception = expected_exception
            
        @property
        def is_open(self) -> bool:
            """Check if circuit breaker is open."""
            return self.current_state == STATE_OPEN
            
        @property
        def is_closed(self) -> bool:
            """Check if circuit breaker is closed."""
            return self.current_state == STATE_CLOSED
            
        @property
        def is_half_open(self) -> bool:
            """Check if circuit breaker is half open."""
            return self.current_state == STATE_HALF_OPEN
            
        def get_status(self) -> Dict[str, Any]:
            """Get circuit breaker status."""
            return {
                "name": self.name,
                "state": self.current_state,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure": self.last_failure,
                "expected_exception": self.expected_exception.__name__,
            }
            
except ImportError:
    # Fallback implementation if pybreaker is not installed
    import time
    from typing import Any, Callable, Dict, List, Optional, Type
    from functools import wraps
    from threading import Lock
    
    from .exceptions import CircuitBreakerError
    
    class CircuitBreaker:
        """Simple circuit breaker implementation."""
        
        def __init__(
            self,
            failure_threshold: int = 5,
            recovery_timeout: float = 60.0,
            expected_exception: type = Exception,
            name: Optional[str] = None,
            exclude: Optional[List[Type[Exception]]] = None,
        ) -> None:
            self.failure_threshold = failure_threshold
            self.recovery_timeout = recovery_timeout
            self.expected_exception = expected_exception
            self.name = name or "CircuitBreaker"
            self.exclude = exclude or []
            
            self._failure_count = 0
            self._last_failure_time = None
            self._state = "closed"
            self._lock = Lock()
            
        def __call__(self, func: Callable) -> Callable:
            """Decorator for circuit breaker protection."""
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self._lock:
                    if self._state == "open":
                        if self._should_attempt_reset():
                            self._state = "half-open"
                        else:
                            raise CircuitBreakerError(
                                f"Circuit breaker is open for {self.name}",
                                self.name,
                                self.recovery_timeout
                            )
                            
                try:
                    result = func(*args, **kwargs)
                    with self._lock:
                        if self._state == "half-open":
                            self._state = "closed"
                        self._failure_count = 0
                    return result
                    
                except Exception as e:
                    if any(isinstance(e, exc_type) for exc_type in self.exclude):
                        raise
                        
                    with self._lock:
                        self._failure_count += 1
                        self._last_failure_time = time.time()
                        
                        if self._failure_count >= self.failure_threshold:
                            self._state = "open"
                            
                    raise
                    
            return wrapper
            
        def _should_attempt_reset(self) -> bool:
            """Check if we should attempt to reset the circuit."""
            return (
                self._last_failure_time and
                time.time() - self._last_failure_time >= self.recovery_timeout
            )
            
        @property
        def is_open(self) -> bool:
            """Check if circuit breaker is open."""
            return self._state == "open"
            
        @property
        def is_closed(self) -> bool:
            """Check if circuit breaker is closed."""
            return self._state == "closed"
            
        @property
        def is_half_open(self) -> bool:
            """Check if circuit breaker is half open."""
            return self._state == "half-open"
            
        def get_status(self) -> Dict[str, Any]:
            """Get circuit breaker status."""
            return {
                "name": self.name,
                "state": self._state,
                "failure_count": self._failure_count,
                "last_failure": self._last_failure_time,
                "expected_exception": self.expected_exception.__name__,
            }