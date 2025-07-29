"""Common decorators for various use cases."""

import asyncio
import functools
import time
import warnings
from typing import Any, Callable, Dict, Optional, TypeVar, Union
from threading import Lock
from datetime import datetime, timedelta

T = TypeVar("T")


def cached(
    ttl: Optional[float] = None,
    maxsize: int = 128,
    key_func: Optional[Callable[..., str]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Cache function results with optional TTL."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache: Dict[str, tuple[T, float]] = {}
        cache_lock = Lock()
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = str((args, tuple(sorted(kwargs.items()))))
                
            # Check cache
            with cache_lock:
                if key in cache:
                    value, timestamp = cache[key]
                    if ttl is None or time.time() - timestamp < ttl:
                        return value
                        
                # Clean old entries if cache is full
                if len(cache) >= maxsize:
                    if ttl:
                        # Remove expired entries
                        now = time.time()
                        expired = [k for k, (_, t) in cache.items() if now - t >= ttl]
                        for k in expired:
                            del cache[k]
                    else:
                        # Remove oldest entry
                        oldest_key = min(cache.keys(), key=lambda k: cache[k][1])
                        del cache[oldest_key]
                        
            # Call function
            result = func(*args, **kwargs)
            
            # Store in cache
            with cache_lock:
                cache[key] = (result, time.time())
                
            return result
            
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_info = lambda: {"size": len(cache), "maxsize": maxsize, "ttl": ttl}
        
        return wrapper
    return decorator


def measure_time(
    log_func: Optional[Callable[[str, float], None]] = None,
    prefix: str = "",
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Measure function execution time."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start
                message = f"{prefix}{func.__name__} took {duration:.3f}s"
                
                if log_func:
                    log_func(message, duration)
                else:
                    print(message)
                    
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start
                message = f"{prefix}{func.__name__} took {duration:.3f}s"
                
                if log_func:
                    log_func(message, duration)
                else:
                    print(message)
                    
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


def deprecated(
    reason: str,
    version: Optional[str] = None,
    alternative: Optional[str] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Mark function as deprecated."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            message = f"{func.__name__} is deprecated"
            if version:
                message += f" since version {version}"
            message += f": {reason}"
            if alternative:
                message += f". Use {alternative} instead."
                
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
            
        # Add deprecation info to docstring
        doc = func.__doc__ or ""
        deprecation_note = f"\n\n.. deprecated:: {version or 'unknown'}\n   {reason}"
        if alternative:
            deprecation_note += f"\n   Use :func:`{alternative}` instead."
        wrapper.__doc__ = doc + deprecation_note
        
        return wrapper
    return decorator


def singleton(cls: type) -> type:
    """Make a class a singleton."""
    instances = {}
    lock = Lock()
    
    @functools.wraps(cls)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        return instances[cls]
        
    return wrapper


def synchronized(lock: Optional[Lock] = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Synchronize function calls with a lock."""
    if lock is None:
        lock = Lock()
        
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            with lock:
                return func(*args, **kwargs)
        return wrapper
    return decorator


def throttle(
    max_calls: int,
    period: float,
    raise_on_limit: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Throttle function calls."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        calls = []
        lock = Lock()
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Optional[T]:
            now = time.time()
            
            with lock:
                # Remove old calls
                calls[:] = [call_time for call_time in calls if now - call_time < period]
                
                if len(calls) >= max_calls:
                    if raise_on_limit:
                        raise RuntimeError(
                            f"Rate limit exceeded: {max_calls} calls per {period}s"
                        )
                    return None
                    
                calls.append(now)
                
            return func(*args, **kwargs)
            
        wrapper.reset = lambda: calls.clear()
        
        return wrapper
    return decorator