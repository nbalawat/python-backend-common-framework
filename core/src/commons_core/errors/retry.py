"""Retry logic with exponential backoff."""

import asyncio
import random
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Type, TypeVar, Union
from dataclasses import dataclass

from tenacity import (
    retry as tenacity_retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log,
)
import logging

from .exceptions import RetryableError

T = TypeVar("T")
logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    exceptions: Tuple[Type[Exception], ...] = (RetryableError, ConnectionError)


def retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (RetryableError, ConnectionError),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying functions with exponential backoff."""
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            @tenacity_retry(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(
                    multiplier=initial_delay,
                    max=max_delay,
                    exp_base=backoff_factor,
                ),
                retry=retry_if_exception_type(exceptions),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                after=after_log(logger, logging.INFO),
            )
            def _retry_wrapper() -> T:
                return func(*args, **kwargs)
                
            return _retry_wrapper()
            
        return wrapper
    return decorator


def retry_async(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (RetryableError, ConnectionError),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying async functions with exponential backoff."""
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            attempt = 0
            delay = initial_delay
            
            while attempt < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error(f"Max retries ({max_attempts}) exceeded for {func.__name__}")
                        raise
                        
                    # Calculate next delay
                    if jitter:
                        actual_delay = delay * (0.5 + random.random())
                    else:
                        actual_delay = delay
                        
                    logger.warning(
                        f"Retry {attempt}/{max_attempts} for {func.__name__} "
                        f"after {actual_delay:.2f}s due to: {e}"
                    )
                    
                    await asyncio.sleep(actual_delay)
                    
                    # Exponential backoff
                    delay = min(delay * backoff_factor, max_delay)
                    
            return None  # Should never reach here
            
        return wrapper
    return decorator