"""Asynchronous utilities and helpers."""

import asyncio
from typing import Any, Callable, Coroutine, List, Optional, TypeVar, Union
from functools import wraps
import time
from collections import deque
from contextlib import asynccontextmanager

T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run async function in sync context."""
    loop = None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        pass
        
    if loop and loop.is_running():
        # If we're already in an async context, create a new thread
        import concurrent.futures
        import threading
        
        result = None
        exception = None
        
        def run_in_thread():
            nonlocal result, exception
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                result = new_loop.run_until_complete(coro)
            except Exception as e:
                exception = e
            finally:
                new_loop.close()
                
        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()
        
        if exception:
            raise exception
        return result
    else:
        # No running loop, we can create one
        return asyncio.run(coro)


async def gather_with_timeout(
    *coros: Coroutine[Any, Any, Any],
    timeout: float,
    return_exceptions: bool = False,
) -> List[Any]:
    """Gather coroutines with a timeout."""
    try:
        return await asyncio.wait_for(
            asyncio.gather(*coros, return_exceptions=return_exceptions),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        # Cancel all pending tasks
        for coro in coros:
            if asyncio.iscoroutine(coro):
                coro.close()
        raise


class RateLimiter:
    """Async rate limiter using token bucket algorithm."""
    
    def __init__(self, max_calls: int, period: float) -> None:
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
        self._lock = asyncio.Lock()
        
    async def acquire(self) -> None:
        """Acquire permission to make a call."""
        async with self._lock:
            now = time.time()
            
            # Remove old calls outside the period
            while self.calls and self.calls[0] <= now - self.period:
                self.calls.popleft()
                
            if len(self.calls) >= self.max_calls:
                # Calculate wait time
                sleep_time = self.period - (now - self.calls[0])
                await asyncio.sleep(sleep_time)
                return await self.acquire()
                
            self.calls.append(now)
            
    def __call__(self, func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        """Decorator to rate limit async functions."""
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            await self.acquire()
            return await func(*args, **kwargs)
        return wrapper
        
    @asynccontextmanager
    async def __aenter__(self):
        """Context manager support."""
        await self.acquire()
        yield self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass


class Semaphore:
    """Enhanced async semaphore with timeout support."""
    
    def __init__(self, value: int = 1) -> None:
        self._semaphore = asyncio.Semaphore(value)
        self._value = value
        
    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire semaphore with optional timeout."""
        if timeout is None:
            await self._semaphore.acquire()
            return True
            
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout)
            return True
        except asyncio.TimeoutError:
            return False
            
    def release(self) -> None:
        """Release semaphore."""
        self._semaphore.release()
        
    @property
    def available(self) -> int:
        """Get number of available slots."""
        return self._semaphore._value
        
    @asynccontextmanager
    async def __aenter__(self):
        """Context manager support."""
        await self.acquire()
        yield self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


async def batch_process(
    items: List[T],
    processor: Callable[[List[T]], Coroutine[Any, Any, List[Any]]],
    batch_size: int = 100,
    max_concurrent: int = 5,
) -> List[Any]:
    """Process items in batches with concurrency control."""
    results = []
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_batch(batch: List[T]) -> List[Any]:
        async with semaphore:
            return await processor(batch)
            
    # Create batches
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    
    # Process batches concurrently
    batch_results = await asyncio.gather(
        *[process_batch(batch) for batch in batches],
        return_exceptions=True
    )
    
    # Flatten results
    for batch_result in batch_results:
        if isinstance(batch_result, Exception):
            raise batch_result
        results.extend(batch_result)
        
    return results