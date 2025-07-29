"""Async testing fixtures."""

import asyncio
from typing import AsyncGenerator, Callable, TypeVar, Any
from functools import wraps
import pytest
import httpx
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from redis.asyncio import Redis

T = TypeVar("T")


@pytest.fixture
async def async_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Async HTTP client fixture."""
    async with httpx.AsyncClient() as client:
        yield client


@pytest.fixture
async def async_db(database_url: str = "sqlite+aiosqlite:///:memory:") -> AsyncGenerator[AsyncSession, None]:
    """Async database session fixture."""
    engine = create_async_engine(database_url)
    
    async with engine.begin() as conn:
        # Create tables if needed
        pass
        
    async with AsyncSession(engine) as session:
        yield session
        
    await engine.dispose()


@pytest.fixture
async def async_redis(redis_url: str = "redis://localhost:6379") -> AsyncGenerator[Redis, None]:
    """Async Redis client fixture."""
    client = Redis.from_url(redis_url)
    
    try:
        yield client
    finally:
        await client.close()


def create_async_fixture(setup_func: Callable[..., T]) -> Callable[..., AsyncGenerator[T, None]]:
    """Create custom async fixture."""
    @wraps(setup_func)
    async def fixture_wrapper(*args: Any, **kwargs: Any) -> AsyncGenerator[T, None]:
        # Setup
        if asyncio.iscoroutinefunction(setup_func):
            resource = await setup_func(*args, **kwargs)
        else:
            resource = setup_func(*args, **kwargs)
            
        try:
            yield resource
        finally:
            # Teardown
            if hasattr(resource, "close"):
                if asyncio.iscoroutinefunction(resource.close):
                    await resource.close()
                else:
                    resource.close()
                    
    return pytest.fixture(fixture_wrapper)