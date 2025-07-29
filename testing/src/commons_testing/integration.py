"""Integration testing utilities."""

from typing import Any, Dict, Optional, Union, List
import httpx
from unittest.mock import MagicMock
import asyncio
from contextlib import asynccontextmanager

class APITestClient:
    """HTTP client for API testing."""
    
    def __init__(self, base_url: str = "http://testserver", timeout: float = 30.0):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self._client = None
    
    async def __aenter__(self):
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
    
    async def get(self, url: str, **kwargs) -> httpx.Response:
        """Make GET request."""
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            return await client.get(url, **kwargs)
    
    async def post(self, url: str, **kwargs) -> httpx.Response:
        """Make POST request."""
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            return await client.post(url, **kwargs)
    
    async def put(self, url: str, **kwargs) -> httpx.Response:
        """Make PUT request."""
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            return await client.put(url, **kwargs)
    
    async def delete(self, url: str, **kwargs) -> httpx.Response:
        """Make DELETE request."""
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            return await client.delete(url, **kwargs)

class MockServer:
    """Mock HTTP server for testing."""
    
    def __init__(self, port: int = 8000):
        self.port = port
        self.base_url = f"http://localhost:{port}"
        self.routes: Dict[str, Dict[str, Any]] = {}
    
    def add_route(self, method: str, path: str, response_data: Any, status_code: int = 200):
        """Add a mock route."""
        key = f"{method.upper()}:{path}"
        self.routes[key] = {
            "response_data": response_data,
            "status_code": status_code
        }
    
    async def start(self):
        """Start the mock server."""
        print(f"Mock server would start on {self.base_url}")
        # In a real implementation, this would start an actual server
        pass
    
    async def stop(self):
        """Stop the mock server."""
        print("Mock server stopped")
        pass
    
    @asynccontextmanager
    async def run(self):
        """Context manager for running the server."""
        await self.start()
        try:
            yield self
        finally:
            await self.stop()

class DatabaseTestHelper:
    """Helper for database testing."""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.connection = None
    
    async def setup(self):
        """Setup test database."""
        # Mock implementation
        self.connection = MagicMock()
    
    async def teardown(self):
        """Teardown test database."""
        if self.connection:
            self.connection.close()
    
    async def create_tables(self, tables: List[str]):
        """Create test tables."""
        for table in tables:
            print(f"Would create table: {table}")
    
    async def drop_tables(self, tables: List[str]):
        """Drop test tables.""" 
        for table in tables:
            print(f"Would drop table: {table}")