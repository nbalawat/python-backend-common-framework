"""Database factory for creating database connections."""

from typing import Any, Dict, Optional, Type
from .abstractions import DatabaseClient

class DatabaseFactory:
    """Factory for creating database connections."""
    
    _drivers: Dict[str, Type[DatabaseClient]] = {}
    
    @classmethod
    def register(cls, driver_name: str, driver_class: Type[DatabaseClient]) -> None:
        """Register a database driver."""
        cls._drivers[driver_name.lower()] = driver_class
    
    @classmethod
    def create(
        cls,
        url: str,
        **options: Any,
    ) -> DatabaseClient:
        """Create database client instance."""
        # Parse URL to determine driver
        if url.startswith("sqlite"):
            driver = "sqlite"
        elif url.startswith("postgresql") or url.startswith("postgres"):
            driver = "postgresql"
        elif url.startswith("mysql"):
            driver = "mysql"
        elif url.startswith("mongodb"):
            driver = "mongodb"
        else:
            driver = "mock"
        
        if driver in cls._drivers:
            driver_class = cls._drivers[driver]
            return driver_class(url, **options)
        else:
            # Return mock client for testing
            return MockDatabaseClient(url, **options)
    
    @classmethod
    def available_drivers(cls) -> list[str]:
        """Get list of available drivers."""
        return list(cls._drivers.keys())

class MockDatabaseClient(DatabaseClient):
    """Mock database client for testing."""
    
    def __init__(self, url: str, **options):
        super().__init__(url)
        self.options = options
        self.connected = False
    
    async def connect(self) -> None:
        """Connect to database."""
        self.connected = True
        print(f"Mock connection to {self.url}")
    
    async def disconnect(self) -> None:
        """Disconnect from database."""
        self.connected = False
        print(f"Mock disconnection from {self.url}")
    
    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute query."""
        print(f"Mock execute: {query}")
        return {"rows_affected": 1}
    
    async def fetch_one(self, query: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Fetch one row."""
        print(f"Mock fetch_one: {query}")
        return {"id": 1, "name": "test"}
    
    async def fetch_many(self, query: str, params: Optional[Dict[str, Any]] = None) -> list[Dict[str, Any]]:
        """Fetch multiple rows."""
        print(f"Mock fetch_many: {query}")
        return [{"id": 1, "name": "test1"}, {"id": 2, "name": "test2"}]