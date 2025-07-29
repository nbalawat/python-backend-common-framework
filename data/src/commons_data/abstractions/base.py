"""Base database abstractions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Union
from enum import Enum
from contextlib import asynccontextmanager

from commons_core.logging import get_logger
from commons_core.types import BaseModel

logger = get_logger(__name__)


class DatabaseType(Enum):
    """Supported database types."""
    
    # Relational
    POSTGRES = "postgres"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    SQLSERVER = "sqlserver"
    ORACLE = "oracle"
    
    # NoSQL
    MONGODB = "mongodb"
    REDIS = "redis"
    DYNAMODB = "dynamodb"
    COSMOSDB = "cosmosdb"
    CASSANDRA = "cassandra"
    NEO4J = "neo4j"
    
    # Time-series
    INFLUXDB = "influxdb"
    TIMESCALEDB = "timescaledb"
    PROMETHEUS = "prometheus"
    
    # Vector
    PINECONE = "pinecone"
    QDRANT = "qdrant"
    WEAVIATE = "weaviate"
    
    # Search
    ELASTICSEARCH = "elasticsearch"
    OPENSEARCH = "opensearch"


@dataclass
class ConnectionConfig:
    """Database connection configuration."""
    
    host: str
    port: int
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    ssl: bool = False
    pool_size: int = 10
    max_overflow: int = 5
    pool_timeout: float = 30.0
    pool_recycle: int = 3600
    options: Optional[Dict[str, Any]] = None
    
    def to_url(self, driver: str) -> str:
        """Convert to connection URL."""
        auth = ""
        if self.username:
            auth = self.username
            if self.password:
                auth += f":{self.password}"
            auth += "@"
            
        url = f"{driver}://{auth}{self.host}:{self.port}"
        if self.database:
            url += f"/{self.database}"
            
        return url


class DatabaseInterface(ABC):
    """Abstract database interface."""
    
    def __init__(self, config: ConnectionConfig) -> None:
        self.config = config
        
    @abstractmethod
    async def connect(self) -> None:
        """Establish database connection."""
        pass
        
    @abstractmethod
    async def disconnect(self) -> None:
        """Close database connection."""
        pass
        
    @abstractmethod
    async def execute(
        self,
        query: str,
        params: Optional[Union[list, dict]] = None,
    ) -> Any:
        """Execute a query."""
        pass
        
    @abstractmethod
    async def fetch_one(
        self,
        query: str,
        params: Optional[Union[list, dict]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Fetch single row."""
        pass
        
    @abstractmethod
    async def fetch_all(
        self,
        query: str,
        params: Optional[Union[list, dict]] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch all rows."""
        pass
        
    @abstractmethod
    async def fetch_many(
        self,
        query: str,
        size: int,
        params: Optional[Union[list, dict]] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch limited rows."""
        pass
        
    @abstractmethod
    @asynccontextmanager
    async def transaction(self):
        """Transaction context manager."""
        pass
        
    @abstractmethod
    async def insert(
        self,
        table: str,
        data: Dict[str, Any],
        returning: Optional[str] = None,
    ) -> Any:
        """Insert single row."""
        pass
        
    @abstractmethod
    async def insert_many(
        self,
        table: str,
        data: List[Dict[str, Any]],
        batch_size: int = 1000,
    ) -> int:
        """Insert multiple rows."""
        pass
        
    @abstractmethod
    async def update(
        self,
        table: str,
        data: Dict[str, Any],
        where: Dict[str, Any],
    ) -> int:
        """Update rows."""
        pass
        
    @abstractmethod
    async def delete(
        self,
        table: str,
        where: Dict[str, Any],
    ) -> int:
        """Delete rows."""
        pass
        
    @abstractmethod
    async def select(
        self,
        table: str,
        columns: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Select rows with conditions."""
        pass


class AsyncDatabase:
    """Async database wrapper with connection management."""
    
    def __init__(
        self,
        connection_string: str,
        pool_size: int = 10,
        **kwargs: Any,
    ) -> None:
        self.connection_string = connection_string
        self.pool_size = pool_size
        self.kwargs = kwargs
        self._db: Optional[DatabaseInterface] = None
        
    async def __aenter__(self) -> DatabaseInterface:
        """Async context manager entry."""
        await self.connect()
        return self._db
        
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()
        
    async def connect(self) -> None:
        """Connect to database."""
        # This would create appropriate database instance
        # based on connection string
        pass
        
    async def disconnect(self) -> None:
        """Disconnect from database."""
        if self._db:
            await self._db.disconnect()
            
    async def execute(self, query: str, params: Optional[Any] = None) -> Any:
        """Execute query."""
        if not self._db:
            raise RuntimeError("Database not connected")
        return await self._db.execute(query, params)


class DatabaseClient:
    """High-level database client."""
    
    def __init__(self, url: str, **options):
        self.url = url
        self.options = options
    
    async def connect(self) -> None:
        """Connect to database."""
        pass
    
    async def disconnect(self) -> None:
        """Disconnect from database."""
        pass
    
    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute query."""
        return {"result": "executed"}
    
    async def fetch_one(self, query: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Fetch one row."""
        return {"id": 1, "data": "test"}
    
    async def fetch_many(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Fetch multiple rows."""
        return [{"id": 1, "data": "test1"}, {"id": 2, "data": "test2"}]


class Model(BaseModel):
    """Base model class."""
    
    __tablename__: str = "base_table"
    
    def save(self) -> None:
        """Save model to database."""
        pass
    
    def delete(self) -> None:
        """Delete model from database."""
        pass
    
    @classmethod
    def find_by_id(cls, id: Any) -> Optional["Model"]:
        """Find model by ID."""
        return None


class Repository(ABC):
    """Base repository interface."""
    
    def __init__(self, db_client: DatabaseClient):
        self.db_client = db_client
    
    @abstractmethod
    async def create(self, data: Dict[str, Any]) -> Model:
        """Create new record."""
        pass
    
    @abstractmethod
    async def find_by_id(self, id: Any) -> Optional[Model]:
        """Find record by ID."""
        pass
    
    @abstractmethod
    async def update(self, id: Any, data: Dict[str, Any]) -> Optional[Model]:
        """Update record."""
        pass
    
    @abstractmethod
    async def delete(self, id: Any) -> bool:
        """Delete record."""
        pass
    
    @abstractmethod
    async def find_all(self) -> List[Model]:
        """Find all records."""
        pass