"""Database testing fixtures."""

import pytest
from typing import Generator, Any, Dict
from unittest.mock import MagicMock

@pytest.fixture
def postgres_db() -> Generator[Dict[str, Any], None, None]:
    """Mock PostgreSQL database fixture."""
    mock_db = {
        "connection": MagicMock(),
        "url": "postgresql://test:test@localhost/test_db",
        "name": "test_db"
    }
    yield mock_db

@pytest.fixture
def mysql_db() -> Generator[Dict[str, Any], None, None]:
    """Mock MySQL database fixture."""
    mock_db = {
        "connection": MagicMock(),
        "url": "mysql://test:test@localhost/test_db",
        "name": "test_db"
    }
    yield mock_db

@pytest.fixture
def mongodb() -> Generator[Dict[str, Any], None, None]:
    """Mock MongoDB fixture."""
    mock_db = {
        "client": MagicMock(),
        "database": MagicMock(),
        "url": "mongodb://localhost:27017/test_db",
        "name": "test_db"
    }
    yield mock_db

@pytest.fixture
def redis_cache() -> Generator[Dict[str, Any], None, None]:
    """Mock Redis cache fixture."""
    mock_redis = {
        "client": MagicMock(),
        "url": "redis://localhost:6379/0",
        "db": 0
    }
    yield mock_redis