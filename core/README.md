# Commons Core

Foundation utilities for the Python Commons library collection.

## Features

- **Configuration Management**: Multi-source configuration with environment variables, files, and command line
- **Structured Logging**: JSON logging with context propagation and performance metrics
- **Error Handling**: Retry decorators, circuit breakers, and custom exceptions
- **Type System**: Pydantic models with validators and serializers
- **Utilities**: Async helpers, date/time utilities, and common decorators

## Installation

```bash
pip install commons-core
```

## Usage

### Configuration Management

```python
from commons_core.config import ConfigManager

# Initialize config manager
config = ConfigManager()

# Load from multiple sources
config.load_from_env()
config.load_from_file("config.yaml")
config.load_from_dict({"api": {"timeout": 30}})

# Access configuration
api_key = config.get("api.key", secret=True)
timeout = config.get("api.timeout", default=10)

# Type-safe configuration with Pydantic
from commons_core.config import BaseConfig

class AppConfig(BaseConfig):
    api_key: str
    timeout: int = 30
    
app_config = config.load_model(AppConfig, prefix="app")
```

### Structured Logging

```python
from commons_core.logging import get_logger, configure_logging

# Configure logging globally
configure_logging(
    level="INFO",
    format="json",
    handlers=["console", "file"],
    context={"service": "my-app", "version": "1.0.0"}
)

# Get logger instance
logger = get_logger(__name__)

# Log with context
logger.info("User logged in", user_id=123, ip="192.168.1.1")
logger.error("Failed to process", error=str(e), retry_count=3)

# Performance logging
with logger.timer("database_query"):
    results = db.query("SELECT * FROM users")
```

### Error Handling

```python
from commons_core.errors import retry, CircuitBreaker, RetryableError

# Retry decorator with exponential backoff
@retry(
    max_attempts=3,
    backoff_factor=2,
    exceptions=(RetryableError, ConnectionError)
)
async def fetch_data(url: str):
    response = await http_client.get(url)
    if response.status >= 500:
        raise RetryableError("Server error")
    return response.json()

# Circuit breaker pattern
breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60,
    expected_exception=ConnectionError
)

@breaker
async def call_external_service():
    return await external_api.call()
```

### Type System

```python
from commons_core.types import BaseModel, validator, SecretStr
from datetime import datetime

class User(BaseModel):
    id: int
    email: str
    password: SecretStr
    created_at: datetime
    
    @validator("email")
    def validate_email(cls, v):
        if "@" not in v:
            raise ValueError("Invalid email")
        return v.lower()

# Serialization
user = User(
    id=1,
    email="USER@EXAMPLE.COM",
    password="secret123",
    created_at=datetime.now()
)

# JSON serialization with secret masking
json_data = user.model_dump_json()

# MessagePack serialization
msgpack_data = user.to_msgpack()
```

### Utilities

```python
from commons_core.utils import (
    run_async,
    RateLimiter,
    cached,
    measure_time,
    retry_async
)

# Async utilities
@retry_async(max_attempts=3)
async def async_operation():
    return await some_async_call()

# Rate limiting
rate_limiter = RateLimiter(max_calls=100, period=60)

@rate_limiter
async def api_call():
    return await external_api.call()

# Caching
@cached(ttl=300)
def expensive_computation(x: int) -> int:
    return x ** 2

# Performance measurement
@measure_time
def slow_function():
    time.sleep(1)
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=commons_core
```