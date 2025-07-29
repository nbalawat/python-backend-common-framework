# Commons Core

Foundation utilities for Python Commons library providing configuration management, structured logging, error handling, and essential data types.

## Installation

```bash
pip install commons-core
```

## Features

- **Configuration Management**: Multi-source configuration with validation
- **Structured Logging**: JSON logging with context propagation
- **Error Handling**: Custom exceptions with retry logic and circuit breakers
- **Type System**: Enhanced Pydantic models with validation
- **Utilities**: Async helpers, decorators, and datetime utilities

## Quick Start

```python
from commons_core import ConfigManager, get_logger, BaseModel

# Configuration
config = ConfigManager()
api_key = config.get("API_KEY", "default-key")

# Logging
logger = get_logger(__name__)
logger.info("Application started", extra={"version": "1.0.0"})

# Data Models
class User(BaseModel):
    name: str
    email: str
    age: int = 0

user = User(name="John", email="john@example.com")
```

## Detailed Usage Examples

### Configuration Management

#### Basic Configuration Usage
```python
from commons_core.config import ConfigManager

# Initialize configuration manager
config = ConfigManager()

# Get configuration values with defaults and type conversion
database_url = config.get("DATABASE_URL", "sqlite:///default.db")
port = config.get("PORT", 8000, int)
debug = config.get("DEBUG", False, bool)
allowed_hosts = config.get("ALLOWED_HOSTS", [], list)

print(f"Database: {database_url}")
print(f"Port: {port}")
print(f"Debug mode: {debug}")
```

#### Multi-Source Configuration
```python
from commons_core.config import ConfigManager, EnvProvider, FileProvider, DictProvider

# Create configuration manager with multiple providers
config = ConfigManager()

# Add providers in priority order (first wins)
config.add_provider(EnvProvider())  # Environment variables first
config.add_provider(FileProvider("config.yaml"))  # YAML file second
config.add_provider(FileProvider("config.json"))  # JSON file third
config.add_provider(DictProvider({  # Default values last
    "database_url": "sqlite:///app.db",
    "log_level": "INFO",
    "cache_ttl": 3600
}))

# Access configuration
db_config = {
    "url": config.get("DATABASE_URL"),
    "pool_size": config.get("DB_POOL_SIZE", 10, int),
    "timeout": config.get("DB_TIMEOUT", 30, int)
}
```

#### Configuration Validation
```python
from commons_core.config import ConfigManager, BaseConfig
from pydantic import Field

class DatabaseConfig(BaseConfig):
    url: str = Field(..., description="Database connection URL")
    pool_size: int = Field(10, ge=1, le=100)
    timeout: int = Field(30, ge=1)
    ssl_required: bool = Field(False)

class AppConfig(BaseConfig):
    app_name: str = Field("MyApp")
    debug: bool = Field(False)
    database: DatabaseConfig

# Load and validate configuration
config = ConfigManager()
app_config = config.load_config(AppConfig)

print(f"App: {app_config.app_name}")
print(f"DB Pool Size: {app_config.database.pool_size}")
```

### Structured Logging

#### Basic Logging
```python
from commons_core.logging import get_logger, configure_logging

# Configure global logging
configure_logging(
    level="INFO",
    format="json",  # or "text"
    output="stdout"  # or file path
)

# Get logger for your module
logger = get_logger(__name__)

# Basic logging
logger.debug("Debug information")
logger.info("Application started")
logger.warning("This is a warning")
logger.error("An error occurred")
logger.critical("Critical system error")
```

#### Structured Logging with Context
```python
from commons_core.logging import get_logger
import uuid

logger = get_logger(__name__)

# Log with structured data
request_id = str(uuid.uuid4())
user_id = "user123"

logger.info("Processing request", extra={
    "request_id": request_id,
    "user_id": user_id,
    "endpoint": "/api/users",
    "method": "POST"
})

# Log with error context
try:
    # Some operation
    result = process_user_data(user_id)
    logger.info("User processed successfully", extra={
        "request_id": request_id,
        "user_id": user_id,
        "processing_time_ms": 150
    })
except Exception as e:
    logger.error("Failed to process user", extra={
        "request_id": request_id,
        "user_id": user_id,
        "error": str(e),
        "error_type": type(e).__name__
    }, exc_info=True)
```

#### Custom Log Formatters
```python
from commons_core.logging import get_logger, configure_logging
from commons_core.logging.formatters import JSONFormatter, TextFormatter

# Custom JSON formatter with additional fields
class CustomJSONFormatter(JSONFormatter):
    def format(self, record):
        # Add custom fields to every log record
        record.service_name = "my-service"
        record.version = "1.0.0"
        return super().format(record)

# Configure with custom formatter
configure_logging(
    level="INFO",
    formatter=CustomJSONFormatter()
)

logger = get_logger(__name__)
logger.info("Custom formatted log")
```

### Error Handling and Resilience

#### Retry Decorator
```python
from commons_core.errors import retry, RetryableError
import httpx
import asyncio

@retry(
    max_attempts=3,
    backoff_factor=2,  # 1s, 2s, 4s delays
    exceptions=(httpx.RequestError, httpx.TimeoutException)
)
async def fetch_user_data(user_id: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.example.com/users/{user_id}")
        if response.status_code >= 500:
            raise RetryableError("Server error, will retry")
        response.raise_for_status()
        return response.json()

# Usage
async def main():
    try:
        user_data = await fetch_user_data("123")
        print(f"User data: {user_data}")
    except Exception as e:
        print(f"Failed after retries: {e}")

asyncio.run(main())
```

#### Circuit Breaker Pattern
```python
from commons_core.errors import CircuitBreaker
import asyncio
import random

# Create circuit breaker
breaker = CircuitBreaker(
    failure_threshold=5,     # Trip after 5 failures
    recovery_timeout=30.0,   # Wait 30s before trying again
    expected_exception=Exception
)

@breaker
async def unreliable_service():
    # Simulate service that fails randomly
    if random.random() < 0.7:  # 70% failure rate
        raise Exception("Service unavailable")
    return {"status": "success", "data": "response"}

async def main():
    for i in range(20):
        try:
            result = await unreliable_service()
            print(f"Attempt {i+1}: {result}")
        except Exception as e:
            print(f"Attempt {i+1}: Failed - {e}")
        
        await asyncio.sleep(1)

asyncio.run(main())
```

#### Custom Exception Handling
```python
from commons_core.errors import CommonsError, ValidationError, ErrorHandler
from commons_core.logging import get_logger

# Custom exceptions
class UserNotFoundError(CommonsError):
    """Raised when user is not found."""
    pass

class InvalidUserDataError(ValidationError):
    """Raised when user data is invalid."""
    pass

# Global error handler
logger = get_logger(__name__)
error_handler = ErrorHandler(logger=logger)

@error_handler.handle_errors
async def process_user(user_id: str, user_data: dict):
    if not user_id:
        raise InvalidUserDataError("User ID is required")
    
    # Simulate user lookup
    if user_id == "nonexistent":
        raise UserNotFoundError(f"User {user_id} not found")
    
    return {"user_id": user_id, "processed": True}

# Usage with automatic error handling
async def main():
    test_cases = [
        ("valid_user", {"name": "John"}),
        ("", {"name": "Jane"}),  # Will raise InvalidUserDataError
        ("nonexistent", {"name": "Bob"})  # Will raise UserNotFoundError
    ]
    
    for user_id, data in test_cases:
        try:
            result = await process_user(user_id, data)
            print(f"Success: {result}")
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")

asyncio.run(main())
```

### Data Models and Validation

#### Basic Models
```python
from commons_core.types import BaseModel, validator
from typing import Optional, List
from datetime import datetime
from enum import Enum

class Status(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str
    country: str = "US"

class User(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str] = None
    status: Status = Status.ACTIVE
    addresses: List[Address] = []
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower()
    
    @validator('username')
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError('Username too short')
        return v

# Usage
user = User(
    id=1,
    username="johndoe",
    email="John.Doe@Example.com",
    full_name="John Doe",
    addresses=[
        Address(
            street="123 Main St",
            city="Springfield",
            state="IL",
            zip_code="62701"
        )
    ],
    created_at=datetime.now()
)

print(f"User: {user.username}")
print(f"Email: {user.email}")  # Normalized to lowercase
print(f"JSON: {user.model_dump_json(indent=2)}")
```

#### Advanced Model Features
```python
from commons_core.types import BaseModel, SecretStr
from typing import ClassVar
import hashlib

class UserAccount(BaseModel):
    # Class configuration
    model_config = {"frozen": False, "validate_assignment": True}
    
    # Class variables
    TABLE_NAME: ClassVar[str] = "user_accounts"
    
    # Fields
    username: str
    password: SecretStr  # Automatically masked in logs/serialization
    email: str
    is_active: bool = True
    login_attempts: int = 0
    
    def verify_password(self, plain_password: str) -> bool:
        """Verify password against stored hash."""
        stored_hash = self.password.get_secret_value()
        provided_hash = hashlib.sha256(plain_password.encode()).hexdigest()
        return stored_hash == provided_hash
    
    def increment_login_attempts(self):
        """Increment failed login attempts."""
        self.login_attempts += 1
        if self.login_attempts >= 5:
            self.is_active = False
    
    def reset_login_attempts(self):
        """Reset login attempts on successful login."""
        self.login_attempts = 0

# Usage
account = UserAccount(
    username="johndoe",
    password=hashlib.sha256("secret123".encode()).hexdigest(),
    email="john@example.com"
)

# Password is masked in serialization
print(account.model_dump())  # password shows as '**********'

# But can be accessed when needed
if account.verify_password("secret123"):
    print("Password verified!")
    account.reset_login_attempts()
```

### Async Utilities

#### Async Helpers
```python
from commons_core.utils import async_retry, timeout, gather_with_limit
import asyncio
import httpx

# Async retry with backoff
@async_retry(max_attempts=3, backoff_seconds=1)
async def fetch_data(url: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()

# Timeout decorator
@timeout(seconds=10)
async def long_running_task():
    await asyncio.sleep(15)  # Will timeout after 10 seconds
    return "completed"

# Limited concurrency
async def fetch_multiple_urls():
    urls = [f"https://api.example.com/data/{i}" for i in range(100)]
    
    # Fetch with maximum 10 concurrent requests
    results = await gather_with_limit(
        limit=10,
        *[fetch_data(url) for url in urls]
    )
    
    return results

# Usage
async def main():
    try:
        # Single fetch with retry
        data = await fetch_data("https://api.example.com/data/1")
        print(f"Data: {data}")
        
        # Multiple fetches with concurrency limit
        all_data = await fetch_multiple_urls()
        print(f"Fetched {len(all_data)} items")
        
    except asyncio.TimeoutError:
        print("Operation timed out")
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(main())
```

### Complete Application Example

```python
from commons_core import ConfigManager, get_logger, BaseModel
from commons_core.errors import retry, CircuitBreaker, ErrorHandler
from commons_core.utils.decorators import measure_time
import asyncio
import httpx
from typing import List, Optional

# Configuration
class AppConfig(BaseModel):
    api_base_url: str
    api_timeout: int = 30
    max_retries: int = 3
    circuit_breaker_threshold: int = 5

# Data models
class User(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str] = None

class APIResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None

# Application class
class UserService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.error_handler = ErrorHandler(self.logger)
        
        # Circuit breaker for external API
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker_threshold,
            recovery_timeout=60
        )
    
    @measure_time()
    @retry(max_attempts=3, backoff_factor=2)
    async def fetch_user(self, user_id: int) -> Optional[User]:
        """Fetch user from external API with retry and circuit breaker."""
        
        @self.circuit_breaker
        async def _api_call():
            async with httpx.AsyncClient(timeout=self.config.api_timeout) as client:
                response = await client.get(f"{self.config.api_base_url}/users/{user_id}")
                response.raise_for_status()
                return response.json()
        
        try:
            data = await _api_call()
            user = User(**data)
            
            self.logger.info(
                "User fetched successfully",
                extra={"user_id": user_id, "username": user.username}
            )
            
            return user
            
        except Exception as e:
            self.logger.error(
                "Failed to fetch user",
                extra={"user_id": user_id, "error": str(e)},
                exc_info=True
            )
            return None
    
    async def fetch_multiple_users(self, user_ids: List[int]) -> List[User]:
        """Fetch multiple users concurrently."""
        tasks = [self.fetch_user(user_id) for user_id in user_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        users = []
        for result in results:
            if isinstance(result, User):
                users.append(result)
            elif isinstance(result, Exception):
                self.logger.warning(f"Failed to fetch user: {result}")
        
        return users

# Application main
async def main():
    # Load configuration
    config_manager = ConfigManager()
    config_manager.add_provider(DictProvider({
        "api_base_url": "https://jsonplaceholder.typicode.com",
        "api_timeout": 10,
        "max_retries": 3
    }))
    
    app_config = config_manager.load_config(AppConfig)
    
    # Initialize service
    user_service = UserService(app_config)
    logger = get_logger("main")
    
    logger.info("Application started", extra={"config": app_config.model_dump()})
    
    try:
        # Fetch single user
        user = await user_service.fetch_user(1)
        if user:
            logger.info(f"Fetched user: {user.username}")
        
        # Fetch multiple users
        users = await user_service.fetch_multiple_users([1, 2, 3, 4, 5])
        logger.info(f"Fetched {len(users)} users")
        
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
    
    logger.info("Application completed")

if __name__ == "__main__":
    asyncio.run(main())
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