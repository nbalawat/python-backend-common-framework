# Commons Testing

Advanced testing utilities for Python applications.

## Features

- **Async Testing**: Enhanced fixtures and utilities for async testing
- **Database Testing**: Test containers and fixtures for various databases
- **Integration Testing**: Utilities for API, message queue, and service testing
- **Test Data**: Factories and generators for test data creation
- **Mocking**: Enhanced mocking utilities for external services

## Installation

```bash
# Basic installation
pip install commons-testing

# With database testing support
pip install commons-testing[databases]

# With cloud service mocking
pip install commons-testing[cloud]
```

## Usage

### Async Testing

```python
import pytest
from commons_testing.async_fixtures import async_client, async_db

@pytest.mark.asyncio
async def test_async_operation(async_client):
    """Test with async HTTP client."""
    response = await async_client.get("/api/users")
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_with_async_db(async_db):
    """Test with async database."""
    await async_db.execute("INSERT INTO users (name) VALUES ('test')")
    result = await async_db.fetch_one("SELECT * FROM users WHERE name = 'test'")
    assert result is not None

# Custom async fixtures
from commons_testing.fixtures import create_async_fixture

@create_async_fixture
async def my_service():
    """Create async service fixture."""
    service = MyService()
    await service.start()
    yield service
    await service.stop()
```

### Database Testing

```python
from commons_testing.databases import PostgresContainer, MySQLContainer

# PostgreSQL testing
@pytest.fixture
def postgres():
    with PostgresContainer() as postgres:
        yield postgres.get_connection_url()

def test_with_postgres(postgres):
    # Use postgres connection URL
    engine = create_engine(postgres)
    # Run tests...

# MySQL testing
@pytest.fixture
def mysql():
    with MySQLContainer() as mysql:
        yield mysql.get_connection_url()

# MongoDB testing
from commons_testing.databases import MongoContainer

@pytest.fixture
def mongodb():
    with MongoContainer() as mongo:
        yield mongo.get_connection_url()

# Redis testing
from commons_testing.databases import RedisContainer

@pytest.fixture
def redis():
    with RedisContainer() as redis:
        yield redis.get_client()
```

### Integration Testing

```python
from commons_testing.integration import APITestClient, MockServer

# API testing
@pytest.fixture
def api_client():
    return APITestClient(base_url="http://localhost:8000")

async def test_api_endpoint(api_client):
    response = await api_client.post(
        "/users",
        json={"name": "test", "email": "test@example.com"},
        headers={"Authorization": "Bearer token"},
    )
    assert response.status_code == 201
    assert response.json()["name"] == "test"

# Mock external services
@pytest.fixture
def mock_server():
    with MockServer() as server:
        server.expect(
            method="GET",
            path="/external/api",
            response={"status": "ok"},
            status_code=200,
        )
        yield server

# Message queue testing
from commons_testing.integration import KafkaContainer, RabbitMQContainer

@pytest.fixture
def kafka():
    with KafkaContainer() as kafka:
        yield kafka.get_broker()

@pytest.fixture
def rabbitmq():
    with RabbitMQContainer() as rabbitmq:
        yield rabbitmq.get_connection()
```

### Test Data Generation

```python
from commons_testing.factories import Factory, SubFactory, LazyAttribute
from commons_testing.generators import fake

# Define factories
class UserFactory(Factory):
    class Meta:
        model = User
    
    id = fake.uuid4
    name = fake.name
    email = fake.email
    created_at = fake.date_time
    
class PostFactory(Factory):
    class Meta:
        model = Post
    
    title = fake.sentence
    content = fake.text
    author = SubFactory(UserFactory)
    published = LazyAttribute(lambda obj: fake.boolean())

# Use factories
def test_user_creation():
    user = UserFactory()
    assert user.name
    assert "@" in user.email
    
    # Create with overrides
    user = UserFactory(name="John Doe", email="john@example.com")
    assert user.name == "John Doe"
    
    # Batch creation
    users = UserFactory.create_batch(10)
    assert len(users) == 10

# Data generators
from commons_testing.generators import DataGenerator

generator = DataGenerator()

# Generate various data types
test_data = {
    "users": generator.users(count=5),
    "products": generator.products(count=10),
    "orders": generator.orders(count=20),
    "events": generator.events(count=100, start_date="2024-01-01"),
}
```

### Cloud Service Mocking

```python
from commons_testing.cloud import mock_aws, mock_gcp, mock_azure

# Mock AWS services
@mock_aws
def test_s3_operations():
    import boto3
    
    s3 = boto3.client("s3")
    s3.create_bucket(Bucket="test-bucket")
    s3.put_object(Bucket="test-bucket", Key="test.txt", Body=b"content")
    
    response = s3.get_object(Bucket="test-bucket", Key="test.txt")
    assert response["Body"].read() == b"content"

# Mock GCP services
@mock_gcp
def test_gcs_operations():
    from google.cloud import storage
    
    client = storage.Client()
    bucket = client.create_bucket("test-bucket")
    blob = bucket.blob("test.txt")
    blob.upload_from_string("content")
    
    assert blob.download_as_text() == "content"

# LocalStack integration
from commons_testing.cloud import LocalStackContainer

@pytest.fixture
def localstack():
    with LocalStackContainer(services=["s3", "dynamodb", "sqs"]) as ls:
        yield ls.get_service_url("s3")
```

### Advanced Testing Patterns

```python
# Parameterized testing
from commons_testing.parametrize import parametrize_class

@parametrize_class(
    "database",
    ["postgres", "mysql", "sqlite"],
)
class TestDatabaseOperations:
    def test_insert(self, database):
        # Test runs for each database type
        pass

# Property-based testing
from commons_testing.hypothesis import strategies

@given(
    users=strategies.lists(
        strategies.builds(User, name=strategies.text(), age=strategies.integers(0, 120))
    )
)
def test_user_sorting(users):
    sorted_users = sorted(users, key=lambda u: u.age)
    for i in range(len(sorted_users) - 1):
        assert sorted_users[i].age <= sorted_users[i + 1].age

# Snapshot testing
from commons_testing.snapshot import snapshot

def test_api_response(snapshot):
    response = api_call()
    snapshot.assert_match(response.json())

# Performance testing
from commons_testing.performance import benchmark

@benchmark(max_time=0.1)  # 100ms
def test_performance():
    # Code that should execute within 100ms
    process_data(large_dataset)
```

## Fixtures Reference

```python
# Available fixtures
from commons_testing import fixtures

# Async fixtures
async_client      # Async HTTP client
async_db          # Async database connection
async_redis       # Async Redis client
async_event_loop  # Custom event loop

# Database fixtures
postgres_db       # PostgreSQL test database
mysql_db          # MySQL test database
mongodb           # MongoDB test database
redis_cache       # Redis test instance

# Service fixtures
mock_server       # HTTP mock server
kafka_broker      # Kafka test broker
rabbitmq          # RabbitMQ test instance
elasticsearch     # Elasticsearch test instance

# Utility fixtures
temp_dir          # Temporary directory
temp_file         # Temporary file
faker             # Faker instance
time_machine      # Time manipulation
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev,databases,cloud]"

# Run tests
pytest

# Run specific test categories
pytest -m "not slow"
pytest -k "database"
```