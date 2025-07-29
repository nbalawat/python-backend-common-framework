# Commons Testing

Advanced testing utilities for Python applications providing comprehensive fixtures, mocking, data generation, and testing patterns.

## Installation

```bash
pip install commons-testing
```

## Features

- **Async Testing**: Enhanced fixtures and test cases for async/await code
- **Database Testing**: Test containers and fixtures for various databases
- **Integration Testing**: API, message queue, and service testing utilities
- **Test Data Generation**: Factories and generators with realistic test data
- **Mocking**: Enhanced mocking utilities for external services
- **Performance Testing**: Benchmarking and load testing utilities
- **Snapshot Testing**: State comparison and regression testing

## Quick Start

```python
import pytest
from commons_testing import AsyncTestCase, DataGenerator, fake

# Async testing
class TestMyService(AsyncTestCase):
    async def test_async_operation(self):
        result = await my_async_function()
        self.assertEqual(result, "expected")

# Test data generation
def test_user_creation():
    generator = DataGenerator(seed=42)
    user_data = {
        "name": fake.name(),
        "email": fake.email(),
        "age": generator.random_int(18, 65)
    }
    user = create_user(user_data)
    assert user.name == user_data["name"]
```

## Detailed Usage Examples

### Async Testing

#### Basic Async Testing
```python
import pytest
import asyncio
from commons_testing import AsyncTestCase
from commons_testing.fixtures import async_client, async_db

# Using AsyncTestCase
class TestAsyncOperations(AsyncTestCase):
    """Test case with async support."""
    
    async def setUp(self):
        """Async setup method."""
        self.client = await create_test_client()
        self.db = await create_test_db()
    
    async def tearDown(self):
        """Async teardown method."""
        await self.client.close()
        await self.db.close()
    
    async def test_user_creation(self):
        """Test async user creation."""
        user_data = {"name": "John Doe", "email": "john@example.com"}
        
        # Create user via API
        response = await self.client.post("/users", json=user_data)
        self.assertEqual(response.status_code, 201)
        
        # Verify in database
        user = await self.db.fetch_one(
            "SELECT * FROM users WHERE email = $1",
            user_data["email"]
        )
        self.assertIsNotNone(user)
        self.assertEqual(user["name"], user_data["name"])
    
    async def test_concurrent_operations(self):
        """Test concurrent async operations."""
        tasks = [
            self.client.get(f"/users/{i}")
            for i in range(1, 11)
        ]
        
        responses = await asyncio.gather(*tasks)
        
        # All requests should succeed
        for response in responses:
            self.assertEqual(response.status_code, 200)

# Using pytest with async fixtures
@pytest.mark.asyncio
async def test_with_async_fixtures(async_client, async_db):
    """Test using pytest async fixtures."""
    # Insert test data
    await async_db.execute(
        "INSERT INTO users (name, email) VALUES ($1, $2)",
        "Test User", "test@example.com"
    )
    
    # Test API endpoint
    response = await async_client.get("/users/test@example.com")
    assert response.status_code == 200
    
    user_data = response.json()
    assert user_data["name"] == "Test User"
    assert user_data["email"] == "test@example.com"

# Custom async fixtures
from commons_testing.fixtures import async_fixture

@async_fixture
async def redis_client():
    """Provide async Redis client for testing."""
    import aioredis
    
    client = aioredis.from_url("redis://localhost:6379/0")
    await client.flushdb()  # Clean state
    
    yield client
    
    await client.flushdb()  # Cleanup
    await client.close()

@async_fixture
async def message_queue():
    """Provide async message queue for testing."""
    from commons_testing.messaging import TestMessageQueue
    
    queue = TestMessageQueue()
    await queue.start()
    
    yield queue
    
    await queue.stop()

@pytest.mark.asyncio
async def test_caching_service(redis_client):
    """Test service with Redis caching."""
    service = CachingService(redis_client)
    
    # Test cache miss
    result = await service.get_user(123)
    assert result is not None
    
    # Test cache hit
    cached_result = await service.get_user(123)
    assert cached_result == result
    
    # Verify cache was used
    cache_key = "user:123"
    cached_data = await redis_client.get(cache_key)
    assert cached_data is not None
```

### Database Testing

#### Database Containers and Fixtures
```python
import pytest
from commons_testing.databases import (
    PostgresContainer, MySQLContainer, MongoContainer, 
    RedisContainer, DatabaseFixture
)
from sqlalchemy import create_engine, text
import asyncpg
import pymongo

# PostgreSQL testing with container
@pytest.fixture(scope="session")
def postgres_container():
    """Provide PostgreSQL container for entire test session."""
    with PostgresContainer(
        image="postgres:15",
        port=5432,
        username="testuser",
        password="testpass",
        database="testdb"
    ) as container:
        # Wait for database to be ready
        container.wait_for_ready(timeout=30)
        yield container

@pytest.fixture
def postgres_url(postgres_container):
    """Get PostgreSQL connection URL."""
    return postgres_container.get_connection_url()

@pytest.fixture
async def postgres_connection(postgres_url):
    """Provide async PostgreSQL connection."""
    conn = await asyncpg.connect(postgres_url)
    
    # Setup test schema
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    yield conn
    
    # Cleanup
    await conn.execute("TRUNCATE TABLE users RESTART IDENTITY")
    await conn.close()

# Test with PostgreSQL
@pytest.mark.asyncio
async def test_user_crud_operations(postgres_connection):
    """Test CRUD operations with PostgreSQL."""
    conn = postgres_connection
    
    # Create user
    user_id = await conn.fetchval(
        "INSERT INTO users (name, email) VALUES ($1, $2) RETURNING id",
        "John Doe", "john@example.com"
    )
    assert user_id is not None
    
    # Read user
    user = await conn.fetchrow(
        "SELECT * FROM users WHERE id = $1", user_id
    )
    assert user["name"] == "John Doe"
    assert user["email"] == "john@example.com"
    
    # Update user
    await conn.execute(
        "UPDATE users SET name = $1 WHERE id = $2",
        "Jane Doe", user_id
    )
    
    updated_user = await conn.fetchrow(
        "SELECT * FROM users WHERE id = $1", user_id
    )
    assert updated_user["name"] == "Jane Doe"
    
    # Delete user
    deleted_count = await conn.fetchval(
        "DELETE FROM users WHERE id = $1", user_id
    )
    assert deleted_count == 1

# MySQL testing
@pytest.fixture(scope="session")
def mysql_container():
    """Provide MySQL container."""
    with MySQLContainer(
        image="mysql:8.0",
        username="root",
        password="testpass",
        database="testdb"
    ) as container:
        yield container

def test_mysql_operations(mysql_container):
    """Test operations with MySQL."""
    engine = create_engine(mysql_container.get_connection_url())
    
    with engine.connect() as conn:
        # Create table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS products (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                price DECIMAL(10,2) NOT NULL,
                category VARCHAR(50)
            )
        """))
        conn.commit()
        
        # Insert test data
        conn.execute(text("""
            INSERT INTO products (name, price, category) VALUES 
            ('Laptop', 999.99, 'Electronics'),
            ('Book', 29.99, 'Education'),
            ('Coffee', 4.99, 'Food')
        """))
        conn.commit()
        
        # Query data
        result = conn.execute(text(
            "SELECT * FROM products WHERE category = 'Electronics'"
        ))
        products = result.fetchall()
        
        assert len(products) == 1
        assert products[0].name == "Laptop"
        assert float(products[0].price) == 999.99

# MongoDB testing
@pytest.fixture
def mongodb_client(mongodb_container):
    """Provide MongoDB client."""
    client = pymongo.MongoClient(mongodb_container.get_connection_url())
    db = client.testdb
    
    yield db
    
    # Cleanup
    client.drop_database("testdb")
    client.close()

def test_mongodb_operations(mongodb_client):
    """Test MongoDB operations."""
    collection = mongodb_client.users
    
    # Insert documents
    users = [
        {"name": "Alice", "age": 30, "city": "New York"},
        {"name": "Bob", "age": 25, "city": "San Francisco"},
        {"name": "Charlie", "age": 35, "city": "New York"}
    ]
    
    result = collection.insert_many(users)
    assert len(result.inserted_ids) == 3
    
    # Query documents
    ny_users = list(collection.find({"city": "New York"}))
    assert len(ny_users) == 2
    
    # Update document
    collection.update_one(
        {"name": "Alice"},
        {"$set": {"age": 31}}
    )
    
    alice = collection.find_one({"name": "Alice"})
    assert alice["age"] == 31
    
    # Aggregation
    pipeline = [
        {"$group": {"_id": "$city", "avg_age": {"$avg": "$age"}}},
        {"$sort": {"avg_age": -1}}
    ]
    
    result = list(collection.aggregate(pipeline))
    assert len(result) == 2

# Redis testing
@pytest.fixture
def redis_client(redis_container):
    """Provide Redis client."""
    client = redis_container.get_client()
    client.flushdb()  # Clean state
    
    yield client
    
    client.flushdb()  # Cleanup
    client.close()

def test_redis_operations(redis_client):
    """Test Redis operations."""
    # String operations
    redis_client.set("user:1:name", "John Doe")
    assert redis_client.get("user:1:name").decode() == "John Doe"
    
    # Hash operations
    user_data = {
        "name": "Jane Smith",
        "email": "jane@example.com",
        "age": "28"
    }
    redis_client.hset("user:2", mapping=user_data)
    
    stored_user = redis_client.hgetall("user:2")
    assert stored_user[b"name"].decode() == "Jane Smith"
    assert stored_user[b"email"].decode() == "jane@example.com"
    
    # List operations
    redis_client.lpush("messages", "Hello", "World", "Redis")
    messages = [msg.decode() for msg in redis_client.lrange("messages", 0, -1)]
    assert messages == ["Redis", "World", "Hello"]
    
    # Set operations
    redis_client.sadd("tags", "python", "testing", "redis")
    tags = {tag.decode() for tag in redis_client.smembers("tags")}
    assert tags == {"python", "testing", "redis"}
    
    # Expiration
    redis_client.setex("temp_key", 1, "temporary_value")
    assert redis_client.get("temp_key").decode() == "temporary_value"
    assert redis_client.ttl("temp_key") <= 1

# Database fixtures with automatic migration
@pytest.fixture
def migrated_database():
    """Provide database with migrations applied."""
    with PostgresContainer() as container:
        db_url = container.get_connection_url()
        
        # Apply migrations
        from alembic.config import Config
        from alembic import command
        
        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", db_url)
        command.upgrade(alembic_cfg, "head")
        
        yield db_url
```

### Integration Testing

#### API Testing
```python
import pytest
import json
from commons_testing.integration import APITestClient, MockServer
from commons_testing.auth import create_test_token

# Advanced API client
class TestAPIClient:
    """Enhanced API test client with authentication and retry."""
    
    def __init__(self, base_url: str):
        self.client = APITestClient(
            base_url=base_url,
            timeout=30,
            retry_attempts=3
        )
        self.auth_token = None
    
    async def authenticate(self, username: str, password: str):
        """Authenticate and store token."""
        response = await self.client.post("/auth/login", json={
            "username": username,
            "password": password
        })
        assert response.status_code == 200
        
        self.auth_token = response.json()["access_token"]
        self.client.set_auth_header(f"Bearer {self.auth_token}")
    
    async def create_user(self, user_data: dict):
        """Helper to create user via API."""
        response = await self.client.post("/users", json=user_data)
        assert response.status_code == 201
        return response.json()
    
    async def get_user(self, user_id: int):
        """Helper to get user via API."""
        response = await self.client.get(f"/users/{user_id}")
        assert response.status_code == 200
        return response.json()

@pytest.fixture
async def authenticated_client():
    """Provide authenticated API client."""
    client = TestAPIClient("http://localhost:8000")
    await client.authenticate("testuser", "testpass")
    yield client
    await client.client.close()

@pytest.mark.asyncio
async def test_user_lifecycle(authenticated_client):
    """Test complete user lifecycle through API."""
    # Create user
    user_data = {
        "name": "John Doe",
        "email": "john@example.com",
        "role": "user"
    }
    
    created_user = await authenticated_client.create_user(user_data)
    user_id = created_user["id"]
    
    assert created_user["name"] == user_data["name"]
    assert created_user["email"] == user_data["email"]
    assert "created_at" in created_user
    
    # Get user
    retrieved_user = await authenticated_client.get_user(user_id)
    assert retrieved_user == created_user
    
    # Update user
    update_data = {"name": "Jane Doe"}
    response = await authenticated_client.client.patch(
        f"/users/{user_id}", json=update_data
    )
    assert response.status_code == 200
    
    updated_user = response.json()
    assert updated_user["name"] == "Jane Doe"
    assert updated_user["email"] == user_data["email"]
    
    # Delete user
    response = await authenticated_client.client.delete(f"/users/{user_id}")
    assert response.status_code == 204
    
    # Verify deletion
    response = await authenticated_client.client.get(f"/users/{user_id}")
    assert response.status_code == 404

# Mock external services
@pytest.fixture
def payment_service_mock():
    """Mock external payment service."""
    with MockServer(port=9001) as server:
        # Mock successful payment
        server.expect(
            method="POST",
            path="/payments",
            response={
                "payment_id": "pay_123456",
                "status": "completed",
                "amount": 100.00
            },
            status_code=200,
            headers={"Content-Type": "application/json"}
        )
        
        # Mock payment failure
        server.expect(
            method="POST",
            path="/payments/fail",
            response={"error": "Insufficient funds"},
            status_code=400
        )
        
        # Mock payment status check
        server.expect(
            method="GET",
            path="/payments/pay_123456",
            response={
                "payment_id": "pay_123456",
                "status": "completed",
                "created_at": "2024-01-01T12:00:00Z"
            },
            status_code=200
        )
        
        yield server

@pytest.mark.asyncio
async def test_payment_integration(authenticated_client, payment_service_mock):
    """Test payment service integration."""
    # Create order
    order_data = {
        "items": [{"product_id": 1, "quantity": 2, "price": 50.00}],
        "total": 100.00
    }
    
    response = await authenticated_client.client.post("/orders", json=order_data)
    assert response.status_code == 201
    
    order = response.json()
    order_id = order["id"]
    
    # Process payment
    payment_data = {
        "order_id": order_id,
        "amount": 100.00,
        "payment_method": "credit_card",
        "card_token": "tok_123456"
    }
    
    response = await authenticated_client.client.post(
        f"/orders/{order_id}/payment", json=payment_data
    )
    assert response.status_code == 200
    
    payment_result = response.json()
    assert payment_result["status"] == "completed"
    assert payment_result["payment_id"] == "pay_123456"
    
    # Verify order status updated
    order = await authenticated_client.client.get(f"/orders/{order_id}")
    assert order.json()["status"] == "paid"

#### Message Queue Testing
```python
from commons_testing.messaging import (
    KafkaContainer, RabbitMQContainer, RedisStreams
)
import asyncio
from kafka import KafkaProducer, KafkaConsumer
import pika
import json

# Kafka testing
@pytest.fixture(scope="session")
def kafka_cluster():
    """Provide Kafka cluster for testing."""
    with KafkaContainer(
        image="confluentinc/cp-kafka:latest",
        port=9092
    ) as kafka:
        yield kafka

@pytest.fixture
def kafka_producer(kafka_cluster):
    """Provide Kafka producer."""
    producer = KafkaProducer(
        bootstrap_servers=[kafka_cluster.get_bootstrap_servers()],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    yield producer
    producer.close()

@pytest.fixture
def kafka_consumer(kafka_cluster):
    """Provide Kafka consumer."""
    consumer = KafkaConsumer(
        bootstrap_servers=[kafka_cluster.get_bootstrap_servers()],
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        consumer_timeout_ms=5000,
        auto_offset_reset='earliest'
    )
    yield consumer
    consumer.close()

def test_kafka_message_flow(kafka_producer, kafka_consumer):
    """Test Kafka producer-consumer flow."""
    topic = "test-events"
    
    # Subscribe consumer to topic
    kafka_consumer.subscribe([topic])
    
    # Send messages
    test_messages = [
        {"event_type": "user_created", "user_id": 1, "timestamp": "2024-01-01T12:00:00Z"},
        {"event_type": "order_placed", "order_id": 123, "amount": 99.99},
        {"event_type": "payment_completed", "payment_id": "pay_456"}
    ]
    
    for message in test_messages:
        kafka_producer.send(topic, message)
    
    kafka_producer.flush()  # Ensure messages are sent
    
    # Consume messages
    received_messages = []
    for message in kafka_consumer:
        received_messages.append(message.value)
        if len(received_messages) == len(test_messages):
            break
    
    assert len(received_messages) == len(test_messages)
    assert received_messages[0]["event_type"] == "user_created"
    assert received_messages[1]["order_id"] == 123
    assert received_messages[2]["payment_id"] == "pay_456"

# RabbitMQ testing
@pytest.fixture
def rabbitmq_connection(rabbitmq_container):
    """Provide RabbitMQ connection."""
    connection = pika.BlockingConnection(
        pika.URLParameters(rabbitmq_container.get_connection_url())
    )
    yield connection
    connection.close()

def test_rabbitmq_queue_operations(rabbitmq_connection):
    """Test RabbitMQ queue operations."""
    channel = rabbitmq_connection.channel()
    
    # Declare queue
    queue_name = "test_queue"
    channel.queue_declare(queue=queue_name, durable=True)
    
    # Publish messages
    messages = [
        "Hello RabbitMQ",
        "Message 2",
        "Final message"
    ]
    
    for message in messages:
        channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=message,
            properties=pika.BasicProperties(delivery_mode=2)  # Persistent
        )
    
    # Consume messages
    received_messages = []
    
    def callback(ch, method, properties, body):
        received_messages.append(body.decode())
        ch.basic_ack(delivery_tag=method.delivery_tag)
    
    channel.basic_consume(queue=queue_name, on_message_callback=callback)
    
    # Process messages with timeout
    import time
    start_time = time.time()
    while len(received_messages) < len(messages) and time.time() - start_time < 10:
        rabbitmq_connection.process_data_events(time_limit=1)
    
    assert len(received_messages) == len(messages)
    assert received_messages == messages
    
    channel.close()

# Service integration testing
@pytest.fixture
async def microservices_setup():
    """Setup multiple services for integration testing."""
    services = {
        "user_service": TestService("user-service", port=8001),
        "order_service": TestService("order-service", port=8002),
        "payment_service": TestService("payment-service", port=8003)
    }
    
    # Start all services
    for service in services.values():
        await service.start()
    
    yield services
    
    # Stop all services
    for service in services.values():
        await service.stop()

@pytest.mark.asyncio
async def test_microservices_integration(microservices_setup):
    """Test integration between multiple microservices."""
    services = microservices_setup
    
    # Create user in user service
    user_response = await services["user_service"].post("/users", {
        "name": "John Doe",
        "email": "john@example.com"
    })
    user_id = user_response["id"]
    
    # Create order in order service
    order_response = await services["order_service"].post("/orders", {
        "user_id": user_id,
        "items": [{"product_id": 1, "quantity": 2, "price": 25.00}],
        "total": 50.00
    })
    order_id = order_response["id"]
    
    # Process payment in payment service
    payment_response = await services["payment_service"].post("/payments", {
        "order_id": order_id,
        "amount": 50.00,
        "method": "credit_card"
    })
    
    assert payment_response["status"] == "completed"
    
    # Verify order status updated via service communication
    updated_order = await services["order_service"].get(f"/orders/{order_id}")
    assert updated_order["status"] == "paid"
    assert updated_order["payment_id"] == payment_response["payment_id"]
```

### Test Data Generation

#### Factories and Realistic Data
```python
from commons_testing.factories import Factory, SubFactory, LazyAttribute, Sequence
from commons_testing.generators import DataGenerator, fake
from datetime import datetime, timedelta
import uuid
from typing import List, Dict, Any

# Advanced factory definitions
class UserFactory(Factory):
    """Factory for creating test users."""
    
    class Meta:
        model = User
    
    id = Sequence(lambda n: n)
    uuid = LazyAttribute(lambda obj: str(uuid.uuid4()))
    username = LazyAttribute(lambda obj: fake.user_name())
    email = LazyAttribute(lambda obj: f"{obj.username}@{fake.domain_name()}")
    first_name = fake.first_name
    last_name = fake.last_name
    full_name = LazyAttribute(lambda obj: f"{obj.first_name} {obj.last_name}")
    date_of_birth = fake.date_of_birth(minimum_age=18, maximum_age=80)
    phone_number = fake.phone_number
    address = SubFactory('AddressFactory')
    is_active = True
    role = fake.random_element(["user", "admin", "moderator"])
    created_at = fake.date_time_between(start_date="-1y", end_date="now")
    updated_at = LazyAttribute(lambda obj: fake.date_time_between(
        start_date=obj.created_at, end_date="now"
    ))
    
    @classmethod
    def create_admin(cls, **kwargs):
        """Create admin user."""
        return cls(role="admin", is_active=True, **kwargs)
    
    @classmethod
    def create_inactive(cls, **kwargs):
        """Create inactive user."""
        return cls(is_active=False, **kwargs)

class AddressFactory(Factory):
    """Factory for creating addresses."""
    
    class Meta:
        model = Address
    
    street_address = fake.street_address
    city = fake.city
    state = fake.state
    postal_code = fake.postcode
    country = fake.country_code
    
class ProductFactory(Factory):
    """Factory for creating products."""
    
    class Meta:
        model = Product
    
    id = Sequence(lambda n: n)
    name = fake.catch_phrase
    description = fake.text(max_nb_chars=200)
    price = LazyAttribute(lambda obj: round(fake.random.uniform(9.99, 999.99), 2))
    category = fake.random_element([
        "Electronics", "Books", "Clothing", "Home", "Sports", "Toys"
    ])
    sku = LazyAttribute(lambda obj: f"{obj.category[:3].upper()}-{fake.random_int(1000, 9999)}")
    stock_quantity = fake.random_int(0, 1000)
    is_available = LazyAttribute(lambda obj: obj.stock_quantity > 0)
    weight = LazyAttribute(lambda obj: round(fake.random.uniform(0.1, 50.0), 2))
    dimensions = LazyAttribute(lambda obj: {
        "length": fake.random.uniform(1, 100),
        "width": fake.random.uniform(1, 100),
        "height": fake.random.uniform(1, 100)
    })
    tags = LazyAttribute(lambda obj: fake.words(nb=fake.random_int(1, 5)))

class OrderFactory(Factory):
    """Factory for creating orders."""
    
    class Meta:
        model = Order
    
    id = Sequence(lambda n: n)
    order_number = LazyAttribute(lambda obj: f"ORD-{fake.random_int(100000, 999999)}")
    user = SubFactory(UserFactory)
    items = LazyAttribute(lambda obj: [
        OrderItemFactory(order_id=obj.id) for _ in range(fake.random_int(1, 5))
    ])
    subtotal = LazyAttribute(lambda obj: sum(item.total_price for item in obj.items))
    tax_amount = LazyAttribute(lambda obj: round(obj.subtotal * 0.08, 2))
    shipping_cost = LazyAttribute(lambda obj: 9.99 if obj.subtotal < 50 else 0)
    total_amount = LazyAttribute(lambda obj: obj.subtotal + obj.tax_amount + obj.shipping_cost)
    status = fake.random_element(["pending", "confirmed", "shipped", "delivered", "cancelled"])
    payment_method = fake.random_element(["credit_card", "paypal", "bank_transfer"])
    shipping_address = SubFactory(AddressFactory)
    billing_address = LazyAttribute(lambda obj: obj.shipping_address)
    order_date = fake.date_time_between(start_date="-30d", end_date="now")
    shipped_date = LazyAttribute(lambda obj: (
        fake.date_time_between(start_date=obj.order_date, end_date="now")
        if obj.status in ["shipped", "delivered"] else None
    ))

class OrderItemFactory(Factory):
    """Factory for order items."""
    
    class Meta:
        model = OrderItem
    
    product = SubFactory(ProductFactory)
    quantity = fake.random_int(1, 10)
    unit_price = LazyAttribute(lambda obj: obj.product.price)
    total_price = LazyAttribute(lambda obj: obj.quantity * obj.unit_price)

# Comprehensive test data generation
class TestDataGenerator:
    """Advanced test data generator with realistic relationships."""
    
    def __init__(self, seed: int = None):
        self.generator = DataGenerator(seed=seed)
        if seed:
            fake.seed_instance(seed)
    
    def create_user_ecosystem(self, num_users: int = 10) -> Dict[str, List[Any]]:
        """Create complete user ecosystem with related data."""
        users = UserFactory.create_batch(num_users)
        
        # Create orders for random users
        orders = []
        for _ in range(num_users * 2):  # 2 orders per user on average
            user = fake.random_element(users)
            order = OrderFactory(user=user)
            orders.append(order)
        
        # Create products used in orders
        products = set()
        for order in orders:
            for item in order.items:
                products.add(item.product)
        
        return {
            "users": users,
            "orders": orders,
            "products": list(products),
            "addresses": [user.address for user in users]
        }
    
    def create_time_series_data(
        self, 
        start_date: str, 
        end_date: str, 
        frequency: str = "daily"
    ) -> List[Dict[str, Any]]:
        """Create time series test data."""
        from datetime import datetime, timedelta
        
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        
        delta_map = {
            "hourly": timedelta(hours=1),
            "daily": timedelta(days=1),
            "weekly": timedelta(weeks=1),
            "monthly": timedelta(days=30)
        }
        
        delta = delta_map.get(frequency, timedelta(days=1))
        
        data_points = []
        current_date = start
        
        while current_date <= end:
            data_points.append({
                "timestamp": current_date.isoformat(),
                "value": fake.random.uniform(0, 1000),
                "category": fake.random_element(["A", "B", "C", "D"]),
                "metadata": {
                    "source": fake.random_element(["web", "mobile", "api"]),
                    "region": fake.random_element(["us-east", "us-west", "eu-central"])
                }
            })
            current_date += delta
        
        return data_points
    
    def create_hierarchical_data(self, depth: int = 3, breadth: int = 3) -> Dict[str, Any]:
        """Create hierarchical test data structure."""
        def create_node(level: int, parent_id: str = None) -> Dict[str, Any]:
            node_id = str(uuid.uuid4())
            node = {
                "id": node_id,
                "name": fake.catch_phrase(),
                "description": fake.sentence(),
                "level": level,
                "parent_id": parent_id,
                "metadata": {
                    "created_at": fake.date_time().isoformat(),
                    "tags": fake.words(nb=fake.random_int(1, 3))
                },
                "children": []
            }
            
            if level < depth:
                node["children"] = [
                    create_node(level + 1, node_id) 
                    for _ in range(fake.random_int(1, breadth))
                ]
            
            return node
        
        return create_node(0)
    
    def create_event_stream(
        self, 
        num_events: int = 100, 
        event_types: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Create realistic event stream data."""
        if not event_types:
            event_types = [
                "user.created", "user.updated", "user.deleted",
                "order.placed", "order.confirmed", "order.shipped", "order.delivered",
                "payment.initiated", "payment.completed", "payment.failed",
                "product.created", "product.updated", "product.deleted"
            ]
        
        events = []
        base_time = fake.date_time_between(start_date="-7d", end_date="now")
        
        for i in range(num_events):
            event_time = base_time + timedelta(seconds=i * fake.random_int(1, 300))
            
            event = {
                "id": str(uuid.uuid4()),
                "event_type": fake.random_element(event_types),
                "timestamp": event_time.isoformat(),
                "source": fake.random_element(["web-app", "mobile-app", "api", "admin"]),
                "user_id": fake.random_int(1, 1000),
                "session_id": str(uuid.uuid4()),
                "data": {
                    "ip_address": fake.ipv4(),
                    "user_agent": fake.user_agent(),
                    "referrer": fake.url(),
                    "custom_field": fake.sentence()
                },
                "version": "1.0"
            }
            
            events.append(event)
        
        return events

# Usage examples
def test_factory_usage():
    """Test various factory usage patterns."""
    # Basic usage
    user = UserFactory()
    assert user.username
    assert "@" in user.email
    assert user.created_at <= user.updated_at
    
    # Custom attributes
    admin = UserFactory.create_admin(first_name="Admin", last_name="User")
    assert admin.role == "admin"
    assert admin.full_name == "Admin User"
    
    # Batch creation
    users = UserFactory.create_batch(5, is_active=True)
    assert len(users) == 5
    assert all(user.is_active for user in users)
    
    # Related objects
    order = OrderFactory(user=users[0])
    assert order.user == users[0]
    assert len(order.items) > 0
    assert order.total_amount > 0

def test_data_generator():
    """Test comprehensive data generation."""
    generator = TestDataGenerator(seed=42)
    
    # Create user ecosystem
    ecosystem = generator.create_user_ecosystem(num_users=5)
    
    assert len(ecosystem["users"]) == 5
    assert len(ecosystem["orders"]) == 10  # 2 per user
    assert len(ecosystem["products"]) > 0
    assert len(ecosystem["addresses"]) == 5
    
    # Create time series data
    time_series = generator.create_time_series_data(
        start_date="2024-01-01T00:00:00",
        end_date="2024-01-07T23:59:59",
        frequency="daily"
    )
    
    assert len(time_series) == 7  # 7 days
    assert all("timestamp" in point for point in time_series)
    assert all("value" in point for point in time_series)
    
    # Create hierarchical data
    hierarchy = generator.create_hierarchical_data(depth=3, breadth=2)
    
    assert hierarchy["level"] == 0
    assert len(hierarchy["children"]) >= 1
    assert all(child["level"] == 1 for child in hierarchy["children"])
    
    # Create event stream
    events = generator.create_event_stream(num_events=50)
    
    assert len(events) == 50
    assert all("event_type" in event for event in events)
    assert all("timestamp" in event for event in events)
    
    # Events should be chronologically ordered
    timestamps = [event["timestamp"] for event in events]
    assert timestamps == sorted(timestamps)
```

### Cloud Service Mocking

#### AWS Service Mocking
```python
import pytest
import boto3
from moto import mock_s3, mock_dynamodb2, mock_sqs, mock_lambda
from commons_testing.cloud import mock_aws, LocalStackContainer
import json
from datetime import datetime

# S3 Testing
@mock_s3
def test_s3_operations():
    """Test S3 operations with moto."""
    # Create S3 client
    s3_client = boto3.client("s3", region_name="us-east-1")
    
    # Create bucket
    bucket_name = "test-bucket"
    s3_client.create_bucket(Bucket=bucket_name)
    
    # Upload file
    file_content = b"This is test file content"
    s3_client.put_object(
        Bucket=bucket_name,
        Key="test-folder/test-file.txt",
        Body=file_content,
        ContentType="text/plain",
        Metadata={"author": "test-user", "version": "1.0"}
    )
    
    # List objects
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    assert response["KeyCount"] == 1
    assert response["Contents"][0]["Key"] == "test-folder/test-file.txt"
    
    # Get object
    response = s3_client.get_object(Bucket=bucket_name, Key="test-folder/test-file.txt")
    assert response["Body"].read() == file_content
    assert response["Metadata"]["author"] == "test-user"
    
    # Copy object
    s3_client.copy_object(
        CopySource={"Bucket": bucket_name, "Key": "test-folder/test-file.txt"},
        Bucket=bucket_name,
        Key="backup/test-file-copy.txt"
    )
    
    # Verify copy
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    assert response["KeyCount"] == 2
    
    # Delete object
    s3_client.delete_object(Bucket=bucket_name, Key="test-folder/test-file.txt")
    
    # Verify deletion
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    assert response["KeyCount"] == 1

# DynamoDB Testing
@mock_dynamodb2
def test_dynamodb_operations():
    """Test DynamoDB operations with moto."""
    dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
    
    # Create table
    table_name = "test-users"
    table = dynamodb.create_table(
        TableName=table_name,
        KeySchema=[
            {"AttributeName": "user_id", "KeyType": "HASH"},
            {"AttributeName": "created_at", "KeyType": "RANGE"}
        ],
        AttributeDefinitions=[
            {"AttributeName": "user_id", "AttributeType": "S"},
            {"AttributeName": "created_at", "AttributeType": "S"},
            {"AttributeName": "email", "AttributeType": "S"}
        ],
        GlobalSecondaryIndexes=[
            {
                "IndexName": "email-index",
                "KeySchema": [{"AttributeName": "email", "KeyType": "HASH"}],
                "Projection": {"ProjectionType": "ALL"},
                "ProvisionedThroughput": {"ReadCapacityUnits": 5, "WriteCapacityUnits": 5}
            }
        ],
        BillingMode="PROVISIONED",
        ProvisionedThroughput={"ReadCapacityUnits": 5, "WriteCapacityUnits": 5}
    )
    
    # Put items
    test_users = [
        {
            "user_id": "user-1",
            "email": "user1@example.com",
            "name": "Alice Johnson",
            "created_at": "2024-01-01T10:00:00Z",
            "active": True
        },
        {
            "user_id": "user-2",
            "email": "user2@example.com", 
            "name": "Bob Smith",
            "created_at": "2024-01-02T10:00:00Z",
            "active": False
        }
    ]
    
    for user in test_users:
        table.put_item(Item=user)
    
    # Get item
    response = table.get_item(
        Key={"user_id": "user-1", "created_at": "2024-01-01T10:00:00Z"}
    )
    assert "Item" in response
    assert response["Item"]["name"] == "Alice Johnson"
    
    # Query by partition key
    response = table.query(
        KeyConditionExpression="user_id = :user_id",
        ExpressionAttributeValues={":user_id": "user-1"}
    )
    assert response["Count"] == 1
    
    # Scan with filter
    response = table.scan(
        FilterExpression="active = :active",
        ExpressionAttributeValues={":active": True}
    )
    assert response["Count"] == 1
    assert response["Items"][0]["name"] == "Alice Johnson"
    
    # Update item
    table.update_item(
        Key={"user_id": "user-2", "created_at": "2024-01-02T10:00:00Z"},
        UpdateExpression="SET active = :active, updated_at = :updated_at",
        ExpressionAttributeValues={
            ":active": True,
            ":updated_at": datetime.now().isoformat()
        }
    )
    
    # Verify update
    response = table.get_item(
        Key={"user_id": "user-2", "created_at": "2024-01-02T10:00:00Z"}
    )
    assert response["Item"]["active"] is True
    assert "updated_at" in response["Item"]

# SQS Testing
@mock_sqs
def test_sqs_operations():
    """Test SQS operations with moto."""
    sqs = boto3.client("sqs", region_name="us-east-1")
    
    # Create queue
    queue_name = "test-queue"
    response = sqs.create_queue(
        QueueName=queue_name,
        Attributes={
            "DelaySeconds": "0",
            "MessageRetentionPeriod": "1209600",  # 14 days
            "VisibilityTimeoutSeconds": "60"
        }
    )
    queue_url = response["QueueUrl"]
    
    # Send messages
    messages = [
        {"id": 1, "action": "create_user", "data": {"name": "Alice"}},
        {"id": 2, "action": "update_user", "data": {"id": 1, "active": True}},
        {"id": 3, "action": "delete_user", "data": {"id": 1}}
    ]
    
    for msg in messages:
        sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(msg),
            MessageAttributes={
                "action": {
                    "StringValue": msg["action"],
                    "DataType": "String"
                }
            }
        )
    
    # Receive messages
    received_messages = []
    while len(received_messages) < len(messages):
        response = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=10,
            MessageAttributeNames=["All"],
            WaitTimeSeconds=1
        )
        
        if "Messages" in response:
            for message in response["Messages"]:
                msg_body = json.loads(message["Body"])
                received_messages.append(msg_body)
                
                # Delete message after processing
                sqs.delete_message(
                    QueueUrl=queue_url,
                    ReceiptHandle=message["ReceiptHandle"]
                )
    
    assert len(received_messages) == len(messages)
    assert received_messages[0]["action"] == "create_user"

# Lambda Testing  
@mock_lambda
def test_lambda_operations():
    """Test Lambda operations with moto."""
    lambda_client = boto3.client("lambda", region_name="us-east-1")
    
    # Create Lambda function
    function_name = "test-function"
    
    # Simple Lambda function code
    lambda_code = """
def lambda_handler(event, context):
    name = event.get('name', 'World')
    return {
        'statusCode': 200,
        'body': f'Hello, {name}!'
    }
"""
    
    import zipfile
    import io
    
    # Create deployment package
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr('lambda_function.py', lambda_code)
    zip_buffer.seek(0)
    
    lambda_client.create_function(
        FunctionName=function_name,
        Runtime="python3.9",
        Role="arn:aws:iam::123456789012:role/test-role",
        Handler="lambda_function.lambda_handler",
        Code={"ZipFile": zip_buffer.read()},
        Description="Test Lambda function",
        Timeout=30,
        MemorySize=128
    )
    
    # Invoke function
    response = lambda_client.invoke(
        FunctionName=function_name,
        InvocationType="RequestResponse",
        Payload=json.dumps({"name": "Test User"})
    )
    
    assert response["StatusCode"] == 200
    
    payload = json.loads(response["Payload"].read())
    assert payload["statusCode"] == 200
    assert "Hello, Test User!" in payload["body"]

#### LocalStack Integration
@pytest.fixture(scope="session")
def localstack_container():
    """Provide LocalStack container for AWS service emulation."""
    with LocalStackContainer(
        image="localstack/localstack:latest",
        services=["s3", "dynamodb", "sqs", "lambda", "sns", "kinesis"],
        environment={
            "SERVICES": "s3,dynamodb,sqs,lambda,sns,kinesis",
            "DEBUG": "1",
            "DATA_DIR": "/tmp/localstack/data",
            "LAMBDA_EXECUTOR": "docker"
        }
    ) as container:
        yield container

def test_localstack_integration(localstack_container):
    """Test AWS services with LocalStack."""
    # Configure boto3 to use LocalStack
    endpoint_url = localstack_container.get_service_url("s3")
    
    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name="us-east-1"
    )
    
    # Test S3 operations
    bucket_name = "localstack-test-bucket"
    s3_client.create_bucket(Bucket=bucket_name)
    
    s3_client.put_object(
        Bucket=bucket_name,
        Key="test.json",
        Body=json.dumps({"message": "Hello LocalStack!"}),
        ContentType="application/json"
    )
    
    response = s3_client.get_object(Bucket=bucket_name, Key="test.json")
    content = json.loads(response["Body"].read())
    
    assert content["message"] == "Hello LocalStack!"

#### GCP Service Mocking
@pytest.fixture
def mock_gcp_services():
    """Mock GCP services for testing."""
    from commons_testing.cloud import mock_gcp
    
    with mock_gcp() as mock:
        yield mock

def test_gcs_operations(mock_gcp_services):
    """Test Google Cloud Storage operations."""
    from google.cloud import storage
    
    # Create client
    client = storage.Client(project="test-project")
    
    # Create bucket
    bucket_name = "test-gcs-bucket"
    bucket = client.create_bucket(bucket_name)
    
    # Upload blob
    blob = bucket.blob("test-file.txt")
    test_content = "This is test content for GCS"
    blob.upload_from_string(test_content)
    
    # Download blob
    downloaded_content = blob.download_as_text()
    assert downloaded_content == test_content
    
    # List blobs
    blobs = list(bucket.list_blobs())
    assert len(blobs) == 1
    assert blobs[0].name == "test-file.txt"
    
    # Update metadata
    blob.metadata = {"author": "test-user", "version": "1.0"}
    blob.patch()
    
    # Verify metadata
    blob.reload()
    assert blob.metadata["author"] == "test-user"

def test_bigquery_operations(mock_gcp_services):
    """Test BigQuery operations."""
    from google.cloud import bigquery
    
    client = bigquery.Client(project="test-project")
    
    # Create dataset
    dataset_id = "test_dataset"
    dataset = bigquery.Dataset(f"test-project.{dataset_id}")
    dataset = client.create_dataset(dataset)
    
    # Create table
    table_id = "test_table"
    schema = [
        bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("age", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED")
    ]
    
    table_ref = dataset.table(table_id)
    table = bigquery.Table(table_ref, schema=schema)
    table = client.create_table(table)
    
    # Insert data
    rows_to_insert = [
        {"id": "1", "name": "Alice", "age": 30, "created_at": "2024-01-01 10:00:00"},
        {"id": "2", "name": "Bob", "age": 25, "created_at": "2024-01-02 10:00:00"}
    ]
    
    errors = client.insert_rows_json(table, rows_to_insert)
    assert len(errors) == 0
    
    # Query data
    query = f"""
        SELECT id, name, age
        FROM `test-project.{dataset_id}.{table_id}`
        WHERE age > 25
        ORDER BY age DESC
    """
    
    query_job = client.query(query)
    results = list(query_job)
    
    assert len(results) == 1
    assert results[0].name == "Alice"
    assert results[0].age == 30
```

### Advanced Testing Patterns

#### Parameterized and Property-Based Testing
```python
import pytest
from hypothesis import given, strategies as st, assume, settings
from commons_testing.parametrize import parametrize_class, parametrize_method
from commons_testing.snapshot import SnapshotTester
from commons_testing.performance import PerformanceTester, benchmark
from typing import List
import time
import json

# Parameterized testing with multiple databases
@parametrize_class(
    "database_config",
    [
        {"type": "postgres", "url": "postgresql://test:test@localhost:5432/testdb"},
        {"type": "mysql", "url": "mysql://test:test@localhost:3306/testdb"},
        {"type": "sqlite", "url": "sqlite:///test.db"}
    ],
    ids=["postgres", "mysql", "sqlite"]
)
class TestDatabaseOperations:
    """Test database operations across multiple database types."""
    
    def setup_method(self, method):
        """Setup database connection."""
        self.db = create_database_connection(self.database_config["url"])
        self.db.execute("CREATE TABLE users (id INT, name VARCHAR(100), email VARCHAR(100))")
    
    def teardown_method(self, method):
        """Cleanup database."""
        self.db.execute("DROP TABLE IF EXISTS users")
        self.db.close()
    
    def test_insert_user(self):
        """Test user insertion works across all database types."""
        user_data = {"id": 1, "name": "John Doe", "email": "john@example.com"}
        
        result = self.db.execute(
            "INSERT INTO users (id, name, email) VALUES (%(id)s, %(name)s, %(email)s)",
            user_data
        )
        
        # Verify insertion
        user = self.db.fetch_one("SELECT * FROM users WHERE id = %(id)s", {"id": 1})
        assert user["name"] == "John Doe"
        assert user["email"] == "john@example.com"
    
    def test_query_performance(self):
        """Test query performance across database types."""
        # Insert test data
        users = [(i, f"User{i}", f"user{i}@example.com") for i in range(1000)]
        self.db.execute_many(
            "INSERT INTO users (id, name, email) VALUES (?, ?, ?)",
            users
        )
        
        start_time = time.time()
        results = self.db.fetch_all("SELECT * FROM users WHERE id > 500")
        execution_time = time.time() - start_time
        
        assert len(results) == 499
        assert execution_time < 0.5  # Should complete within 500ms

# Method-level parameterization
class TestUserValidation:
    @parametrize_method("email,is_valid", [
        ("valid@example.com", True),
        ("also.valid+tag@example.co.uk", True),
        ("invalid-email", False),
        ("@invalid.com", False),
        ("invalid@", False),
        ("", False)
    ])
    def test_email_validation(self, email, is_valid):
        """Test email validation with various inputs."""
        result = validate_email(email)
        assert result == is_valid

# Property-based testing with Hypothesis
@given(
    users=st.lists(
        st.builds(
            User,
            name=st.text(min_size=1, max_size=50),
            age=st.integers(min_value=0, max_value=120),
            email=st.emails()
        ),
        min_size=0,
        max_size=100
    )
)
@settings(max_examples=100, deadline=5000)  # Run 100 examples, 5s deadline
def test_user_sorting_properties(users: List[User]):
    """Test that user sorting maintains invariants."""
    assume(len(users) > 0)  # Skip empty lists
    
    # Sort by age
    sorted_by_age = sorted(users, key=lambda u: u.age)
    
    # Property: sorted list should have same length
    assert len(sorted_by_age) == len(users)
    
    # Property: all users should still be present
    assert set(u.name for u in sorted_by_age) == set(u.name for u in users)
    
    # Property: ages should be in non-decreasing order
    for i in range(len(sorted_by_age) - 1):
        assert sorted_by_age[i].age <= sorted_by_age[i + 1].age
    
    # Property: minimum and maximum should be at correct positions
    if len(sorted_by_age) > 1:
        min_age = min(u.age for u in users)
        max_age = max(u.age for u in users)
        assert sorted_by_age[0].age == min_age
        assert sorted_by_age[-1].age == max_age

@given(
    data=st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.one_of(
            st.text(),
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.booleans()
        ),
        min_size=1
    )
)
def test_json_serialization_roundtrip(data):
    """Test that JSON serialization is idempotent."""
    # Serialize to JSON
    json_str = json.dumps(data, sort_keys=True)
    
    # Deserialize back
    roundtrip_data = json.loads(json_str)
    
    # Should be equal to original
    assert roundtrip_data == data
    
    # Second roundtrip should be identical
    json_str2 = json.dumps(roundtrip_data, sort_keys=True)
    assert json_str == json_str2

#### Snapshot Testing
@pytest.fixture
def snapshot():
    """Provide snapshot tester."""
    return SnapshotTester()

def test_api_response_structure(snapshot):
    """Test API response structure remains consistent."""
    # Make API call
    response = api_client.get("/users/123")
    
    # Remove dynamic fields
    data = response.json()
    data.pop("last_login", None)  # Remove timestamp
    data.pop("session_id", None)  # Remove session ID
    
    # Assert structure matches snapshot
    snapshot.assert_match(data, "user_response_structure.json")

def test_html_rendering(snapshot):
    """Test HTML template rendering."""
    user = User(name="John Doe", email="john@example.com")
    posts = [Post(title="Post 1", content="Content 1")]
    
    html = render_template("user_profile.html", user=user, posts=posts)
    
    # Normalize whitespace and remove timestamps
    normalized_html = normalize_html(html)
    
    snapshot.assert_match(normalized_html, "user_profile.html")

def test_database_migration_result(snapshot):
    """Test database schema after migration."""
    # Run migration
    run_migration("001_create_users_table")
    
    # Get schema information
    schema = get_table_schema("users")
    
    # Remove database-specific details
    normalized_schema = {
        "columns": [
            {"name": col["name"], "type": normalize_type(col["type"]), "nullable": col["nullable"]}
            for col in schema["columns"]
        ],
        "indexes": [idx["name"] for idx in schema["indexes"]],
        "constraints": [const["name"] for const in schema["constraints"]]
    }
    
    snapshot.assert_match(normalized_schema, "users_table_schema.json")

#### Performance Testing
class TestPerformance:
    """Performance testing suite."""
    
    def setup_method(self):
        """Setup performance testing environment."""
        self.perf_tester = PerformanceTester()
        self.large_dataset = generate_test_data(size=10000)
    
    @benchmark(max_time=0.1, max_memory_mb=50)
    def test_data_processing_performance(self):
        """Test data processing performance requirements."""
        result = process_data(self.large_dataset)
        assert len(result) == len(self.large_dataset)
    
    def test_api_endpoint_performance(self):
        """Test API endpoint performance under load."""
        endpoint = "/api/search"
        
        # Warm up
        for _ in range(10):
            api_client.get(endpoint, params={"q": "test"})
        
        # Measure performance
        with self.perf_tester.measure() as measurement:
            responses = []
            for i in range(100):
                response = api_client.get(endpoint, params={"q": f"query{i}"})
                responses.append(response)
        
        # Assert performance requirements
        assert measurement.avg_response_time < 0.1  # 100ms average
        assert measurement.p95_response_time < 0.2   # 200ms 95th percentile
        assert measurement.max_response_time < 0.5   # 500ms maximum
        assert all(r.status_code == 200 for r in responses)
    
    def test_memory_usage(self):
        """Test memory usage during data processing."""
        with self.perf_tester.monitor_memory() as monitor:
            large_list = []
            for i in range(100000):
                large_list.append({"id": i, "data": f"item_{i}"})
            
            # Process in batches to test memory efficiency
            results = process_in_batches(large_list, batch_size=1000)
        
        # Memory should not exceed 100MB
        assert monitor.peak_memory_mb < 100
        assert monitor.memory_leaked_mb < 5  # Minimal memory leaks
    
    @pytest.mark.slow
    def test_concurrent_performance(self):
        """Test performance under concurrent load."""
        import asyncio
        import aiohttp
        
        async def make_request(session, url):
            async with session.get(url) as response:
                return await response.json()
        
        async def load_test():
            async with aiohttp.ClientSession() as session:
                tasks = [
                    make_request(session, "http://localhost:8000/api/users")
                    for _ in range(100)  # 100 concurrent requests
                ]
                
                start_time = time.time()
                results = await asyncio.gather(*tasks)
                total_time = time.time() - start_time
        
                return results, total_time
        
        # Run load test
        results, duration = asyncio.run(load_test())
        
        # Verify results
        assert len(results) == 100
        assert all(isinstance(result, dict) for result in results)
        
        # Performance requirements
        assert duration < 5.0  # All requests should complete within 5 seconds
        throughput = len(results) / duration
        assert throughput > 20  # At least 20 requests per second

#### Custom Test Decorators
def retry_on_failure(max_attempts=3, delay=1.0):
    """Decorator to retry flaky tests."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
                        print(f"Test failed, retrying... (attempt {attempt + 2}/{max_attempts})")
                    
            raise last_exception
        return wrapper
    return decorator

def requires_external_service(service_name):
    """Skip test if external service is not available."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not is_service_available(service_name):
                pytest.skip(f"External service {service_name} not available")
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Usage examples
@retry_on_failure(max_attempts=3, delay=2.0)
def test_flaky_network_operation():
    """Test that might fail due to network issues."""
    response = external_api_call()
    assert response.status_code == 200

@requires_external_service("elasticsearch")
def test_elasticsearch_integration():
    """Test that requires Elasticsearch to be running."""
    es_client = Elasticsearch(["localhost:9200"])
    result = es_client.search(index="test", body={"query": {"match_all": {}}})
    assert "hits" in result
```

## Complete Application Testing Example

```python
import pytest
import asyncio
from commons_testing import (
    AsyncTestCase, DataGenerator, fake,
    PostgresContainer, RedisContainer,
    APITestClient, MockServer
)
from datetime import datetime

class TestECommerceApplication(AsyncTestCase):
    """Complete integration test for e-commerce application."""
    
    @classmethod
    async def setUpClass(cls):
        """Set up test environment."""
        # Start database containers
        cls.postgres = PostgresContainer()
        cls.redis = RedisContainer()
        
        await cls.postgres.start()
        await cls.redis.start()
        
        # Initialize test data generator
        cls.data_generator = DataGenerator(seed=42)
        
        # Start application with test config
        cls.app = await start_test_application({
            "database_url": cls.postgres.get_connection_url(),
            "redis_url": cls.redis.get_connection_url(),
            "payment_service_url": "http://localhost:9001"
        })
        
        cls.api_client = APITestClient("http://localhost:8000")
    
    @classmethod
    async def tearDownClass(cls):
        """Clean up test environment."""
        await cls.app.stop()
        await cls.api_client.close()
        await cls.postgres.stop()
        await cls.redis.stop()
    
    async def setUp(self):
        """Set up each test."""
        # Clear database
        await self.postgres.execute("TRUNCATE users, products, orders CASCADE")
        
        # Clear Redis cache
        await self.redis.flushdb()
        
        # Create test user
        self.test_user = await self.create_test_user()
        await self.api_client.authenticate(self.test_user["email"], "testpass123")
    
    async def create_test_user(self):
        """Create test user."""
        user_data = {
            "name": fake.name(),
            "email": fake.email(),
            "password": "testpass123",
            "address": {
                "street": fake.street_address(),
                "city": fake.city(),
                "postal_code": fake.postcode(),
                "country": "US"
            }
        }
        
        response = await self.api_client.post("/auth/register", json=user_data)
        self.assertEqual(response.status_code, 201)
        return response.json()
    
    async def test_complete_purchase_flow(self):
        """Test complete purchase flow from product creation to order fulfillment."""
        # 1. Create products
        products = []
        for _ in range(3):
            product_data = {
                "name": fake.catch_phrase(),
                "description": fake.text(max_nb_chars=200),
                "price": round(fake.random.uniform(10.0, 100.0), 2),
                "stock_quantity": fake.random_int(10, 100),
                "category": fake.random_element(["Electronics", "Books", "Clothing"])
            }
            
            response = await self.api_client.post("/products", json=product_data)
            self.assertEqual(response.status_code, 201)
            products.append(response.json())
        
        # 2. Add products to cart
        cart_items = []
        for product in products[:2]:  # Add first 2 products
            item_data = {
                "product_id": product["id"],
                "quantity": fake.random_int(1, 3)
            }
            
            response = await self.api_client.post("/cart/items", json=item_data)
            self.assertEqual(response.status_code, 201)
            cart_items.append(response.json())
        
        # 3. Get cart total
        response = await self.api_client.get("/cart")
        self.assertEqual(response.status_code, 200)
        
        cart = response.json()
        self.assertEqual(len(cart["items"]), 2)
        self.assertGreater(cart["total"], 0)
        
        # 4. Create order
        order_data = {
            "shipping_address": self.test_user["address"],
            "payment_method": "credit_card"
        }
        
        response = await self.api_client.post("/orders", json=order_data)
        self.assertEqual(response.status_code, 201)
        
        order = response.json()
        self.assertEqual(order["status"], "pending")
        self.assertEqual(len(order["items"]), 2)
        
        # 5. Process payment (with mock payment service)
        with MockServer(port=9001) as payment_mock:
            payment_mock.expect(
                method="POST",
                path="/payments",
                response={
                    "payment_id": "pay_123456",
                    "status": "completed",
                    "amount": order["total"]
                },
                status_code=200
            )
            
            payment_data = {
                "order_id": order["id"],
                "amount": order["total"],
                "card_token": "tok_test_123"
            }
            
            response = await self.api_client.post(
                f"/orders/{order['id']}/payment", 
                json=payment_data
            )
            self.assertEqual(response.status_code, 200)
            
            payment_result = response.json()
            self.assertEqual(payment_result["status"], "completed")
        
        # 6. Verify order status updated
        response = await self.api_client.get(f"/orders/{order['id']}")
        updated_order = response.json()
        self.assertEqual(updated_order["status"], "paid")
        self.assertIsNotNone(updated_order["payment_id"])
        
        # 7. Verify inventory updated
        for i, item in enumerate(cart_items):
            response = await self.api_client.get(f"/products/{item['product_id']}")
            product = response.json()
            
            original_stock = products[i]["stock_quantity"]
            expected_stock = original_stock - item["quantity"]
            self.assertEqual(product["stock_quantity"], expected_stock)
        
        # 8. Verify cart cleared
        response = await self.api_client.get("/cart")
        cart = response.json()
        self.assertEqual(len(cart["items"]), 0)
    
    async def test_concurrent_order_processing(self):
        """Test handling of concurrent orders."""
        # Create product with limited stock
        product_data = {
            "name": "Limited Edition Item",
            "price": 99.99,
            "stock_quantity": 5  # Only 5 items available
        }
        
        response = await self.api_client.post("/products", json=product_data)
        product = response.json()
        
        # Create multiple users
        users = []
        for _ in range(10):
            user = await self.create_test_user()
            users.append(user)
        
        # Simulate concurrent orders
        async def place_order(user):
            client = APITestClient("http://localhost:8000")
            await client.authenticate(user["email"], "testpass123")
            
            # Add item to cart
            await client.post("/cart/items", json={
                "product_id": product["id"],
                "quantity": 2
            })
            
            # Place order
            response = await client.post("/orders", json={
                "shipping_address": user["address"],
                "payment_method": "credit_card"
            })
            
            await client.close()
            return response
        
        # Execute concurrent orders
        tasks = [place_order(user) for user in users]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful orders
        successful_orders = [
            r for r in responses 
            if not isinstance(r, Exception) and r.status_code == 201
        ]
        
        # Should have at most 2 successful orders (5 items / 2 per order)
        self.assertLessEqual(len(successful_orders), 2)
        
        # Verify final stock is correct
        response = await self.api_client.get(f"/products/{product['id']}")
        final_product = response.json()
        
        items_sold = len(successful_orders) * 2
        expected_stock = 5 - items_sold
        self.assertEqual(final_product["stock_quantity"], expected_stock)
    
    async def test_order_cancellation_and_refund(self):
        """Test order cancellation and refund process."""
        # Create and place order (reusing helper methods)
        order = await self._create_paid_order()
        
        # Cancel order
        response = await self.api_client.post(
            f"/orders/{order['id']}/cancel",
            json={"reason": "Customer request"}
        )
        self.assertEqual(response.status_code, 200)
        
        # Verify order status
        response = await self.api_client.get(f"/orders/{order['id']}")
        cancelled_order = response.json()
        self.assertEqual(cancelled_order["status"], "cancelled")
        
        # Verify refund was processed
        self.assertIsNotNone(cancelled_order["refund_id"])
        self.assertEqual(cancelled_order["refund_amount"], order["total"])
        
        # Verify inventory restored
        for item in order["items"]:
            response = await self.api_client.get(f"/products/{item['product_id']}")
            product = response.json()
            # Stock should be restored (simplified check)
            self.assertGreater(product["stock_quantity"], 0)
    
    async def _create_paid_order(self):
        """Helper method to create a paid order."""
        # Create product
        product_data = {
            "name": "Test Product",
            "price": 29.99,
            "stock_quantity": 10
        }
        
        response = await self.api_client.post("/products", json=product_data)
        product = response.json()
        
        # Add to cart and create order
        await self.api_client.post("/cart/items", json={
            "product_id": product["id"],
            "quantity": 1
        })
        
        response = await self.api_client.post("/orders", json={
            "shipping_address": self.test_user["address"],
            "payment_method": "credit_card"
        })
        
        order = response.json()
        
        # Process payment
        with MockServer(port=9001) as payment_mock:
            payment_mock.expect(
                method="POST",
                path="/payments",
                response={"payment_id": "pay_test", "status": "completed"},
                status_code=200
            )
            
            await self.api_client.post(
                f"/orders/{order['id']}/payment",
                json={"amount": order["total"], "card_token": "tok_test"}
            )
        
        # Return updated order
        response = await self.api_client.get(f"/orders/{order['id']}")
        return response.json()

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
```

## Fixtures Reference

```python
# Available fixtures from commons-testing
from commons_testing import fixtures

# Async fixtures
async_client      # Async HTTP client with retry and auth support
async_db          # Async database connection with transaction support
async_redis       # Async Redis client with cleanup
async_event_loop  # Custom event loop with proper cleanup
async_session     # Async SQLAlchemy session

# Database fixtures
postgres_db       # PostgreSQL test database with migrations
mysql_db          # MySQL test database with cleanup
mongodb           # MongoDB test database with collections
redis_cache       # Redis test instance with namespace isolation
sqlite_db         # SQLite in-memory database

# Container fixtures
postgres_container    # PostgreSQL container with custom config
mysql_container      # MySQL container with custom config
mongo_container      # MongoDB container with replica set
redis_container      # Redis container with persistence disabled
kafka_container      # Kafka container with Zookeeper
rabbitmq_container   # RabbitMQ container with management UI
elasticsearch_container  # Elasticsearch container with plugins

# Service fixtures
mock_server       # HTTP mock server with expectation matching
api_client        # Test API client with authentication
kafka_producer    # Kafka producer for message testing
kafka_consumer    # Kafka consumer with automatic cleanup
rabbitmq_channel  # RabbitMQ channel with queue management

# Cloud service fixtures
mock_aws_services     # Mocked AWS services (S3, DynamoDB, SQS, etc.)
mock_gcp_services     # Mocked GCP services (GCS, BigQuery, etc.)
localstack_container  # LocalStack container for AWS emulation

# Utility fixtures
temp_dir          # Temporary directory with automatic cleanup
temp_file         # Temporary file with custom content
faker             # Faker instance with consistent seed
time_machine      # Time manipulation for testing time-dependent code
data_generator    # Comprehensive test data generator
performance_tester # Performance monitoring and benchmarking
snapshot_tester   # Snapshot testing for regression detection

# Authentication fixtures
auth_token        # JWT token for API authentication
test_user         # Test user with known credentials
admin_user        # Admin user with elevated permissions

# File system fixtures
test_files        # Directory with sample test files
config_file       # Temporary configuration file
log_capture       # Log message capture and assertion
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests with coverage
pytest --cov=commons_testing --cov-report=html

# Run specific test categories
pytest -m "not slow"           # Skip slow tests
pytest -m "integration"        # Run only integration tests
pytest -k "database"           # Run database-related tests
pytest -k "async"              # Run async tests

# Run tests with different verbosity
pytest -v                      # Verbose output
pytest -s                      # Show print statements
pytest --tb=short             # Short traceback format

# Performance testing
pytest -m "performance" --benchmark-only

# Generate test report
pytest --html=report.html --self-contained-html
```