# Commons Events

Comprehensive event-driven architecture framework for building scalable, resilient distributed systems with support for multiple message brokers, stream processing, and event sourcing patterns.

## Installation

```bash
pip install commons-events
```

## Features

- **Multi-Broker Support**: Unified API for Kafka, RabbitMQ, Google Pub/Sub, AWS SNS/SQS
- **Event Modeling**: Type-safe events with Avro, JSON Schema, and Protobuf serialization  
- **Stream Processing**: Real-time data processing with windowing and aggregations
- **Event Sourcing**: Complete event sourcing implementation with aggregates
- **Schema Registry**: Centralized schema management with backward compatibility
- **Dead Letter Queues**: Robust error handling and retry mechanisms
- **Testing Support**: Comprehensive testing utilities and test containers
- **Monitoring**: Built-in metrics, tracing, and health checks
- **Security**: End-to-end encryption and authentication support

## Quick Start

```python
import asyncio
from commons_events import Event, EventProducer, ProducerConfig
from datetime import datetime

# Define an event
class UserRegistered(Event):
    event_type = "user.registered"
    user_id: str
    email: str
    registered_at: datetime

async def main():
    # Create event
    event = UserRegistered(
        user_id="user_123",
        email="john@example.com",
        registered_at=datetime.now(),
        source="auth-service"
    )
    
    # Send event
    config = ProducerConfig(broker="memory")  # For demo
    producer = EventProducer(config)
    
    await producer.send("user-events", event)
    print(f"Sent event: {event.event_type}")
    
    await producer.close()

asyncio.run(main())
```

## Detailed Usage Examples

### Event Modeling and Schemas

#### Type-Safe Event Definition
```python
import asyncio
from commons_events import Event, EventMetadata, EventSchema
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from enum import Enum
import uuid

# Define event schemas with validation
class EventStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class UserRole(str, Enum):
    USER = "user"
    ADMIN = "admin"
    MODERATOR = "moderator"

@EventSchema(
    name="user.registered",
    version="1.2.0",
    schema_type="avro",
    description="User registration event with profile information"
)
class UserRegisteredEvent(Event):
    """Event emitted when a new user registers."""
    
    # Required fields
    user_id: str
    email: str
    username: str
    registered_at: datetime
    
    # Optional fields with defaults
    full_name: Optional[str] = None
    phone_number: Optional[str] = None
    role: UserRole = UserRole.USER
    email_verified: bool = False
    terms_accepted: bool = True
    
    # Complex nested data
    profile: Dict[str, Any] = {}
    preferences: Dict[str, Any] = {
        "notifications": True,
        "theme": "light",
        "language": "en"
    }
    
    # List fields
    tags: List[str] = []
    
    def __post_init__(self):
        """Validate event data after initialization."""
        super().__post_init__()
        
        # Custom validation
        if "@" not in self.email:
            raise ValueError("Invalid email format")
        
        if len(self.username) < 3:
            raise ValueError("Username too short")
        
        # Auto-generate event ID if not provided
        if not self.event_id:
            self.event_id = f"evt_{uuid.uuid4().hex[:12]}"
        
        # Set source if not provided
        if not self.source:
            self.source = "user-service"

@EventSchema(
    name="user.profile.updated",
    version="1.0.0",
    schema_type="json"
)
class UserProfileUpdatedEvent(Event):
    """Event for profile updates with change tracking."""
    
    user_id: str
    updated_at: datetime
    updated_fields: List[str]  # Fields that were changed
    old_values: Dict[str, Any]  # Previous values
    new_values: Dict[str, Any]  # New values
    updated_by: str  # Who made the change
    
    @classmethod
    def create_from_changes(
        cls,
        user_id: str,
        changes: Dict[str, tuple],  # field -> (old_value, new_value)
        updated_by: str
    ) -> 'UserProfileUpdatedEvent':
        """Factory method to create event from field changes."""
        
        updated_fields = list(changes.keys())
        old_values = {field: old for field, (old, new) in changes.items()}
        new_values = {field: new for field, (old, new) in changes.items()}
        
        return cls(
            user_id=user_id,
            updated_at=datetime.now(timezone.utc),
            updated_fields=updated_fields,
            old_values=old_values,
            new_values=new_values,
            updated_by=updated_by,
            source="user-service"
        )

@EventSchema(
    name="order.lifecycle",
    version="2.1.0",
    schema_type="protobuf"
)
class OrderLifecycleEvent(Event):
    """Complex event for order state transitions."""
    
    order_id: str
    user_id: str
    status: EventStatus
    previous_status: Optional[EventStatus] = None
    
    # Order details
    items: List[Dict[str, Any]]
    total_amount: float
    currency: str = "USD"
    
    # Timestamps
    status_changed_at: datetime
    
    # Additional context
    reason: Optional[str] = None
    metadata: Dict[str, Any] = {}
    
    # Delivery information
    shipping_address: Optional[Dict[str, str]] = None
    estimated_delivery: Optional[datetime] = None
    
    def calculate_processing_time(self) -> Optional[float]:
        """Calculate time since order was placed."""
        if self.previous_status == EventStatus.PENDING:
            created_time = self.metadata.get("order_created_at")
            if created_time:
                created_dt = datetime.fromisoformat(created_time)
                return (self.status_changed_at - created_dt).total_seconds()
        return None

# Demonstrate event creation and serialization
async def demonstrate_event_modeling():
    """Show comprehensive event modeling patterns."""
    
    print("=== Event Modeling and Schemas ===")
    
    # 1. Create user registration event
    registration_event = UserRegisteredEvent(
        user_id="user_12345",
        email="alice@example.com",
        username="alice_wonder",
        registered_at=datetime.now(timezone.utc),
        full_name="Alice Wonderland",
        phone_number="+1-555-0123",
        role=UserRole.USER,
        profile={
            "age": 28,
            "location": "San Francisco, CA",
            "bio": "Software engineer passionate about distributed systems",
            "avatar_url": "https://example.com/avatars/alice.jpg"
        },
        preferences={
            "notifications": True,
            "theme": "dark",
            "language": "en",
            "timezone": "America/Los_Angeles"
        },
        tags=["engineer", "distributed-systems", "python"]
    )
    
    print(f"Created registration event:")
    print(f"  Event ID: {registration_event.event_id}")
    print(f"  User: {registration_event.username} ({registration_event.email})")
    print(f"  Role: {registration_event.role}")
    print(f"  Tags: {registration_event.tags}")
    
    # 2. Serialization examples
    print("\nSerialization formats:")
    
    # JSON serialization
    json_data = registration_event.to_json()
    print(f"  JSON size: {len(json_data)} bytes")
    
    # Avro serialization (more compact)
    try:
        avro_data = registration_event.serialize()
        print(f"  Avro size: {len(avro_data)} bytes")
        
        # Verify round-trip serialization
        restored_event = UserRegisteredEvent.deserialize(avro_data)
        assert restored_event.user_id == registration_event.user_id
        assert restored_event.email == registration_event.email
        print("  ✓ Avro round-trip serialization successful")
        
    except Exception as e:
        print(f"  ⚠ Avro serialization failed: {e}")
    
    # 3. Profile update event with change tracking
    profile_changes = {
        "full_name": ("Alice Wonderland", "Alice W. Wonderland"),
        "bio": (
            "Software engineer passionate about distributed systems",
            "Senior software engineer specializing in event-driven architectures"
        ),
        "location": ("San Francisco, CA", "Seattle, WA")
    }
    
    profile_event = UserProfileUpdatedEvent.create_from_changes(
        user_id="user_12345",
        changes=profile_changes,
        updated_by="user_12345"  # Self-update
    )
    
    print(f"\nProfile update event:")
    print(f"  Updated fields: {profile_event.updated_fields}")
    print(f"  Changes:")
    for field in profile_event.updated_fields:
        old_val = profile_event.old_values[field]
        new_val = profile_event.new_values[field]
        print(f"    {field}: '{old_val}' → '{new_val}'")
    
    # 4. Complex order lifecycle event
    order_event = OrderLifecycleEvent(
        order_id="order_789",
        user_id="user_12345",
        status=EventStatus.PROCESSING,
        previous_status=EventStatus.PENDING,
        items=[
            {
                "product_id": "prod_123",
                "name": "Wireless Headphones",
                "quantity": 1,
                "price": 299.99
            },
            {
                "product_id": "prod_456",
                "name": "USB-C Cable",
                "quantity": 2,
                "price": 19.99
            }
        ],
        total_amount=339.97,
        currency="USD",
        status_changed_at=datetime.now(timezone.utc),
        reason="Payment confirmed, preparing for shipment",
        shipping_address={
            "street": "123 Main St",
            "city": "Seattle",
            "state": "WA",
            "zip": "98101",
            "country": "US"
        },
        estimated_delivery=datetime.now(timezone.utc) + timedelta(days=3),
        metadata={
            "order_created_at": (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat(),
            "payment_method": "credit_card",
            "shipping_method": "standard",
            "warehouse_id": "wh_west_01"
        },
        source="order-service"
    )
    
    print(f"\nOrder lifecycle event:")
    print(f"  Order: {order_event.order_id}")
    print(f"  Status: {order_event.previous_status} → {order_event.status}")
    print(f"  Items: {len(order_event.items)}")
    print(f"  Total: ${order_event.total_amount}")
    print(f"  Estimated delivery: {order_event.estimated_delivery}")
    
    processing_time = order_event.calculate_processing_time()
    if processing_time:
        print(f"  Processing time: {processing_time:.1f} seconds")
    
    # 5. Event validation examples
    print("\nEvent validation:")
    
    # Valid event
    try:
        valid_event = UserRegisteredEvent(
            user_id="user_999",
            email="valid@example.com",
            username="valid_user",
            registered_at=datetime.now(timezone.utc)
        )
        print("  ✓ Valid event created successfully")
    except Exception as e:
        print(f"  ⚠ Unexpected validation error: {e}")
    
    # Invalid email
    try:
        invalid_event = UserRegisteredEvent(
            user_id="user_998",
            email="invalid-email",  # Missing @
            username="test_user",
            registered_at=datetime.now(timezone.utc)
        )
        print("  ⚠ Invalid email validation failed")
    except ValueError as e:
        print(f"  ✓ Invalid email caught: {e}")
    
    # Invalid username
    try:
        invalid_event = UserRegisteredEvent(
            user_id="user_997",
            email="test@example.com",
            username="ab",  # Too short
            registered_at=datetime.now(timezone.utc)
        )
        print("  ⚠ Invalid username validation failed")
    except ValueError as e:
        print(f"  ✓ Invalid username caught: {e}")
    
    # 6. Event metadata and tracing
    print("\nEvent metadata and tracing:")
    
    # Create event with full tracing context
    traced_event = UserRegisteredEvent(
        user_id="user_traced",
        email="traced@example.com",
        username="traced_user",
        registered_at=datetime.now(timezone.utc),
        event_id="evt_traced_123",
        correlation_id="req_abc123",  # Request that started this flow
        causation_id="cmd_register_456",  # Command that caused this event
        source="auth-service",
        version="1.2.0",
        metadata={
            "user_agent": "Mozilla/5.0 (compatible)",
            "ip_address": "192.168.1.100",
            "referrer": "https://example.com/signup",
            "campaign": "spring_2024",
            "ab_test_group": "variation_b"
        }
    )
    
    print(f"  Event ID: {traced_event.event_id}")
    print(f"  Correlation ID: {traced_event.correlation_id}")
    print(f"  Causation ID: {traced_event.causation_id}")
    print(f"  Source: {traced_event.source}")
    print(f"  Additional metadata: {len(traced_event.metadata)} fields")

# Schema evolution examples
async def demonstrate_schema_evolution():
    """Demonstrate schema evolution and backward compatibility."""
    
    print("\n=== Schema Evolution and Compatibility ===")
    
    # Original schema (v1.0.0)
    @EventSchema(
        name="product.created",
        version="1.0.0",
        schema_type="avro"
    )
    class ProductCreatedV1(Event):
        product_id: str
        name: str
        price: float
        created_at: datetime
    
    # Evolved schema (v1.1.0) - backward compatible
    @EventSchema(
        name="product.created",
        version="1.1.0",
        schema_type="avro"
    )
    class ProductCreatedV1_1(Event):
        product_id: str
        name: str
        price: float
        created_at: datetime
        # New optional fields with defaults
        category: Optional[str] = None
        description: Optional[str] = None
        tags: List[str] = []
    
    # Further evolved schema (v2.0.0) - breaking changes
    @EventSchema(
        name="product.created",
        version="2.0.0",
        schema_type="avro"
    )
    class ProductCreatedV2(Event):
        product_id: str
        name: str
        # Changed: price is now a structured object
        pricing: Dict[str, Any]  # {"amount": 99.99, "currency": "USD"}
        created_at: datetime
        category: str  # Now required
        description: Optional[str] = None
        tags: List[str] = []
        # New fields
        sku: str
        inventory_count: int = 0
    
    # Create events with different schema versions
    v1_event = ProductCreatedV1(
        product_id="prod_001",
        name="Wireless Mouse",
        price=49.99,
        created_at=datetime.now(timezone.utc),
        source="catalog-service"
    )
    
    v1_1_event = ProductCreatedV1_1(
        product_id="prod_002",
        name="Mechanical Keyboard",
        price=129.99,
        created_at=datetime.now(timezone.utc),
        category="Electronics",
        description="RGB mechanical gaming keyboard",
        tags=["gaming", "rgb", "mechanical"],
        source="catalog-service"
    )
    
    v2_event = ProductCreatedV2(
        product_id="prod_003",
        name="USB-C Hub",
        pricing={"amount": 79.99, "currency": "USD"},
        created_at=datetime.now(timezone.utc),
        category="Accessories",
        description="7-in-1 USB-C hub with HDMI",
        tags=["usb-c", "hub", "hdmi"],
        sku="HUB-USBC-001",
        inventory_count=50,
        source="catalog-service"
    )
    
    print(f"Schema versions:")
    print(f"  v1.0.0: {v1_event.name} - ${v1_event.price}")
    print(f"  v1.1.0: {v1_1_event.name} - ${v1_1_event.price} ({v1_1_event.category})")
    print(f"  v2.0.0: {v2_event.name} - ${v2_event.pricing['amount']} ({v2_event.category})")
    
    # Demonstrate schema migration
    def migrate_v1_to_v2(v1_event: ProductCreatedV1) -> ProductCreatedV2:
        """Migrate v1 event to v2 format."""
        return ProductCreatedV2(
            product_id=v1_event.product_id,
            name=v1_event.name,
            pricing={
                "amount": v1_event.price,
                "currency": "USD"  # Default assumption
            },
            created_at=v1_event.created_at,
            category="General",  # Default category
            sku=f"SKU-{v1_event.product_id}",
            source=v1_event.source,
            event_id=v1_event.event_id,
            correlation_id=v1_event.correlation_id
        )
    
    migrated_event = migrate_v1_to_v2(v1_event)
    print(f"\nMigrated v1 → v2:")
    print(f"  Original: {v1_event.name} - ${v1_event.price}")
    print(f"  Migrated: {migrated_event.name} - ${migrated_event.pricing['amount']} ({migrated_event.category})")

# Run demonstrations
asyncio.run(demonstrate_event_modeling())
asyncio.run(demonstrate_schema_evolution())
```

### Producers

```python
from commons_events.producers import KafkaProducer, PubSubProducer, RabbitMQProducer

# Kafka producer
async with KafkaProducer(bootstrap_servers="localhost:9092") as producer:
    await producer.send(
        topic="user-events",
        event=event,
        key=event.user_id,
        partition=None,  # Auto-partition by key
    )
    
    # Batch send
    events = [event1, event2, event3]
    await producer.send_batch(topic="user-events", events=events)

# Google Pub/Sub producer
async with PubSubProducer(project_id="my-project") as producer:
    message_id = await producer.publish(
        topic="user-events",
        event=event,
        ordering_key=event.user_id,  # Message ordering
    )

# RabbitMQ producer
async with RabbitMQProducer(url="amqp://localhost") as producer:
    await producer.publish(
        exchange="events",
        routing_key="user.created",
        event=event,
        persistent=True,
    )

# AWS SNS producer
async with SNSProducer(region="us-east-1") as producer:
    message_id = await producer.publish(
        topic_arn="arn:aws:sns:us-east-1:123456789012:user-events",
        event=event,
        attributes={"event_type": "user.created"},
    )
```

### Consumers

```python
from commons_events.consumers import KafkaConsumer, PubSubConsumer, RabbitMQConsumer

# Kafka consumer
async with KafkaConsumer(
    bootstrap_servers="localhost:9092",
    group_id="my-service",
    topics=["user-events"],
) as consumer:
    async for message in consumer:
        event = UserCreatedEvent.deserialize(message.value)
        print(f"Received: {event.user_id}")
        
        # Acknowledge message
        await consumer.commit(message)

# Pub/Sub consumer with error handling
async with PubSubConsumer(
    project_id="my-project",
    subscription="user-events-sub",
    dead_letter_topic="user-events-dlq",
    max_delivery_attempts=5,
) as consumer:
    async for message in consumer:
        try:
            event = UserCreatedEvent.from_json(message.data)
            await process_event(event)
            await message.ack()
        except Exception as e:
            logger.error(f"Failed to process: {e}")
            await message.nack()  # Will retry or go to DLQ

# RabbitMQ consumer with prefetch
async with RabbitMQConsumer(
    url="amqp://localhost",
    queue="user-events",
    prefetch_count=10,
) as consumer:
    async for message in consumer:
        event = UserCreatedEvent.deserialize(message.body)
        await handle_event(event)
        await message.ack()
```

### Stream Processing

```python
from commons_events.streaming import StreamProcessor, Window, Aggregation

# Create stream processor
processor = StreamProcessor()

# Define processing pipeline
stream = processor.stream("user-events")

# Window operations
windowed = stream.window(
    Window.tumbling(minutes=5),
    group_by=lambda e: e.metadata.correlation_id,
)

# Aggregations
user_counts = windowed.aggregate(
    Aggregation.count(),
    Aggregation.distinct_count("user_id"),
)

# Joins
orders = processor.stream("order-events")
enriched = stream.join(
    orders,
    on=lambda user, order: user.user_id == order.user_id,
    window=Window.sliding(minutes=10),
)

# Output
enriched.sink("enriched-events")

# Run processor
await processor.run()
```

### Schema Registry

```python
from commons_events.schemas import SchemaRegistry, SchemaEvolution

# Initialize registry
registry = SchemaRegistry(url="http://localhost:8081")

# Register schema
schema_id = await registry.register(
    subject="user-events-value",
    schema=UserCreatedEvent.schema,
    schema_type="avro",
)

# Get latest schema
schema = await registry.get_latest("user-events-value")

# Check compatibility
is_compatible = await registry.check_compatibility(
    subject="user-events-value",
    schema=updated_schema,
)

# Schema evolution
evolution = SchemaEvolution(registry)

# Add field with default
new_schema = await evolution.add_field(
    schema=UserCreatedEvent.schema,
    field_name="phone",
    field_type="string",
    default=None,
)

# Remove field (if nullable)
new_schema = await evolution.remove_field(
    schema=UserCreatedEvent.schema,
    field_name="optional_field",
)
```

### Event Sourcing

```python
from commons_events.sourcing import EventStore, AggregateRoot, DomainEvent

# Define aggregate
class UserAggregate(AggregateRoot):
    def __init__(self, user_id: str):
        super().__init__(user_id)
        self.email = None
        self.active = False
    
    def create(self, email: str) -> None:
        self.apply(UserCreated(
            aggregate_id=self.id,
            email=email,
            timestamp=datetime.utcnow(),
        ))
    
    def activate(self) -> None:
        if self.active:
            raise ValueError("Already active")
        self.apply(UserActivated(
            aggregate_id=self.id,
            timestamp=datetime.utcnow(),
        ))
    
    def handle_user_created(self, event: UserCreated) -> None:
        self.email = event.email
    
    def handle_user_activated(self, event: UserActivated) -> None:
        self.active = True

# Use event store
store = EventStore(connection_string="postgresql://localhost/events")

# Create and save aggregate
user = UserAggregate("user_123")
user.create("user@example.com")
user.activate()

await store.save(user)

# Load aggregate from events
loaded = await store.load(UserAggregate, "user_123")
assert loaded.email == "user@example.com"
assert loaded.active is True

# Query events
events = await store.get_events(
    aggregate_id="user_123",
    from_version=0,
)
```

### Testing Support

```python
from commons_events.testing import EventTestHarness, MockProducer, MockConsumer

# Test event processing
harness = EventTestHarness()

# Test producer
producer = MockProducer()
await producer.send("topic", event)

# Verify sent events
assert len(producer.sent_events) == 1
assert producer.sent_events[0].topic == "topic"

# Test consumer
consumer = MockConsumer([event1, event2, event3])

processed = []
async for event in consumer:
    processed.append(event)
    
assert len(processed) == 3

# Integration testing
async with harness.kafka_container() as kafka:
    producer = KafkaProducer(bootstrap_servers=kafka.bootstrap_servers)
    await producer.send("test-topic", event)
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev,kafka,rabbitmq,pubsub,streaming]"

# Run tests
pytest

# Run integration tests
pytest -m integration
```