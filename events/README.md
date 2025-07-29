# Commons Events

Event-driven architecture components for building scalable distributed systems.

## Features

- **Event Model**: Flexible event schemas with Avro, JSON Schema, and Protobuf support
- **Producers**: Kafka, Google Pub/Sub, RabbitMQ, AWS SNS producers
- **Consumers**: At-least-once delivery with consumer groups and dead letter queues
- **Stream Processing**: Window operations, aggregations, and stateful processing
- **Schema Registry**: Centralized schema management with evolution support

## Installation

```bash
# Basic installation
pip install commons-events

# With specific message brokers
pip install commons-events[kafka]
pip install commons-events[rabbitmq]
pip install commons-events[pubsub]
pip install commons-events[sns]

# With stream processing
pip install commons-events[streaming]
```

## Usage

### Event Model

```python
from commons_events import Event, EventSchema, EventMetadata

# Define event schema
@EventSchema(
    name="user.created",
    version="1.0.0",
    schema_type="avro",
)
class UserCreatedEvent(Event):
    user_id: str
    email: str
    created_at: datetime
    metadata: EventMetadata

# Create event
event = UserCreatedEvent(
    user_id="123",
    email="user@example.com",
    created_at=datetime.utcnow(),
    metadata=EventMetadata(
        event_id="evt_123",
        correlation_id="req_456",
        causation_id="cmd_789",
    )
)

# Serialize event
data = event.serialize()  # Avro bytes
json_data = event.to_json()  # JSON string

# Deserialize event
restored = UserCreatedEvent.deserialize(data)
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