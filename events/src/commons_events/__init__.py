"""Commons Events - Event-driven architecture abstractions."""

from .abstractions import Event, EventMetadata, EventSchema
from .abstractions.producer import EventProducer, ProducerConfig
from .abstractions.consumer import EventConsumer, ConsumerConfig, ConsumerGroup

__version__ = "0.1.0"

__all__ = [
    "Event",
    "EventMetadata", 
    "EventSchema",
    "EventProducer",
    "ProducerConfig",
    "EventConsumer",
    "ConsumerConfig",
    "ConsumerGroup",
]
