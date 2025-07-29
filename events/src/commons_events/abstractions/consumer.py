"""Event consumer abstraction."""

from typing import Any, Dict, Optional, List
from .base import Event

class ConsumerConfig:
    """Consumer configuration."""
    
    def __init__(self, **kwargs):
        self.config = kwargs

class ConsumerGroup:
    """Consumer group."""
    
    def __init__(self, group_id: str):
        self.group_id = group_id

class EventConsumer:
    """Event consumer interface."""
    
    def __init__(self, config: ConsumerConfig, group: ConsumerGroup = None):
        self.config = config
        self.group = group
    
    async def consume(self, topics: List[str]) -> List[Event]:
        """Consume events."""
        return []
