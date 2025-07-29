"""Pipeline sink abstraction."""

from typing import Any, Dict, Optional
from enum import Enum

class SinkType(Enum):
    """Sink types."""
    FILE = "file" 
    DATABASE = "database"
    STREAM = "stream"

class SinkOptions:
    """Sink options."""
    
    def __init__(self, **kwargs):
        self.options = kwargs

class Sink:
    """Pipeline sink."""
    
    def __init__(self, sink_type: SinkType, options: SinkOptions):
        self.sink_type = sink_type
        self.options = options
