"""Pipeline source abstraction."""

from typing import Any, Dict, Optional
from enum import Enum

class SourceType(Enum):
    """Source types."""
    FILE = "file"
    DATABASE = "database"
    STREAM = "stream"

class SourceOptions:
    """Source options."""
    
    def __init__(self, **kwargs):
        self.options = kwargs

class Source:
    """Pipeline source."""
    
    def __init__(self, source_type: SourceType, options: SourceOptions):
        self.source_type = source_type
        self.options = options
