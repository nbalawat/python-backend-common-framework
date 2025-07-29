"""Commons Pipelines - Data pipeline abstractions."""

from .abstractions import StreamingPipeline, Schema, DataType
from .abstractions.source import Source, SourceType, SourceOptions
from .abstractions.sink import Sink, SinkType, SinkOptions
from .simple_pipeline import SimplePipeline as Pipeline

__version__ = "0.1.0"

__all__ = [
    "Pipeline",
    "StreamingPipeline",
    "Schema",
    "DataType", 
    "Source",
    "SourceType",
    "SourceOptions",
    "Sink",
    "SinkType",
    "SinkOptions",
]
