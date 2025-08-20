#!/usr/bin/env python3
"""Fix module imports to only include what exists."""

import os
from pathlib import Path

def update_file(filepath: str, content: str):
    """Update a file with given content."""
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Updated: {filepath}")

# Fix k8s __init__.py to only import what exists
update_file("k8s/src/commons_k8s/__init__.py", '''"""Commons K8s - Kubernetes abstractions."""

from .types import ResourceSpec, ResourceStatus, PodSpec
from .resources import Deployment, Service, Pod, ConfigMap, Secret

__version__ = "0.1.0"

__all__ = [
    "ResourceSpec",
    "ResourceStatus", 
    "PodSpec",
    "Deployment",
    "Service",
    "Pod", 
    "ConfigMap",
    "Secret",
]
''')

# Fix events __init__.py
update_file("events/src/commons_events/__init__.py", '''"""Commons Events - Event-driven architecture abstractions."""

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
''')

# Fix llm __init__.py
update_file("llm/src/commons_llm/__init__.py", '''"""Commons LLM - LLM provider abstractions."""

from .abstractions import (
    LLMProvider,
    Message,
    ChatRequest,
    ChatResponse,
    StreamResponse,
    TokenUsage,
)
from .abstractions.functions import Function, FunctionCall, FunctionParameter
from .abstractions.embeddings import EmbeddingProvider, EmbeddingResponse

__version__ = "0.1.0"

__all__ = [
    "LLMProvider",
    "Message",
    "ChatRequest", 
    "ChatResponse",
    "StreamResponse",
    "TokenUsage",
    "Function",
    "FunctionCall",
    "FunctionParameter",
    "EmbeddingProvider",
    "EmbeddingResponse",
]
''')

# Fix pipelines __init__.py
update_file("pipelines/src/commons_pipelines/__init__.py", '''"""Commons Pipelines - Data pipeline abstractions."""

from .abstractions import Pipeline, StreamingPipeline, Schema, DataType
from .abstractions.source import Source, SourceType, SourceOptions
from .abstractions.sink import Sink, SinkType, SinkOptions

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
''')

# Fix workflows __init__.py
update_file("workflows/src/commons_workflows/__init__.py", '''"""Commons Workflows - Business process workflow management."""

from .abstractions import (
    Workflow,
    WorkflowStep,
    WorkflowState,
    Activity,
)

__version__ = "0.1.0"

__all__ = [
    "Workflow",
    "WorkflowStep", 
    "WorkflowState",
    "Activity",
]
''')

# Fix agents __init__.py
update_file("agents/src/commons_agents/__init__.py", '''"""Commons Agents - Agent orchestration."""

from .memory import AgentMemory
from .tools import Tool

__version__ = "0.1.0"

__all__ = [
    "AgentMemory",
    "Tool",
]
''')

# Fix data __init__.py
update_file("data/src/commons_data/__init__.py", '''"""Commons Data - Database abstractions."""

from .abstractions import (
    DatabaseClient,
    Repository,
    Model,
)
from .abstractions.query import Query, QueryBuilder, Condition
from .abstractions.schema import Table, Column, Index, ForeignKey

__version__ = "0.1.0"

__all__ = [
    "DatabaseClient",
    "Repository",
    "Model",
    "Query",
    "QueryBuilder", 
    "Condition",
    "Table",
    "Column",
    "Index",
    "ForeignKey",
]
''')

print("All module imports fixed!")