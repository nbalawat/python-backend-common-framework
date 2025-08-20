"""
Common utilities for Google Dataflow pipelines.

This package provides reusable components for both batch and streaming
Dataflow pipelines, including:

- GCS and Pub/Sub I/O transforms
- BigTable and BigQuery writers
- Data validation and quality checks
- Configuration management
- Testing utilities
- Monitoring and observability
"""

__version__ = "0.1.0"

# Lazy imports to avoid import errors for optional dependencies
def __getattr__(name):
    if name in ['GCSReader', 'BigTableWriter', 'BigQueryWriter', 'DataValidator', 'ErrorHandler']:
        from .transforms import GCSReader, BigTableWriter, BigQueryWriter, DataValidator, ErrorHandler
        globals().update({
            'GCSReader': GCSReader,
            'BigTableWriter': BigTableWriter, 
            'BigQueryWriter': BigQueryWriter,
            'DataValidator': DataValidator,
            'ErrorHandler': ErrorHandler
        })
        return globals()[name]
    
    elif name in ['PubSubReader', 'StreamingBigTableWriter', 'MessageParser', 'DeadLetterHandler']:
        from .streaming import PubSubReader, StreamingBigTableWriter, MessageParser, DeadLetterHandler
        globals().update({
            'PubSubReader': PubSubReader,
            'StreamingBigTableWriter': StreamingBigTableWriter,
            'MessageParser': MessageParser,
            'DeadLetterHandler': DeadLetterHandler
        })
        return globals()[name]
    
    elif name in ['WindowingStrategies', 'TriggerStrategies', 'create_fixed_windows', 'create_sliding_windows', 'create_session_windows']:
        from .windowing import WindowingStrategies, TriggerStrategies, create_fixed_windows, create_sliding_windows, create_session_windows
        globals().update({
            'WindowingStrategies': WindowingStrategies,
            'TriggerStrategies': TriggerStrategies,
            'create_fixed_windows': create_fixed_windows,
            'create_sliding_windows': create_sliding_windows,
            'create_session_windows': create_session_windows
        })
        return globals()[name]
    
    elif name in ['PipelineConfig', 'BatchConfig', 'StreamingConfig', 'get_pipeline_config']:
        from .config import PipelineConfig, BatchConfig, StreamingConfig, get_pipeline_config
        globals().update({
            'PipelineConfig': PipelineConfig,
            'BatchConfig': BatchConfig,
            'StreamingConfig': StreamingConfig,
            'get_pipeline_config': get_pipeline_config
        })
        return globals()[name]
    
    elif name in ['DataflowTestCase', 'MockSource', 'MockSink', 'create_test_data']:
        from .testing import DataflowTestCase, MockSource, MockSink, create_test_data
        globals().update({
            'DataflowTestCase': DataflowTestCase,
            'MockSource': MockSource,
            'MockSink': MockSink,
            'create_test_data': create_test_data
        })
        return globals()[name]
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # Core transforms
    "GCSReader",
    "BigTableWriter", 
    "BigQueryWriter",
    "DataValidator",
    "ErrorHandler",
    # Streaming
    "PubSubReader",
    "StreamingBigTableWriter",
    "MessageParser", 
    "DeadLetterHandler",
    # Windowing
    "WindowingStrategies",
    "TriggerStrategies",
    "create_fixed_windows",
    "create_sliding_windows", 
    "create_session_windows",
    # Configuration
    "PipelineConfig",
    "BatchConfig",
    "StreamingConfig",
    "get_pipeline_config",
    # Testing
    "DataflowTestCase",
    "MockSource",
    "MockSink",
    "create_test_data",
]