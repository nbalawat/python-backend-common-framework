"""
User Events Batch Processing Pipeline.

This pipeline processes user event data from GCS and writes to BigTable,
demonstrating best practices for batch processing with:
- Configurable data validation
- Error handling and dead letter processing  
- Efficient BigTable writing
- Comprehensive monitoring and logging
"""

__version__ = "0.1.0"