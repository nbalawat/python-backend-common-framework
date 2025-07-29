"""Base pipeline abstractions."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, AsyncIterator
from dataclasses import dataclass
from datetime import datetime

from commons_core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    
    name: str
    engine: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    checkpoint_dir: Optional[str] = None
    enable_metrics: bool = True
    enable_lineage: bool = False


class DataFrame(ABC):
    """Abstract DataFrame interface."""
    
    def __init__(self, pipeline: "Pipeline", data: Any) -> None:
        self.pipeline = pipeline
        self._data = data
        
    @abstractmethod
    async def collect(self) -> List[Dict[str, Any]]:
        """Collect all rows as list of dictionaries."""
        pass
        
    @abstractmethod
    async def count(self) -> int:
        """Count number of rows."""
        pass
        
    @abstractmethod
    async def show(self, n: int = 20, truncate: bool = True) -> None:
        """Show first n rows."""
        pass
        
    @abstractmethod
    def select(self, *columns: Union[str, "Column"]) -> "DataFrame":
        """Select columns."""
        pass
        
    @abstractmethod
    def filter(self, condition: Union[str, "Column"]) -> "DataFrame":
        """Filter rows based on condition."""
        pass
        
    @abstractmethod
    def groupby(self, *columns: str) -> "GroupedDataFrame":
        """Group by columns."""
        pass
        
    @abstractmethod
    def join(
        self,
        other: "DataFrame",
        on: Union[str, List[str]],
        how: str = "inner",
        suffix: Optional[tuple[str, str]] = None,
    ) -> "DataFrame":
        """Join with another DataFrame."""
        pass
        
    @abstractmethod
    def union(self, other: "DataFrame") -> "DataFrame":
        """Union with another DataFrame."""
        pass
        
    @abstractmethod
    def distinct(self) -> "DataFrame":
        """Get distinct rows."""
        pass
        
    @abstractmethod
    def sort(self, *columns: str, ascending: bool = True) -> "DataFrame":
        """Sort by columns."""
        pass
        
    @abstractmethod
    def limit(self, n: int) -> "DataFrame":
        """Limit to n rows."""
        pass
        
    @abstractmethod
    def with_column(self, name: str, expr: "Column") -> "DataFrame":
        """Add or replace column."""
        pass
        
    @abstractmethod
    def drop(self, *columns: str) -> "DataFrame":
        """Drop columns."""
        pass
        
    @abstractmethod
    def rename(self, mapping: Dict[str, str]) -> "DataFrame":
        """Rename columns."""
        pass
        
    @abstractmethod
    def cache(self) -> "DataFrame":
        """Cache DataFrame in memory."""
        pass
        
    @abstractmethod
    def persist(self, storage_level: str = "memory") -> "DataFrame":
        """Persist DataFrame."""
        pass
        
    @abstractmethod
    async def write(self, sink: "Sink") -> None:
        """Write to sink."""
        pass


class GroupedDataFrame(ABC):
    """Grouped DataFrame for aggregations."""
    
    @abstractmethod
    def agg(self, *exprs: Union[Dict[str, str], "Column"]) -> DataFrame:
        """Aggregate grouped data."""
        pass
        
    @abstractmethod
    def count(self) -> DataFrame:
        """Count rows in each group."""
        pass
        
    @abstractmethod
    def sum(self, *columns: str) -> DataFrame:
        """Sum columns in each group."""
        pass
        
    @abstractmethod
    def avg(self, *columns: str) -> DataFrame:
        """Average columns in each group."""
        pass
        
    @abstractmethod
    def min(self, *columns: str) -> DataFrame:
        """Min of columns in each group."""
        pass
        
    @abstractmethod
    def max(self, *columns: str) -> DataFrame:
        """Max of columns in each group."""
        pass


class StreamingDataFrame(DataFrame):
    """Streaming DataFrame interface."""
    
    @abstractmethod
    def with_watermark(self, column: str, delay: str) -> "StreamingDataFrame":
        """Add watermark for late data handling."""
        pass
        
    @abstractmethod
    async def start(self) -> None:
        """Start streaming query."""
        pass
        
    @abstractmethod
    async def stop(self) -> None:
        """Stop streaming query."""
        pass
        
    @abstractmethod
    async def await_termination(self, timeout: Optional[float] = None) -> None:
        """Wait for streaming termination."""
        pass


class Pipeline(ABC):
    """Abstract pipeline interface."""
    
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._metrics = {}
        self._lineage = []
        
    @abstractmethod
    async def read(self, source: "Source") -> DataFrame:
        """Read from source."""
        pass
        
    @abstractmethod
    async def read_csv(
        self,
        path: str,
        schema: Optional["Schema"] = None,
        header: bool = True,
        delimiter: str = ",",
        **options: Any,
    ) -> DataFrame:
        """Read CSV files."""
        pass
        
    @abstractmethod
    async def read_json(
        self,
        path: str,
        schema: Optional["Schema"] = None,
        multiline: bool = False,
        **options: Any,
    ) -> DataFrame:
        """Read JSON files."""
        pass
        
    @abstractmethod
    async def read_parquet(
        self,
        path: str,
        columns: Optional[List[str]] = None,
        **options: Any,
    ) -> DataFrame:
        """Read Parquet files."""
        pass
        
    @abstractmethod
    async def read_sql(
        self,
        query: str,
        connection: "DatabaseConnector",
        partition_column: Optional[str] = None,
        num_partitions: Optional[int] = None,
        **options: Any,
    ) -> DataFrame:
        """Read from SQL database."""
        pass
        
    @abstractmethod
    async def read_table(
        self,
        table: str,
        database: Optional[str] = None,
        **options: Any,
    ) -> DataFrame:
        """Read from table."""
        pass
        
    @abstractmethod
    async def create_dataframe(
        self,
        data: List[Dict[str, Any]],
        schema: Optional["Schema"] = None,
    ) -> DataFrame:
        """Create DataFrame from data."""
        pass
        
    @abstractmethod
    async def sql(self, query: str) -> DataFrame:
        """Execute SQL query."""
        pass
        
    @abstractmethod
    async def run(self) -> None:
        """Execute pipeline."""
        pass
        
    @abstractmethod
    async def stop(self) -> None:
        """Stop pipeline execution."""
        pass
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics."""
        return self._metrics
        
    def get_lineage(self) -> List[Dict[str, Any]]:
        """Get data lineage."""
        return self._lineage


class StreamingPipeline(Pipeline):
    """Streaming pipeline interface."""
    
    @abstractmethod
    async def read_stream(self, source: "Source") -> StreamingDataFrame:
        """Read streaming source."""
        pass
        
    @abstractmethod
    async def read_kafka(
        self,
        topic: Union[str, List[str]],
        bootstrap_servers: str,
        start_offset: str = "latest",
        schema: Optional["Schema"] = None,
        **options: Any,
    ) -> StreamingDataFrame:
        """Read from Kafka."""
        pass
        
    @abstractmethod
    async def read_socket(
        self,
        host: str,
        port: int,
        format: str = "text",
        **options: Any,
    ) -> StreamingDataFrame:
        """Read from socket."""
        pass
        
    @abstractmethod
    async def read_rate(
        self,
        rows_per_second: int = 1,
        ramp_up_time: int = 0,
        num_partitions: int = 1,
    ) -> StreamingDataFrame:
        """Generate rate stream for testing."""
        pass