"""Simple concrete pipeline implementation."""

from typing import Any, Dict, List, Optional, Union
from .abstractions.base import Pipeline, DataFrame, PipelineConfig
from .abstractions.source import Source
from .abstractions.schema import Schema

class SimpleDataFrame(DataFrame):
    """Simple DataFrame implementation."""
    
    def __init__(self, pipeline: "Pipeline", data: List[Dict[str, Any]]):
        super().__init__(pipeline, data)
    
    async def collect(self) -> List[Dict[str, Any]]:
        """Collect all rows as list of dictionaries."""
        return self._data if isinstance(self._data, list) else []
    
    async def count(self) -> int:
        """Count number of rows."""
        return len(self._data) if isinstance(self._data, list) else 0
    
    async def show(self, n: int = 20, truncate: bool = True) -> None:
        """Show first n rows."""
        data = self._data if isinstance(self._data, list) else []
        for i, row in enumerate(data[:n]):
            print(f"Row {i}: {row}")
    
    def select(self, *columns: Union[str, "Column"]) -> "DataFrame":
        """Select columns."""
        if not isinstance(self._data, list):
            return SimpleDataFrame(self.pipeline, [])
        
        selected_data = []
        for row in self._data:
            selected_row = {col: row.get(col) for col in columns if isinstance(col, str)}
            selected_data.append(selected_row)
        
        return SimpleDataFrame(self.pipeline, selected_data)
    
    def filter(self, condition: Any) -> "DataFrame":
        """Filter rows."""
        return SimpleDataFrame(self.pipeline, self._data)
    
    def groupBy(self, *columns: str) -> "GroupedDataFrame":
        """Group by columns."""
        from .grouped import SimpleGroupedDataFrame
        return SimpleGroupedDataFrame(self.pipeline, self._data, list(columns))
    
    def orderBy(self, *columns: Union[str, "Column"], ascending: bool = True) -> "DataFrame":
        """Order by columns."""
        return SimpleDataFrame(self.pipeline, self._data)
    
    def limit(self, num: int) -> "DataFrame":
        """Limit number of rows."""
        data = self._data[:num] if isinstance(self._data, list) else []
        return SimpleDataFrame(self.pipeline, data)

class SimplePipeline(Pipeline):
    """Simple concrete pipeline implementation."""
    
    def __init__(self, name: str = "simple-pipeline"):
        config = PipelineConfig(name=name, engine="simple")
        super().__init__(config)
    
    async def read(self, source: Source) -> DataFrame:
        """Read from source."""
        # Mock read implementation
        return SimpleDataFrame(self, [{"id": 1, "name": "test"}])
    
    async def read_csv(
        self,
        path: str,
        schema: Optional[Schema] = None,
        header: bool = True,
        delimiter: str = ",",
        **options: Any,
    ) -> DataFrame:
        """Read CSV files."""
        # Mock CSV read
        return SimpleDataFrame(self, [{"col1": "value1", "col2": "value2"}])
    
    async def read_json(
        self,
        path: str,
        schema: Optional[Schema] = None,
        **options: Any,
    ) -> DataFrame:
        """Read JSON files."""
        # Mock JSON read
        return SimpleDataFrame(self, [{"json_field": "json_value"}])
    
    async def read_parquet(
        self,
        path: str,
        schema: Optional[Schema] = None,
        **options: Any,
    ) -> DataFrame:
        """Read Parquet files."""
        # Mock parquet read
        return SimpleDataFrame(self, [{"parquet_col": "parquet_val"}])
    
    async def read_table(
        self,
        table_name: str,
        schema: Optional[Schema] = None,
        **options: Any,
    ) -> DataFrame:
        """Read from table."""
        # Mock table read
        return SimpleDataFrame(self, [{"table_field": "table_value"}])
    
    async def read_sql(
        self,
        query: str,
        **options: Any,
    ) -> DataFrame:
        """Execute SQL query."""
        # Mock SQL execution
        return SimpleDataFrame(self, [{"result": "sql_result"}])
    
    def create_dataframe(
        self,
        data: List[Dict[str, Any]],
        schema: Optional[Schema] = None,
    ) -> DataFrame:
        """Create DataFrame from data."""
        return SimpleDataFrame(self, data)
    
    def sql(self, query: str) -> DataFrame:
        """Execute SQL query synchronously."""
        # Mock SQL execution
        return SimpleDataFrame(self, [{"sql_result": "executed"}])
    
    async def run(self) -> None:
        """Run the pipeline."""
        print(f"Running pipeline: {self.config.name}")
    
    async def stop(self) -> None:
        """Stop the pipeline."""
        print(f"Stopping pipeline: {self.config.name}")