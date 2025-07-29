# Commons Pipelines

Data pipeline abstractions supporting multiple processing engines and formats.

## Features

- **Multiple Engines**: Spark, Polars, DuckDB, Beam, Dask, Ray
- **Format Support**: CSV, JSON, Parquet, Avro, ORC with schema inference
- **Transformations**: Map, filter, aggregate, join, window functions
- **Connectors**: Databases, files, APIs, message queues
- **Orchestration**: Integration with Argo, Airflow, Prefect

## Installation

```bash
# Basic installation
pip install commons-pipelines

# With specific engines
pip install commons-pipelines[spark]
pip install commons-pipelines[polars]
pip install commons-pipelines[duckdb]
pip install commons-pipelines[all]  # All engines
```

## Usage

### Basic Pipeline

```python
from commons_pipelines import Pipeline, Source, Sink

# Create pipeline
pipeline = Pipeline("data-processing")

# Define source
source = Source.csv("s3://bucket/data/*.csv", schema="auto")

# Apply transformations
result = (
    pipeline
    .read(source)
    .filter("age > 18")
    .select("user_id", "email", "age")
    .groupby("age")
    .agg({"user_id": "count"})
    .rename({"user_id": "user_count"})
)

# Write results
sink = Sink.parquet("s3://bucket/output/")
await result.write(sink)

# Execute pipeline
await pipeline.run()
```

### Multi-Engine Support

```python
from commons_pipelines import EngineType, PipelineFactory

# Create pipeline with specific engine
spark_pipeline = PipelineFactory.create(
    engine=EngineType.SPARK,
    config={"spark.sql.adaptive.enabled": "true"}
)

polars_pipeline = PipelineFactory.create(
    engine=EngineType.POLARS,
    config={"streaming": True}
)

duckdb_pipeline = PipelineFactory.create(
    engine=EngineType.DUCKDB,
    config={"memory_limit": "4GB"}
)

# Use the same API regardless of engine
df = await spark_pipeline.read_parquet("data.parquet")
result = df.filter("value > 100").groupby("category").sum("value")
await result.write_csv("output.csv")
```

### Schema Management

```python
from commons_pipelines import Schema, DataType

# Define schema
schema = Schema([
    ("user_id", DataType.STRING),
    ("age", DataType.INT),
    ("created_at", DataType.TIMESTAMP),
    ("metadata", DataType.STRUCT({
        "source": DataType.STRING,
        "version": DataType.INT,
    })),
])

# Read with schema
df = await pipeline.read_csv("data.csv", schema=schema)

# Infer schema
inferred = await pipeline.infer_schema("data.parquet")
print(inferred)

# Schema evolution
new_schema = schema.add_column("email", DataType.STRING, nullable=True)
df = df.with_schema(new_schema)
```

### Complex Transformations

```python
from commons_pipelines import Window, Functions as F

# Window functions
window = Window.partition_by("department").order_by("salary")

df = (
    df
    .with_column("rank", F.row_number().over(window))
    .with_column("dept_avg", F.avg("salary").over(window))
    .with_column("salary_pct", F.col("salary") / F.col("dept_avg"))
)

# Complex aggregations
result = (
    df
    .groupby("department", "year")
    .agg(
        F.count("*").alias("employee_count"),
        F.avg("salary").alias("avg_salary"),
        F.stddev("salary").alias("salary_stddev"),
        F.collect_list("employee_id").alias("employee_ids"),
    )
    .filter(F.col("employee_count") > 5)
)

# Joins
employees = await pipeline.read_table("employees")
departments = await pipeline.read_table("departments")

joined = employees.join(
    departments,
    on="department_id",
    how="left",
    suffix=("_emp", "_dept")
)

# Pivot tables
pivot = (
    df
    .groupby("department")
    .pivot("year", values=[2022, 2023, 2024])
    .agg(F.sum("revenue"))
)
```

### Streaming Pipelines

```python
from commons_pipelines import StreamingPipeline, Watermark

# Create streaming pipeline
stream = StreamingPipeline("real-time-analytics")

# Read from Kafka
source = Source.kafka(
    topic="events",
    bootstrap_servers="localhost:9092",
    format="json",
)

# Apply watermark for late data
watermarked = (
    stream
    .read(source)
    .with_watermark("event_time", "10 minutes")
)

# Windowed aggregation
windowed = (
    watermarked
    .groupby(
        F.window("event_time", "5 minutes", "1 minute"),
        "event_type"
    )
    .count()
)

# Write to multiple sinks
await windowed.write_multi([
    Sink.kafka("output-topic", format="avro"),
    Sink.parquet("s3://bucket/streaming/", mode="append"),
    Sink.console(truncate=False),
])

# Start streaming
await stream.start()
```

### Connectors

```python
from commons_pipelines.connectors import DatabaseConnector, APIConnector

# Database connector
db = DatabaseConnector(
    url="postgresql://localhost/mydb",
    driver="psycopg2",
)

# Read from database
df = await pipeline.read_sql(
    "SELECT * FROM users WHERE created_at > '2024-01-01'",
    connection=db,
    partition_column="user_id",
    num_partitions=4,
)

# Write to database
await df.write_sql(
    table="processed_users",
    connection=db,
    mode="overwrite",
    batch_size=1000,
)

# API connector
api = APIConnector(
    base_url="https://api.example.com",
    auth_token="secret",
    rate_limit=100,  # requests per second
)

# Read from API with pagination
df = await pipeline.read_api(
    connector=api,
    endpoint="/users",
    pagination="cursor",
    max_pages=10,
)

# Parallel API calls
urls = ["endpoint1", "endpoint2", "endpoint3"]
dfs = await pipeline.read_api_parallel(
    connector=api,
    endpoints=urls,
    max_concurrent=5,
)
combined = dfs[0].union_all(*dfs[1:])
```

### Orchestration

```python
from commons_pipelines.orchestration import ArgoWorkflow, AirflowDAG

# Create Argo workflow
workflow = ArgoWorkflow("etl-pipeline")

# Define tasks
extract = workflow.task(
    name="extract",
    image="etl:latest",
    command=["python", "extract.py"],
)

transform = workflow.task(
    name="transform",
    image="etl:latest",
    command=["python", "transform.py"],
    dependencies=[extract],
)

load = workflow.task(
    name="load",
    image="etl:latest",
    command=["python", "load.py"],
    dependencies=[transform],
)

# Generate workflow YAML
yaml = workflow.to_yaml()

# Create Airflow DAG
dag = AirflowDAG(
    "etl_pipeline",
    schedule="0 2 * * *",  # Daily at 2 AM
    catchup=False,
)

# Convert pipeline to Airflow tasks
pipeline_task = dag.pipeline_operator(
    task_id="run_pipeline",
    pipeline=pipeline,
    engine="spark",
    config={"spark.executor.memory": "4g"},
)
```

### Performance Optimization

```python
from commons_pipelines import Optimizer

# Create optimizer
optimizer = Optimizer()

# Analyze pipeline
analysis = optimizer.analyze(pipeline)
print(f"Estimated cost: {analysis.cost}")
print(f"Estimated runtime: {analysis.runtime}")

# Get optimization suggestions
suggestions = optimizer.suggest(pipeline)
for suggestion in suggestions:
    print(f"{suggestion.type}: {suggestion.description}")

# Auto-optimize
optimized = optimizer.optimize(pipeline)

# Caching
df = df.cache()  # Cache in memory
df = df.persist("disk")  # Persist to disk

# Partitioning
df = df.repartition(100, "user_id")  # Repartition by column
df = df.coalesce(10)  # Reduce partitions

# Broadcast joins
small_df = await pipeline.read_csv("small_lookup.csv")
result = df.join(F.broadcast(small_df), on="id")
```

### Testing

```python
from commons_pipelines.testing import PipelineTest, MockSource

# Create test
class TestETLPipeline(PipelineTest):
    def test_transformation(self):
        # Create mock data
        source = MockSource([
            {"id": 1, "value": 100},
            {"id": 2, "value": 200},
        ])
        
        # Run pipeline
        result = (
            self.pipeline
            .read(source)
            .filter("value > 150")
            .collect()
        )
        
        # Assert results
        assert len(result) == 1
        assert result[0]["value"] == 200

# Data quality tests
from commons_pipelines.quality import DataQuality

quality = DataQuality(df)

# Run quality checks
report = quality.check(
    completeness=["user_id", "email"],
    uniqueness=["user_id"],
    validity={
        "email": r"^[\w\.-]+@[\w\.-]+\.\w+$",
        "age": lambda x: 0 <= x <= 150,
    },
    consistency=[
        ("created_at", "updated_at", lambda c, u: c <= u),
    ],
)

print(f"Quality score: {report.score:.2%}")
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev,all]"

# Run tests
pytest

# Run specific engine tests
pytest -k spark
pytest -k polars
```