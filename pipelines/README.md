# Commons Pipelines

Comprehensive data pipeline framework with unified APIs for multiple processing engines, supporting batch and streaming workloads with enterprise-grade features.

## Installation

```bash
pip install commons-pipelines
```

## Features

- **Multi-Engine Support**: Spark, Polars, DuckDB, Apache Beam, Dask, Ray with unified APIs
- **Rich Format Support**: CSV, JSON, Parquet, Avro, ORC, Delta Lake with automatic schema inference
- **Advanced Transformations**: Complex SQL operations, window functions, aggregations, joins
- **Streaming Pipelines**: Real-time processing with watermarks, windowing, and exactly-once semantics
- **Extensive Connectors**: Databases, cloud storage, APIs, message queues, data warehouses
- **Data Quality**: Built-in validation, profiling, and monitoring capabilities
- **Performance Optimization**: Automatic query optimization, caching, partitioning strategies
- **Orchestration Integration**: Native support for Airflow, Argo Workflows, Prefect
- **Testing Framework**: Comprehensive testing utilities with mock data sources
- **Observability**: Metrics, logging, lineage tracking, and cost monitoring

## Quick Start

```python
import asyncio
from commons_pipelines import Pipeline, Source, SourceType, SourceOptions

async def main():
    # Create pipeline
    pipeline = Pipeline(name="demo-pipeline")
    
    # Read CSV data
    options = SourceOptions(path="data/users.csv")
    source = Source(SourceType.FILE, options)
    
    # Transform data
    result = (
        pipeline
        .read(source)
        .filter("age >= 18")
        .select(["user_id", "name", "email", "age"])
        .groupby("age")
        .agg({"user_id": "count"})
        .rename({"user_id": "user_count"})
    )
    
    # Display results
    data = await result.collect()
    for row in data:
        print(f"Age {row['age']}: {row['user_count']} users")
    
    # Execute pipeline
    await pipeline.run()

asyncio.run(main())
```

## Detailed Usage Examples

### Multi-Engine Pipeline Development

#### Engine-Agnostic Pipeline Creation
```python
import asyncio
from commons_pipelines import (
    Pipeline, PipelineFactory, EngineType, Source, Sink,
    SourceType, SourceOptions, SinkOptions, DataType, Schema
)
from datetime import datetime, timedelta
import pandas as pd

async def demonstrate_multi_engine_pipelines():
    """Demonstrate pipeline creation across different engines."""
    
    print("=== Multi-Engine Pipeline Development ===")
    
    # Engine configurations
    engine_configs = {
        EngineType.SPARK: {
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
            "spark.executor.memory": "4g",
            "spark.driver.memory": "2g",
            "spark.sql.warehouse.dir": "/tmp/spark-warehouse"
        },
        EngineType.POLARS: {
            "streaming": True,
            "lazy": True,
            "n_threads": 4
        },
        EngineType.DUCKDB: {
            "memory_limit": "4GB",
            "threads": 4,
            "enable_progress_bar": True
        },
        EngineType.PANDAS: {
            "low_memory": False,
            "engine": "c"
        }
    }
    
    # Sample data for testing
    sample_data = [
        {"user_id": i, "name": f"User {i}", "age": 20 + (i % 40), 
         "city": ["NYC", "LA", "Chicago", "Houston"][i % 4],
         "salary": 50000 + (i * 1000), "department": ["Engineering", "Sales", "Marketing"][i % 3],
         "hire_date": (datetime.now() - timedelta(days=i*30)).isoformat()}
        for i in range(1, 1001)
    ]
    
    # Create temporary CSV file
    import tempfile
    import csv
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        if sample_data:
            writer = csv.DictWriter(f, fieldnames=sample_data[0].keys())
            writer.writeheader()
            writer.writerows(sample_data)
        temp_file = f.name
    
    try:
        # Test the same pipeline logic across different engines
        for engine_type, config in engine_configs.items():
            print(f"\n--- Testing {engine_type.value.upper()} Engine ---")
            
            try:
                # Create pipeline with specific engine
                pipeline = PipelineFactory.create(
                    engine=engine_type,
                    config=config,
                    name=f"demo-{engine_type.value}"
                )
                
                # Define schema
                schema = Schema([
                    ("user_id", DataType.INT),
                    ("name", DataType.STRING),
                    ("age", DataType.INT),
                    ("city", DataType.STRING),
                    ("salary", DataType.FLOAT),
                    ("department", DataType.STRING),
                    ("hire_date", DataType.TIMESTAMP)
                ])
                
                # Read data
                source_options = SourceOptions(
                    path=temp_file,
                    schema=schema,
                    header=True
                )
                source = Source(SourceType.FILE, source_options)
                
                # Apply consistent transformations
                start_time = datetime.now()
                
                result = (
                    pipeline
                    .read(source)
                    .filter("age >= 25 AND age <= 45")  # Working age filter
                    .filter("salary > 55000")  # Salary filter
                    .with_column("age_group", 
                        "CASE WHEN age < 30 THEN 'Young' "
                        "     WHEN age < 40 THEN 'Mid' "
                        "     ELSE 'Senior' END")
                    .with_column("salary_tier",
                        "CASE WHEN salary < 60000 THEN 'Entry' "
                        "     WHEN salary < 80000 THEN 'Mid' "
                        "     ELSE 'Senior' END")
                    .groupby(["department", "age_group", "city"])
                    .agg({
                        "user_id": "count",
                        "salary": ["avg", "min", "max", "std"]
                    })
                    .rename({
                        "user_id_count": "employee_count",
                        "salary_avg": "avg_salary",
                        "salary_min": "min_salary",
                        "salary_max": "max_salary",
                        "salary_std": "salary_stddev"
                    })
                    .order_by(["department", "avg_salary"], ascending=[True, False])
                )
                
                # Collect results
                data = await result.collect()
                processing_time = (datetime.now() - start_time).total_seconds()
                
                print(f"  ✓ Engine: {engine_type.value}")
                print(f"  ✓ Processing time: {processing_time:.2f}s")
                print(f"  ✓ Result rows: {len(data)}")
                print(f"  ✓ Sample result: {data[0] if data else 'No data'}")
                
                # Engine-specific optimizations
                if engine_type == EngineType.SPARK:
                    # Show Spark-specific optimizations
                    execution_plan = pipeline.explain()
                    print(f"  ✓ Spark execution plan available: {len(execution_plan) > 0}")
                
                elif engine_type == EngineType.POLARS:
                    # Polars lazy evaluation benefits
                    print(f"  ✓ Polars lazy evaluation enabled")
                
                elif engine_type == EngineType.DUCKDB:
                    # DuckDB in-memory analytics performance
                    print(f"  ✓ DuckDB in-memory processing optimized")
                
                await pipeline.close()
                
            except Exception as e:
                print(f"  ⚠ {engine_type.value} engine failed: {e}")
    
    finally:
        # Cleanup temporary file
        os.unlink(temp_file)

# Complex data transformation pipeline
async def demonstrate_advanced_transformations():
    """Demonstrate advanced data transformation capabilities."""
    
    print("\n=== Advanced Data Transformations ===")
    
    # Create a comprehensive pipeline with multiple data sources
    pipeline = PipelineFactory.create(
        engine=EngineType.SPARK,  # Use Spark for complex operations
        config={
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.skewJoin.enabled": "true"
        }
    )
    
    # Generate more complex sample data
    import random
    from datetime import datetime, timedelta
    
    # Users data
    users_data = []
    departments = ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations"]
    cities = ["New York", "San Francisco", "Chicago", "Austin", "Seattle", "Boston"]
    
    for i in range(1, 5001):
        hire_date = datetime.now() - timedelta(days=random.randint(30, 1825))  # Last 5 years
        users_data.append({
            "user_id": i,
            "name": f"Employee {i:04d}",
            "email": f"emp{i:04d}@company.com",
            "age": random.randint(22, 65),
            "department": random.choice(departments),
            "city": random.choice(cities),
            "salary": random.randint(45000, 200000),
            "hire_date": hire_date.strftime("%Y-%m-%d"),
            "performance_score": round(random.uniform(2.0, 5.0), 1),
            "is_remote": random.choice([True, False])
        })
    
    # Projects data
    projects_data = []
    project_types = ["Web Development", "Mobile App", "Data Analytics", "Infrastructure", "AI/ML"]
    
    for i in range(1, 201):
        start_date = datetime.now() - timedelta(days=random.randint(0, 365))
        end_date = start_date + timedelta(days=random.randint(30, 365))
        
        projects_data.append({
            "project_id": i,
            "project_name": f"Project {i:03d}",
            "project_type": random.choice(project_types),
            "budget": random.randint(10000, 1000000),
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "status": random.choice(["Planning", "In Progress", "Completed", "On Hold"]),
            "priority": random.choice(["Low", "Medium", "High", "Critical"])
        })
    
    # Project assignments data
    assignments_data = []
    for i in range(1, 1001):
        assignments_data.append({
            "assignment_id": i,
            "user_id": random.randint(1, 5000),
            "project_id": random.randint(1, 200),
            "role": random.choice(["Developer", "Lead", "Architect", "Manager", "Analyst"]),
            "allocation_pct": random.randint(10, 100),
            "start_date": (datetime.now() - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d")
        })
    
    # Create temporary files for each dataset
    import tempfile
    import csv
    
    temp_files = {}
    datasets = {
        "users": users_data,
        "projects": projects_data,
        "assignments": assignments_data
    }
    
    try:
        # Create temporary CSV files
        for name, data in datasets.items():
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                if data:
                    writer = csv.DictWriter(f, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
                temp_files[name] = f.name
        
        # Read all datasets
        users_source = Source(SourceType.FILE, SourceOptions(path=temp_files["users"], header=True))
        projects_source = Source(SourceType.FILE, SourceOptions(path=temp_files["projects"], header=True))
        assignments_source = Source(SourceType.FILE, SourceOptions(path=temp_files["assignments"], header=True))
        
        users_df = pipeline.read(users_source)
        projects_df = pipeline.read(projects_source)
        assignments_df = pipeline.read(assignments_source)
        
        print("\nLoaded datasets:")
        print(f"  Users: {await users_df.count()} records")
        print(f"  Projects: {await projects_df.count()} records")
        print(f"  Assignments: {await assignments_df.count()} records")
        
        # 1. Complex join operations
        print("\n1. Complex Join Operations:")
        
        # Multi-way join to get employee project assignments with details
        employee_projects = (
            assignments_df
            .join(users_df, on="user_id", how="inner")
            .join(projects_df, on="project_id", how="inner")
            .select([
                "assignment_id", "user_id", "name", "department", "city",
                "project_id", "project_name", "project_type", "budget",
                "role", "allocation_pct", "salary", "performance_score"
            ])
        )
        
        sample_assignments = await employee_projects.limit(5).collect()
        print(f"  ✓ Created employee-project view: {len(sample_assignments)} sample records")
        
        # 2. Window functions for ranking and analytics
        print("\n2. Window Functions and Rankings:")
        
        from commons_pipelines import Window, Functions as F
        
        # Create windows for different analyses
        dept_window = Window.partition_by("department").order_by("salary", ascending=False)
        city_window = Window.partition_by("city").order_by("performance_score", ascending=False)
        
        ranked_employees = (
            users_df
            .with_column("salary_rank_in_dept", F.row_number().over(dept_window))
            .with_column("performance_rank_in_city", F.row_number().over(city_window))
            .with_column("dept_avg_salary", F.avg("salary").over(Window.partition_by("department")))
            .with_column("salary_vs_dept_avg", 
                F.col("salary") - F.col("dept_avg_salary"))
            .with_column("percentile_rank", 
                F.percent_rank().over(Window.order_by("salary")))
        )
        
        top_performers = await (
            ranked_employees
            .filter("salary_rank_in_dept <= 3 OR performance_rank_in_city <= 3")
            .select([
                "name", "department", "city", "salary", "performance_score",
                "salary_rank_in_dept", "performance_rank_in_city", 
                "salary_vs_dept_avg", "percentile_rank"
            ])
            .order_by(["department", "salary_rank_in_dept"])
            .collect()
        )
        
        print(f"  ✓ Top performers identified: {len(top_performers)} employees")
        print(f"  ✓ Sample: {top_performers[0]['name']} - Rank {top_performers[0]['salary_rank_in_dept']} in {top_performers[0]['department']}")
        
        # 3. Advanced aggregations with multiple grouping sets
        print("\n3. Advanced Aggregations:")
        
        # Department and city analysis with rollups
        dept_city_analysis = (
            employee_projects
            .groupby(["department", "city", "project_type"])
            .agg({
                "user_id": "count",
                "salary": ["avg", "min", "max", "std"],
                "budget": "sum",
                "allocation_pct": "avg",
                "performance_score": "avg"
            })
            .rename({
                "user_id_count": "employee_count",
                "salary_avg": "avg_salary",
                "salary_min": "min_salary",
                "salary_max": "max_salary",
                "salary_std": "salary_stddev",
                "budget_sum": "total_budget",
                "allocation_pct_avg": "avg_allocation",
                "performance_score_avg": "avg_performance"
            })
            .with_column("budget_per_employee", 
                F.col("total_budget") / F.col("employee_count"))
        )
        
        analysis_results = await dept_city_analysis.collect()
        print(f"  ✓ Department-City-Project analysis: {len(analysis_results)} combinations")
        
        # Find highest budget per employee combinations
        top_budget_combos = sorted(
            analysis_results, 
            key=lambda x: x.get('budget_per_employee', 0), 
            reverse=True
        )[:3]
        
        for combo in top_budget_combos:
            print(f"  ✓ High budget efficiency: {combo['department']} - {combo['city']} - {combo['project_type']}")
            print(f"    Budget per employee: ${combo.get('budget_per_employee', 0):,.0f}")
        
        # 4. Pivot tables and cross-tabulations
        print("\n4. Pivot Tables and Cross-tabulations:")
        
        # Create pivot table showing average salary by department and city
        dept_city_pivot = (
            users_df
            .groupby("department")
            .pivot("city", values=["New York", "San Francisco", "Chicago", "Austin", "Seattle", "Boston"])
            .agg(F.avg("salary"))
        )
        
        pivot_results = await dept_city_pivot.collect()
        print(f"  ✓ Department-City salary pivot: {len(pivot_results)} departments")
        
        # 5. Complex filtering with subqueries
        print("\n5. Complex Filtering and Subqueries:")
        
        # Find employees working on high-budget projects (top 20% by budget)
        high_budget_threshold = await (
            projects_df
            .select(F.percentile_approx("budget", 0.8).alias("threshold"))
            .collect()
        )
        
        threshold_value = high_budget_threshold[0]['threshold']
        
        high_budget_employees = (
            employee_projects
            .filter(f"budget >= {threshold_value}")
            .groupby(["user_id", "name", "department"])
            .agg({
                "project_id": "count",
                "budget": "sum",
                "allocation_pct": "avg"
            })
            .rename({
                "project_id_count": "high_budget_project_count",
                "budget_sum": "total_high_budget_exposure",
                "allocation_pct_avg": "avg_allocation_on_high_budget"
            })
            .filter("high_budget_project_count >= 2")  # At least 2 high-budget projects
            .order_by("total_high_budget_exposure", ascending=False)
        )
        
        high_value_employees = await high_budget_employees.limit(10).collect()
        print(f"  ✓ High-value employees (working on multiple high-budget projects): {len(high_value_employees)}")
        
        if high_value_employees:
            top_employee = high_value_employees[0]
            print(f"  ✓ Top contributor: {top_employee['name']} ({top_employee['department']})")
            print(f"    High-budget projects: {top_employee['high_budget_project_count']}")
            print(f"    Total budget exposure: ${top_employee['total_high_budget_exposure']:,.0f}")
        
        await pipeline.close()
        
    finally:
        # Cleanup temporary files
        import os
        for temp_file in temp_files.values():
            os.unlink(temp_file)

# Run demonstrations
asyncio.run(demonstrate_multi_engine_pipelines())
asyncio.run(demonstrate_advanced_transformations())
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