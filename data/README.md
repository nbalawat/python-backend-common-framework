# Commons Data

Unified database abstractions for SQL, NoSQL, time-series, and vector databases.

## Features

- **Relational Databases**: PostgreSQL, MySQL, SQLite, SQL Server, Oracle
- **NoSQL Databases**: MongoDB, Redis, DynamoDB, Cassandra, Neo4j
- **Time-Series Databases**: InfluxDB, TimescaleDB, Prometheus
- **Vector Databases**: Pinecone, Qdrant, Weaviate, pgvector
- **Search Engines**: Elasticsearch, OpenSearch
- **Cloud Databases**: DynamoDB, Cosmos DB, Firestore

## Installation

```bash
# Basic installation
pip install commons-data

# With specific databases
pip install commons-data[postgres,redis]
pip install commons-data[mongodb,elasticsearch]
pip install commons-data[all]  # All databases
```

## Usage

### Database Factory

```python
from commons_data import DatabaseFactory, DatabaseType

# Create database client
db = await DatabaseFactory.create(
    type=DatabaseType.POSTGRES,
    connection_string="postgresql://user:pass@localhost/mydb",
    pool_size=20,
)

# Or use URLs
db = await DatabaseFactory.from_url(
    "postgresql+asyncpg://user:pass@localhost/mydb"
)

# Execute queries
result = await db.execute("SELECT * FROM users WHERE age > ?", [18])
for row in result:
    print(row["name"], row["age"])
```

### Relational Databases

```python
from commons_data import AsyncDatabase, Table, Column

# Define schema
users_table = Table(
    "users",
    Column("id", "INTEGER", primary_key=True),
    Column("name", "VARCHAR(100)", nullable=False),
    Column("email", "VARCHAR(255)", unique=True),
    Column("created_at", "TIMESTAMP", default="NOW()"),
)

# Create connection
async with AsyncDatabase("postgresql://localhost/mydb") as db:
    # Create table
    await db.create_table(users_table)
    
    # Insert data
    user_id = await db.insert(
        "users",
        {"name": "John Doe", "email": "john@example.com"}
    )
    
    # Query with conditions
    users = await db.select(
        "users",
        where={"age": (">", 18)},
        order_by="created_at DESC",
        limit=10
    )
    
    # Update
    await db.update(
        "users",
        {"status": "active"},
        where={"id": user_id}
    )
    
    # Transaction
    async with db.transaction() as tx:
        await tx.insert("users", user1_data)
        await tx.insert("users", user2_data)
        # Automatically commits or rolls back

# Query builder
from commons_data import Query

query = (
    Query("users")
    .select("id", "name", "email")
    .join("orders", "users.id = orders.user_id")
    .where("users.status", "=", "active")
    .where("orders.total", ">", 100)
    .group_by("users.id")
    .having("COUNT(orders.id)", ">", 5)
    .order_by("users.created_at", "DESC")
    .limit(10)
)

results = await db.execute(query)
```

### NoSQL Databases

```python
# MongoDB
from commons_data.nosql import MongoDatabase

mongo = MongoDatabase("mongodb://localhost:27017/mydb")

# Insert document
doc_id = await mongo.insert_one(
    "users",
    {"name": "Jane Doe", "age": 25, "tags": ["python", "ai"]}
)

# Find documents
users = await mongo.find(
    "users",
    filter={"age": {"$gte": 18}},
    projection={"name": 1, "email": 1},
    sort=[("created_at", -1)],
    limit=10
)

# Aggregation pipeline
pipeline = [
    {"$match": {"status": "active"}},
    {"$group": {
        "_id": "$category",
        "count": {"$sum": 1},
        "avg_age": {"$avg": "$age"}
    }},
    {"$sort": {"count": -1}}
]
results = await mongo.aggregate("users", pipeline)

# Redis
from commons_data.nosql import RedisClient

redis = RedisClient("redis://localhost:6379")

# Key-value operations
await redis.set("user:123", {"name": "John", "email": "john@example.com"})
user = await redis.get("user:123")

# Hash operations
await redis.hset("user:123", "last_login", datetime.now())
fields = await redis.hgetall("user:123")

# Lists and sets
await redis.lpush("queue:tasks", task_data)
task = await redis.rpop("queue:tasks")

await redis.sadd("tags:python", "user:123", "user:456")
python_users = await redis.smembers("tags:python")

# Pub/Sub
async with redis.pubsub() as pubsub:
    await pubsub.subscribe("notifications")
    async for message in pubsub.listen():
        print(f"Received: {message}")
```

### Time-Series Databases

```python
from commons_data.timeseries import InfluxDBClient, TimePoint

# InfluxDB
influx = InfluxDBClient(
    url="http://localhost:8086",
    token="my-token",
    org="my-org",
    bucket="metrics"
)

# Write time-series data
points = [
    TimePoint(
        measurement="temperature",
        tags={"sensor": "sensor1", "location": "room1"},
        fields={"value": 22.5},
        timestamp=datetime.now()
    ),
]
await influx.write(points)

# Query with Flux
query = '''
from(bucket: "metrics")
    |> range(start: -1h)
    |> filter(fn: (r) => r["_measurement"] == "temperature")
    |> filter(fn: (r) => r["location"] == "room1")
    |> mean()
'''
results = await influx.query(query)

# TimescaleDB (PostgreSQL extension)
from commons_data.timeseries import TimescaleDB

ts = TimescaleDB("postgresql://localhost/tsdb")

# Create hypertable
await ts.create_hypertable(
    "sensor_data",
    time_column="time",
    partitioning_column="sensor_id",
    chunk_time_interval="1 day"
)

# Time-based queries
data = await ts.query(
    """
    SELECT 
        time_bucket('5 minutes', time) AS bucket,
        sensor_id,
        AVG(value) as avg_value,
        MAX(value) as max_value
    FROM sensor_data
    WHERE time > NOW() - INTERVAL '1 hour'
    GROUP BY bucket, sensor_id
    ORDER BY bucket DESC
    """
)

# Continuous aggregates
await ts.create_continuous_aggregate(
    "sensor_hourly",
    """
    SELECT 
        time_bucket('1 hour', time) AS hour,
        sensor_id,
        AVG(value) as avg_value,
        COUNT(*) as num_readings
    FROM sensor_data
    GROUP BY hour, sensor_id
    """,
    refresh_interval="1 hour"
)
```

### Vector Databases

```python
from commons_data.vector import VectorStore, Document

# Create vector store
vector_store = VectorStore(
    provider="pinecone",
    api_key="your-api-key",
    index_name="my-index",
    dimension=1536  # For OpenAI embeddings
)

# Index documents
documents = [
    Document(
        id="doc1",
        text="Python is a great programming language",
        metadata={"source": "tutorial", "chapter": 1}
    ),
    Document(
        id="doc2", 
        text="Machine learning with Python is powerful",
        metadata={"source": "guide", "chapter": 5}
    ),
]

await vector_store.index(
    documents,
    embedding_model="openai/text-embedding-3-small"
)

# Semantic search
results = await vector_store.search(
    query="How to learn Python?",
    k=5,
    filter={"source": "tutorial"},
    include_metadata=True
)

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Text: {result.text}")
    print(f"Metadata: {result.metadata}")

# Hybrid search (vector + keyword)
results = await vector_store.hybrid_search(
    query="Python programming",
    vector_weight=0.7,
    keyword_weight=0.3,
    k=10
)

# Update vectors
await vector_store.update(
    id="doc1",
    text="Python is an amazing programming language",
    metadata={"source": "tutorial", "chapter": 1, "updated": True}
)

# Delete vectors
await vector_store.delete(ids=["doc2"])
```

### ORM Models

```python
from commons_data.orm import Model, Field, ForeignKey

# Define models
class User(Model):
    __tablename__ = "users"
    
    id = Field(int, primary_key=True)
    name = Field(str, max_length=100)
    email = Field(str, unique=True, index=True)
    created_at = Field(datetime, default=datetime.now)
    
    # Relationships
    posts = Relationship("Post", back_populates="author")

class Post(Model):
    __tablename__ = "posts"
    
    id = Field(int, primary_key=True)
    title = Field(str, max_length=200)
    content = Field(str)
    author_id = ForeignKey("users.id")
    created_at = Field(datetime, default=datetime.now)
    
    # Relationships
    author = Relationship("User", back_populates="posts")
    tags = Relationship("Tag", secondary="post_tags")

# Use models
async with AsyncSession() as session:
    # Create
    user = User(name="John Doe", email="john@example.com")
    session.add(user)
    await session.commit()
    
    # Query
    users = await session.query(User).filter(
        User.created_at > datetime.now() - timedelta(days=7)
    ).all()
    
    # Join query
    posts_with_authors = await session.query(Post).join(
        User
    ).filter(
        User.status == "active"
    ).all()
    
    # Eager loading
    users_with_posts = await session.query(User).options(
        selectinload(User.posts)
    ).all()
```

### Multi-Database Transactions

```python
from commons_data import MultiDatabaseTransaction

# Coordinate transactions across databases
async with MultiDatabaseTransaction() as tx:
    # Add databases to transaction
    pg_tx = await tx.add_database(postgres_db)
    mongo_tx = await tx.add_database(mongo_db)
    redis_tx = await tx.add_database(redis_db)
    
    try:
        # Perform operations
        user_id = await pg_tx.insert("users", user_data)
        await mongo_tx.insert_one("profiles", profile_data)
        await redis_tx.set(f"user:{user_id}", cache_data)
        
        # All succeed or all rollback
        await tx.commit()
    except Exception as e:
        await tx.rollback()
        raise
```

### Database Migrations

```python
from commons_data.migrations import Migration, MigrationManager

# Define migration
class AddUserStatusColumn(Migration):
    version = "001"
    
    async def up(self, db):
        await db.execute("""
            ALTER TABLE users 
            ADD COLUMN status VARCHAR(20) DEFAULT 'active'
        """)
        
    async def down(self, db):
        await db.execute("ALTER TABLE users DROP COLUMN status")

# Run migrations
manager = MigrationManager(db, migrations_table="schema_migrations")
await manager.add_migration(AddUserStatusColumn())
await manager.upgrade()  # Run pending migrations
await manager.downgrade(version="001")  # Rollback to specific version
```

### Performance Features

```python
# Connection pooling
db = AsyncDatabase(
    "postgresql://localhost/mydb",
    pool_size=20,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=3600
)

# Prepared statements
stmt = await db.prepare(
    "SELECT * FROM users WHERE age > $1 AND status = $2"
)
results = await stmt.fetch(18, "active")

# Batch operations
await db.insert_many(
    "events",
    [
        {"type": "login", "user_id": 1, "timestamp": now},
        {"type": "purchase", "user_id": 2, "amount": 99.99},
        # ... many more
    ],
    batch_size=1000
)

# Query caching
from commons_data import CachedDatabase

cached_db = CachedDatabase(
    db,
    cache=redis_client,
    ttl=300,  # 5 minutes
    key_prefix="db:cache:"
)

# Subsequent identical queries hit cache
users = await cached_db.select("users", where={"status": "active"})
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev,postgres,mongodb,redis]"

# Run tests
pytest

# Run database-specific tests
pytest -k postgres
pytest -k mongodb
pytest -k redis
```