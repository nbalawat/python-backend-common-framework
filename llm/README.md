# Commons LLM

Unified LLM provider abstractions with advanced features for building AI applications.

## Features

- **Provider Support**: OpenAI, Anthropic, Google, Mistral, Cohere, and local models
- **Unified API**: Same interface for all providers with streaming support
- **Framework Integrations**: LangChain, LangGraph, LangSmith, LlamaIndex adapters
- **Advanced Features**: Function calling, embeddings, caching, fallbacks, A/B testing
- **Cost Tracking**: Monitor token usage and API costs across providers

## Installation

```bash
# Basic installation
pip install commons-llm

# With specific providers
pip install commons-llm[openai]
pip install commons-llm[anthropic]
pip install commons-llm[google]
pip install commons-llm[all]  # All providers

# With framework integrations
pip install commons-llm[langchain]
pip install commons-llm[llamaindex]
```

## Usage

### Basic Usage

```python
from commons_llm import LLMProvider, Message

# Create provider
llm = LLMProvider.create(
    provider="openai",
    model="gpt-4",
    api_key="your-api-key",
    temperature=0.7,
)

# Simple completion
response = await llm.complete("What is the capital of France?")
print(response.content)  # "The capital of France is Paris."

# Chat completion
messages = [
    Message(role="system", content="You are a helpful assistant."),
    Message(role="user", content="Explain quantum computing in simple terms."),
]
response = await llm.chat(messages)
print(response.content)

# Streaming
async for chunk in llm.stream("Tell me a story"):
    print(chunk.content, end="")
```

### Multi-Provider Support

```python
from commons_llm import LLMFactory, FallbackProvider

# Configure multiple providers
factory = LLMFactory({
    "openai": {
        "api_key": "openai-key",
        "models": ["gpt-4", "gpt-3.5-turbo"],
    },
    "anthropic": {
        "api_key": "anthropic-key",
        "models": ["claude-3-opus", "claude-3-sonnet"],
    },
    "google": {
        "api_key": "google-key",
        "models": ["gemini-pro"],
    },
})

# Create provider with fallbacks
llm = FallbackProvider([
    factory.create("anthropic", model="claude-3-opus"),
    factory.create("openai", model="gpt-4"),
    factory.create("google", model="gemini-pro"),
])

# Will try providers in order until one succeeds
response = await llm.complete("Hello!")
```

### Function Calling

```python
from commons_llm import Function, FunctionCall

# Define functions
functions = [
    Function(
        name="get_weather",
        description="Get the weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    ),
    Function(
        name="search_web",
        description="Search the web for information",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "num_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    ),
]

# Chat with function calling
response = await llm.chat(
    messages=[Message(role="user", content="What's the weather in Paris?")],
    functions=functions,
)

# Handle function calls
if response.function_calls:
    for call in response.function_calls:
        if call.name == "get_weather":
            # Execute function
            weather = get_weather(**call.arguments)
            
            # Send result back
            messages.append(response.to_message())
            messages.append(Message(
                role="function",
                name=call.name,
                content=json.dumps(weather),
            ))
            
            # Get final response
            final_response = await llm.chat(messages)
```

### Embeddings

```python
from commons_llm import EmbeddingProvider

# Create embedding provider
embeddings = EmbeddingProvider.create(
    provider="openai",
    model="text-embedding-3-small",
)

# Single embedding
vector = await embeddings.embed("Hello, world!")
print(f"Dimension: {len(vector)}")  # 1536 for OpenAI

# Batch embeddings
texts = ["First text", "Second text", "Third text"]
vectors = await embeddings.embed_batch(texts)

# With custom dimensions (OpenAI v3)
embeddings = EmbeddingProvider.create(
    provider="openai",
    model="text-embedding-3-large",
    dimensions=256,  # Reduce from 3072
)
```

### Prompt Templates

```python
from commons_llm.prompts import PromptTemplate, ChatTemplate

# Simple template
prompt = PromptTemplate(
    template="Translate the following {language} text to English: {text}",
    input_variables=["language", "text"],
)

rendered = prompt.format(language="French", text="Bonjour le monde")
response = await llm.complete(rendered)

# Chat template
chat_template = ChatTemplate([
    Message(role="system", content="You are a {role} assistant."),
    Message(role="user", content="{user_input}"),
])

messages = chat_template.format(
    role="helpful coding",
    user_input="Write a Python function to sort a list",
)
response = await llm.chat(messages)

# Template library
from commons_llm.prompts import PromptLibrary

library = PromptLibrary()
library.add_template("translate", prompt)
library.add_template("code_assist", chat_template)

# Load from file
library.load_from_file("prompts.yaml")

# Use template
translate_prompt = library.get("translate")
```

### Response Caching

```python
from commons_llm import CachedLLMProvider
from commons_llm.cache import RedisCache, InMemoryCache

# In-memory cache
llm = CachedLLMProvider(
    provider=base_llm,
    cache=InMemoryCache(max_size=1000, ttl=3600),
)

# Redis cache
llm = CachedLLMProvider(
    provider=base_llm,
    cache=RedisCache(
        redis_url="redis://localhost:6379",
        ttl=86400,
        namespace="llm_cache",
    ),
)

# First call hits API
response1 = await llm.complete("What is 2+2?")

# Subsequent identical calls use cache
response2 = await llm.complete("What is 2+2?")  # From cache
```

### Cost Tracking

```python
from commons_llm import CostTracker

# Initialize cost tracker
tracker = CostTracker()

# Track costs
llm = tracker.track(llm)

# Make requests
await llm.complete("Hello")
await llm.chat(messages)

# Get cost report
report = tracker.get_report()
print(f"Total cost: ${report.total_cost:.4f}")
print(f"Total tokens: {report.total_tokens}")

# Detailed breakdown
for model, stats in report.by_model.items():
    print(f"{model}: ${stats.cost:.4f} ({stats.tokens} tokens)")

# Set cost alerts
tracker.set_alert(daily_limit=10.0)  # Alert if daily cost exceeds $10
```

### LangChain Integration

```python
from commons_llm.integrations.langchain import CommonsLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Use Commons LLM in LangChain
langchain_llm = CommonsLLM(
    provider="anthropic",
    model="claude-3-sonnet",
    temperature=0.5,
)

# Create chain
prompt = PromptTemplate(
    input_variables=["product"],
    template="Write a product description for: {product}",
)
chain = LLMChain(llm=langchain_llm, prompt=prompt)

# Run chain
result = await chain.arun(product="Smart Watch")

# Streaming with callbacks
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

langchain_llm = CommonsLLM(
    provider="openai",
    model="gpt-4",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)
```

### A/B Testing

```python
from commons_llm import ABTestProvider, Experiment

# Create A/B test
experiment = Experiment(
    name="model_comparison",
    variants={
        "control": factory.create("openai", model="gpt-3.5-turbo"),
        "treatment": factory.create("anthropic", model="claude-3-haiku"),
    },
    weights=[0.5, 0.5],  # 50/50 split
)

ab_llm = ABTestProvider(experiment)

# Requests are randomly distributed
for i in range(100):
    response = await ab_llm.complete(f"Test prompt {i}")
    
# Get results
results = experiment.get_results()
print(f"Control: {results['control'].success_rate:.2%} success")
print(f"Treatment: {results['treatment'].success_rate:.2%} success")
print(f"Treatment is {results.lift:.2%} better")
```

### Local Models

```python
from commons_llm.providers import LocalLLMProvider

# Load local model
llm = LocalLLMProvider(
    model_name="microsoft/phi-2",
    device="cuda",  # or "cpu"
    load_in_8bit=True,
)

# Use like any other provider
response = await llm.complete("Explain machine learning")

# Custom model loading
llm = LocalLLMProvider.from_pretrained(
    model_path="/path/to/model",
    tokenizer_path="/path/to/tokenizer",
    model_kwargs={"torch_dtype": torch.float16},
)
```

## Advanced Configuration

```python
from commons_llm import LLMConfig, RetryConfig, RateLimitConfig

config = LLMConfig(
    # Model settings
    temperature=0.7,
    max_tokens=2000,
    top_p=0.9,
    frequency_penalty=0.1,
    presence_penalty=0.1,
    
    # Retry configuration
    retry=RetryConfig(
        max_attempts=3,
        backoff_factor=2.0,
        retry_on=[429, 500, 502, 503, 504],
    ),
    
    # Rate limiting
    rate_limit=RateLimitConfig(
        requests_per_minute=60,
        tokens_per_minute=90000,
    ),
    
    # Timeouts
    timeout=30.0,
    stream_timeout=300.0,
    
    # Logging
    log_requests=True,
    log_responses=True,
)

llm = LLMProvider.create("openai", config=config)
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev,all]"

# Run tests
pytest

# Run specific provider tests
pytest -k openai
pytest -k anthropic
```