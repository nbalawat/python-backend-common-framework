# Commons LLM

Unified LLM provider abstractions with enterprise-grade features for building production AI applications. Supports all major LLM providers with consistent APIs, advanced prompting, function calling, and cost optimization.

## Installation

```bash
pip install commons-llm
```

## Features

- **Multi-Provider Support**: OpenAI, Anthropic, Google, AWS Bedrock, Azure OpenAI, Cohere, Mistral, and local models
- **Unified Interface**: Consistent API across all providers with async/await support
- **Advanced Prompting**: Template system, few-shot learning, chain-of-thought reasoning
- **Function Calling**: Tool integration with automatic schema validation  
- **Stream Processing**: Real-time streaming responses with token-by-token output
- **Cost Management**: Usage tracking, budget controls, and optimization suggestions
- **Caching Layer**: Response caching with TTL and semantic similarity matching
- **Fallback Strategies**: Multi-provider failover and load balancing
- **A/B Testing**: Model comparison and performance evaluation
- **Framework Integration**: LangChain, LlamaIndex, and custom framework adapters
- **Enterprise Features**: Rate limiting, retries, logging, monitoring, and security

## Quick Start

```python
import asyncio
from commons_llm import LLMProvider, Message

async def main():
    # Initialize provider
    llm = LLMProvider.create(
        provider="openai",
        model="gpt-4",
        api_key="your-api-key"
    )
    
    # Simple completion
    response = await llm.complete("What is the capital of France?")
    print(f"Answer: {response.content}")
    
    # Chat with conversation
    messages = [
        Message(role="system", content="You are a helpful AI assistant."),
        Message(role="user", content="Explain quantum computing simply.")
    ]
    
    response = await llm.chat(messages)
    print(f"Explanation: {response.content}")
    
    # Streaming response
    print("Streaming story:")
    async for chunk in llm.stream("Tell me a short story about AI"):
        print(chunk.content, end="", flush=True)
    
    await llm.close()

asyncio.run(main())
```

## Detailed Usage Examples

### Provider Configuration and Management

#### Multi-Provider Setup
```python
import asyncio
from commons_llm import LLMProvider, LLMConfig, Message
from commons_llm.providers import (
    OpenAIProvider, AnthropicProvider, GoogleProvider,
    BedrockProvider, AzureOpenAIProvider, CohereProvider
)
from datetime import datetime
import json

async def demonstrate_provider_setup():
    """Demonstrate comprehensive provider configuration."""
    
    print("=== LLM Provider Configuration ===")
    
    # 1. OpenAI Configuration
    openai_config = LLMConfig(
        api_key="sk-...",
        temperature=0.7,
        max_tokens=2000,
        top_p=0.9,
        frequency_penalty=0.1,
        presence_penalty=0.1,
        timeout=30.0,
        retry_attempts=3,
        retry_backoff=2.0
    )
    
    openai_llm = LLMProvider.create(
        provider="openai",
        model="gpt-4-turbo-preview",
        config=openai_config
    )
    
    print(f"✓ OpenAI configured: {openai_llm.model}")
    
    # 2. Anthropic Configuration
    anthropic_config = LLMConfig(
        api_key="sk-ant-...",
        temperature=0.3,
        max_tokens=4000,
        timeout=60.0,
        system_prompt="You are Claude, an AI assistant created by Anthropic."
    )
    
    anthropic_llm = LLMProvider.create(
        provider="anthropic",
        model="claude-3-opus-20240229",
        config=anthropic_config
    )
    
    print(f"✓ Anthropic configured: {anthropic_llm.model}")
    
    # 3. Google Gemini Configuration
    google_config = LLMConfig(
        api_key="AI...",
        temperature=0.5,
        max_tokens=1000,
        candidate_count=1,
        safety_settings={
            "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE"
        }
    )
    
    google_llm = LLMProvider.create(
        provider="google",
        model="gemini-pro",
        config=google_config
    )
    
    print(f"✓ Google configured: {google_llm.model}")
    
    # 4. AWS Bedrock Configuration
    bedrock_config = LLMConfig(
        aws_access_key_id="AKIA...",
        aws_secret_access_key="...",
        aws_region="us-east-1",
        temperature=0.6,
        max_tokens=2048
    )
    
    bedrock_llm = LLMProvider.create(
        provider="bedrock",
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        config=bedrock_config
    )
    
    print(f"✓ Bedrock configured: {bedrock_llm.model}")
    
    # 5. Azure OpenAI Configuration
    azure_config = LLMConfig(
        api_key="...",
        azure_endpoint="https://your-resource.openai.azure.com/",
        api_version="2023-12-01-preview",
        temperature=0.7,
        max_tokens=1500
    )
    
    azure_llm = LLMProvider.create(
        provider="azure_openai",
        model="gpt-4",  # Deployment name in Azure
        config=azure_config
    )
    
    print(f"✓ Azure OpenAI configured: {azure_llm.model}")
    
    # 6. Local Model Configuration
    from commons_llm.providers import LocalLLMProvider
    
    local_config = LLMConfig(
        model_path="./models/llama-2-7b-chat.gguf",
        device="cuda",  # or "cpu", "mps"
        max_tokens=512,
        temperature=0.8,
        context_length=4096,
        gpu_layers=32,  # For GPU acceleration
        batch_size=8
    )
    
    try:
        local_llm = LocalLLMProvider(
            model_name="llama-2-7b-chat",
            config=local_config
        )
        print(f"✓ Local model configured: {local_llm.model}")
    except Exception as e:
        print(f"⚠ Local model not available: {e}")
    
    # 7. Provider Comparison
    providers = {
        "OpenAI GPT-4": openai_llm,
        "Anthropic Claude-3": anthropic_llm,
        "Google Gemini": google_llm,
        "AWS Bedrock": bedrock_llm,
        "Azure OpenAI": azure_llm
    }
    
    test_prompt = "Explain the concept of machine learning in exactly 50 words."
    
    print(f"\nTesting all providers with prompt: '{test_prompt}'")
    
    for provider_name, llm in providers.items():
        try:
            start_time = datetime.now()
            response = await llm.complete(test_prompt)
            duration = (datetime.now() - start_time).total_seconds()
            
            word_count = len(response.content.split())
            
            print(f"\n{provider_name}:")
            print(f"  Response time: {duration:.2f}s")
            print(f"  Word count: {word_count}")
            print(f"  Cost: ${response.usage.cost:.4f}" if response.usage else "  Cost: N/A")
            print(f"  Content: {response.content[:100]}...")
            
        except Exception as e:
            print(f"\n{provider_name}: ⚠ Error - {e}")
    
    # Cleanup
    for llm in providers.values():
        await llm.close()

# Advanced message handling
async def demonstrate_conversation_management():
    """Demonstrate advanced conversation and message handling."""
    
    print("\n=== Advanced Conversation Management ===")
    
    llm = LLMProvider.create(
        provider="openai",
        model="gpt-4",
        temperature=0.7
    )
    
    # 1. Multi-turn conversation with context
    conversation = [
        Message(
            role="system",
            content="You are an expert Python developer and code reviewer. Provide detailed, practical advice."
        )
    ]
    
    questions = [
        "What are the key principles of writing clean Python code?",
        "Can you show me an example of a well-structured Python class?",
        "How would you refactor this class to follow SOLID principles?",
        "What testing strategies would you recommend for this code?"
    ]
    
    print("Multi-turn conversation:")
    
    for i, question in enumerate(questions, 1):
        # Add user message
        conversation.append(Message(role="user", content=question))
        
        print(f"\nTurn {i}: {question}")
        
        # Get response
        response = await llm.chat(conversation, max_tokens=300)
        
        # Add assistant response to conversation
        conversation.append(response.to_message())
        
        print(f"Assistant: {response.content[:150]}...")
        
        # Track conversation metrics
        total_tokens = sum(msg.token_count for msg in conversation if hasattr(msg, 'token_count'))
        print(f"Conversation tokens: {total_tokens}")
    
    # 2. Message preprocessing and postprocessing
    from commons_llm.processors import MessageProcessor, ContentFilter
    
    class CustomMessageProcessor(MessageProcessor):
        async def preprocess_messages(self, messages):
            """Clean and enhance messages before sending."""
            processed = []
            
            for msg in messages:
                # Remove excessive whitespace
                content = ' '.join(msg.content.split())
                
                # Add context hints
                if msg.role == "user":
                    content = f"[Context: Technical discussion] {content}"
                
                processed.append(Message(
                    role=msg.role,
                    content=content,
                    metadata=getattr(msg, 'metadata', {})
                ))
            
            return processed
        
        async def postprocess_response(self, response):
            """Enhance response after receiving from LLM."""
            # Add confidence score based on response characteristics
            confidence = self._calculate_confidence(response.content)
            
            response.metadata = response.metadata or {}
            response.metadata['confidence'] = confidence
            response.metadata['processed_at'] = datetime.now().isoformat()
            
            return response
        
        def _calculate_confidence(self, content):
            """Simple confidence calculation based on content features."""
            # More detailed responses generally indicate higher confidence
            word_count = len(content.split())
            has_examples = 'example' in content.lower() or 'for instance' in content.lower()
            has_caveats = any(word in content.lower() for word in ['however', 'but', 'although'])
            
            confidence = min(0.5 + (word_count / 200), 0.95)
            if has_examples:
                confidence += 0.1
            if has_caveats:
                confidence += 0.05  # Nuanced responses are often more reliable
            
            return round(confidence, 2)
    
    # Use processor
    processor = CustomMessageProcessor()
    processed_llm = processor.wrap(llm)
    
    test_messages = [
        Message(role="system", content="You are a helpful coding assistant."),
        Message(role="user", content="   How   do   I   handle   exceptions   in   Python?   ")
    ]
    
    response = await processed_llm.chat(test_messages)
    
    print(f"\nProcessed response:")
    print(f"Content: {response.content[:100]}...")
    print(f"Confidence: {response.metadata.get('confidence', 'N/A')}")
    print(f"Processed at: {response.metadata.get('processed_at', 'N/A')}")
    
    # 3. Content filtering and safety
    content_filter = ContentFilter(
        blocked_patterns=[
            r'\b(password|secret|key)\s*[:=]\s*\w+',  # Sensitive data
            r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card numbers
        ],
        allowed_topics=['programming', 'technology', 'education'],
        max_toxicity_score=0.7
    )
    
    safe_messages = [
        "How do I implement authentication in my web app?",
        "My credit card number is 4532-1234-5678-9012",  # Should be filtered
        "Can you help me with Python programming?"
    ]
    
    print("\nContent filtering:")
    for msg_content in safe_messages:
        is_safe, reason = await content_filter.check_message(msg_content)
        status = "✓" if is_safe else "⚠"
        print(f"  {status} '{msg_content[:30]}...': {reason or 'Safe'}")
    
    await llm.close()

# Streaming and real-time responses
async def demonstrate_streaming():
    """Demonstrate streaming responses and real-time processing."""
    
    print("\n=== Streaming and Real-time Responses ===")
    
    llm = LLMProvider.create(
        provider="openai",
        model="gpt-4",
        temperature=0.8
    )
    
    # 1. Basic streaming
    print("\nBasic streaming:")
    prompt = "Write a detailed explanation of how neural networks work, including the mathematical concepts."
    
    full_response = ""
    word_count = 0
    start_time = datetime.now()
    
    async for chunk in llm.stream(prompt):
        content = chunk.content
        full_response += content
        word_count += len(content.split())
        
        print(content, end="", flush=True)
        
        # Show progress periodically
        if word_count % 50 == 0 and word_count > 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            wps = word_count / elapsed if elapsed > 0 else 0
            print(f" [{word_count} words, {wps:.1f} wps]", end="", flush=True)
    
    duration = (datetime.now() - start_time).total_seconds()
    print(f"\n\nStreaming completed: {word_count} words in {duration:.2f}s")
    
    # 2. Streaming with real-time processing
    print("\n\nStreaming with real-time analysis:")
    
    from commons_llm.streaming import StreamProcessor
    
    class RealTimeAnalyzer(StreamProcessor):
        def __init__(self):
            self.sentence_count = 0
            self.current_sentence = ""
            self.topics = set()
            self.sentiment_scores = []
        
        async def process_chunk(self, chunk):
            """Process each streaming chunk in real-time."""
            content = chunk.content
            self.current_sentence += content
            
            # Detect sentence completion
            if any(punct in content for punct in ['.', '!', '?']):
                await self._analyze_sentence(self.current_sentence.strip())
                self.current_sentence = ""
            
            return chunk
        
        async def _analyze_sentence(self, sentence):
            """Analyze completed sentence."""
            if len(sentence) < 10:  # Skip very short sentences
                return
            
            self.sentence_count += 1
            
            # Simple topic extraction (in practice, use NLP libraries)
            tech_keywords = {
                'neural', 'network', 'algorithm', 'machine', 'learning',
                'artificial', 'intelligence', 'data', 'model', 'training'
            }
            
            sentence_lower = sentence.lower()
            found_topics = {word for word in tech_keywords if word in sentence_lower}
            self.topics.update(found_topics)
            
            # Simple sentiment (positive indicators)
            positive_words = {'good', 'excellent', 'effective', 'powerful', 'successful'}
            sentiment = len([w for w in positive_words if w in sentence_lower])
            self.sentiment_scores.append(sentiment)
            
            # Real-time feedback
            if self.sentence_count % 5 == 0:
                avg_sentiment = sum(self.sentiment_scores) / len(self.sentiment_scores)
                print(f"\n[Analysis: {self.sentence_count} sentences, {len(self.topics)} topics, sentiment: {avg_sentiment:.1f}]")
    
    analyzer = RealTimeAnalyzer()
    
    prompt = "Explain the evolution of artificial intelligence and its impact on modern technology."
    
    print(f"Analyzing: {prompt}")
    print("Response with real-time analysis:")
    
    async for chunk in llm.stream(prompt):
        processed_chunk = await analyzer.process_chunk(chunk)
        print(processed_chunk.content, end="", flush=True)
    
    print(f"\n\nFinal analysis:")
    print(f"  Sentences: {analyzer.sentence_count}")
    print(f"  Topics found: {', '.join(sorted(analyzer.topics))}")
    print(f"  Average sentiment: {sum(analyzer.sentiment_scores) / len(analyzer.sentiment_scores) if analyzer.sentiment_scores else 0:.2f}")
    
    # 3. Streaming with interruption and control
    print("\n\nStreaming with interruption control:")
    
    from commons_llm.streaming import StreamController
    
    controller = StreamController()
    
    async def monitor_stream():
        """Monitor stream and interrupt if needed."""
        await asyncio.sleep(3)  # Let it stream for 3 seconds
        print("\n[INTERRUPTING STREAM]")
        controller.interrupt("User requested interruption")
    
    # Start monitoring task
    monitor_task = asyncio.create_task(monitor_stream())
    
    prompt = "Write a very long story about a robot learning to understand human emotions."
    interrupted_response = ""
    
    try:
        async for chunk in llm.stream(prompt, controller=controller):
            interrupted_response += chunk.content
            print(chunk.content, end="", flush=True)
    
    except asyncio.CancelledError:
        print("\nStream was interrupted successfully")
    
    await monitor_task
    
    print(f"Response length before interruption: {len(interrupted_response)} characters")
    
    await llm.close()

# Run demonstrations
asyncio.run(demonstrate_provider_setup())
asyncio.run(demonstrate_conversation_management())
asyncio.run(demonstrate_streaming())
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