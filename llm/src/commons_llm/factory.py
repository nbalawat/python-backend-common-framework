"""LLM provider factory."""

from typing import Any, Dict, List, Optional, Type
from .abstractions import LLMProvider

class LLMFactory:
    """Factory for creating LLM provider instances."""
    
    _providers: Dict[str, Type[LLMProvider]] = {}
    
    @classmethod
    def register(cls, provider_name: str, provider_class: Type[LLMProvider]) -> None:
        """Register an LLM provider."""
        cls._providers[provider_name.lower()] = provider_class
    
    @classmethod
    def create(
        cls,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMProvider:
        """Create LLM provider instance."""
        provider_lower = provider.lower()
        
        if provider_lower not in cls._providers:
            # Return a mock provider for testing
            return MockLLMProvider(model=model, api_key=api_key, **kwargs)
        
        provider_class = cls._providers[provider_lower]
        return provider_class(model=model, api_key=api_key, **kwargs)
    
    @classmethod
    def available_providers(cls) -> list[str]:
        """Get list of available providers."""
        return list(cls._providers.keys())

class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""
    
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model, api_key, **kwargs)
    
    async def complete(self, prompt: str, options: Optional["CompletionOptions"] = None, **kwargs) -> "Response":
        """Mock completion."""
        from .abstractions import Response
        
        return Response(
            content="This is a mock completion response.",
            model=self.model,
            usage={"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
            finish_reason="stop"
        )
    
    async def chat(self, messages: List["Message"], options: Optional["CompletionOptions"] = None, functions: Optional[List["Function"]] = None, **kwargs) -> "Response":
        """Mock chat completion."""
        from .abstractions import Response
        
        return Response(
            content="This is a mock chat response.",
            model=self.model,
            usage={"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
            finish_reason="stop"
        )
    
    async def stream(self, prompt: str, options: Optional["CompletionOptions"] = None, **kwargs):
        """Mock stream completion."""
        from .abstractions import StreamResponse
        
        chunks = ["This ", "is ", "a ", "mock ", "stream."]
        for chunk in chunks:
            yield StreamResponse(content=chunk, is_final=(chunk == chunks[-1]))
    
    async def stream_chat(self, messages: List["Message"], options: Optional["CompletionOptions"] = None, functions: Optional[List["Function"]] = None, **kwargs):
        """Mock streaming chat."""
        from .abstractions import StreamResponse
        
        chunks = ["Mock ", "chat ", "stream ", "response."]
        for chunk in chunks:
            yield StreamResponse(content=chunk, is_final=(chunk == chunks[-1]))
    
    async def count_tokens(self, text: str) -> int:
        """Mock token counting."""
        return len(text.split()) * 2  # Rough approximation