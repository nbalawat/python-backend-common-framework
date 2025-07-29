"""Base LLM abstractions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Union
from enum import Enum

from commons_core.types import BaseModel


class Role(str, Enum):
    """Message roles."""
    
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


@dataclass
class Message:
    """Chat message."""
    
    role: Union[Role, str]
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "role": self.role.value if isinstance(self.role, Role) else self.role,
            "content": self.content,
        }
        if self.name:
            data["name"] = self.name
        if self.function_call:
            data["function_call"] = self.function_call
        return data


@dataclass
class CompletionOptions:
    """Completion options."""
    
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

@dataclass
class TokenUsage:
    """Token usage statistics."""
    
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

@dataclass
class ChatRequest:
    """Chat completion request."""
    
    messages: List[Message]
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    functions: Optional[List[Dict[str, Any]]] = None

@dataclass 
class ChatResponse:
    """Chat completion response."""
    
    message: Message
    model: str
    usage: TokenUsage
    finish_reason: str = "stop"
    stop: Optional[List[str]] = None
    n: int = 1
    stream: bool = False
    logprobs: Optional[int] = None
    echo: bool = False
    user: Optional[str] = None
    

@dataclass
class Response:
    """LLM response."""
    
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    function_calls: List["FunctionCall"] = field(default_factory=list)
    
    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self.usage.get("total_tokens", 0)
        
    @property
    def prompt_tokens(self) -> int:
        """Get prompt tokens."""
        return self.usage.get("prompt_tokens", 0)
        
    @property
    def completion_tokens(self) -> int:
        """Get completion tokens."""
        return self.usage.get("completion_tokens", 0)
        
    def to_message(self) -> Message:
        """Convert response to message."""
        return Message(
            role=Role.ASSISTANT,
            content=self.content,
            function_call=self.function_calls[0].to_dict() if self.function_calls else None,
        )


@dataclass
class StreamResponse:
    """Streaming response chunk."""
    
    content: str
    is_final: bool = False
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


class LLMProvider(ABC):
    """Abstract LLM provider interface."""
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.kwargs = kwargs
        
    @abstractmethod
    async def complete(
        self,
        prompt: str,
        options: Optional[CompletionOptions] = None,
        **kwargs: Any,
    ) -> Response:
        """Complete a prompt."""
        pass
        
    @abstractmethod
    async def chat(
        self,
        messages: List[Message],
        options: Optional[CompletionOptions] = None,
        functions: Optional[List["Function"]] = None,
        **kwargs: Any,
    ) -> Response:
        """Chat completion."""
        pass
        
    @abstractmethod
    async def stream(
        self,
        prompt: str,
        options: Optional[CompletionOptions] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamResponse]:
        """Stream completion."""
        pass
        
    @abstractmethod
    async def stream_chat(
        self,
        messages: List[Message],
        options: Optional[CompletionOptions] = None,
        functions: Optional[List["Function"]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamResponse]:
        """Stream chat completion."""
        pass
        
    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass
        
    @classmethod
    def create(
        cls,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> "LLMProvider":
        """Factory method to create provider."""
        # This would be implemented in factory module
        raise NotImplementedError()