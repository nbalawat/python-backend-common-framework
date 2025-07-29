"""Agent tool implementation."""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class ToolParameter:
    """Tool parameter definition."""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None

@dataclass
class ToolResult:
    """Tool execution result."""
    success: bool
    result: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class Tool:
    """Agent tool."""
    
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        function: Optional[Callable] = None
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function or self._default_function
    
    def _default_function(self, **kwargs) -> ToolResult:
        """Default function implementation."""
        return ToolResult(
            success=True,
            result=f"Tool {self.name} executed with params: {kwargs}",
            metadata={"tool": self.name, "params": kwargs}
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool."""
        try:
            if hasattr(self.function, '__call__'):
                result = self.function(**kwargs)
                if hasattr(result, '__await__'):
                    result = await result
                
                if isinstance(result, ToolResult):
                    return result
                else:
                    return ToolResult(success=True, result=result)
            else:
                return ToolResult(success=False, error="Tool function not callable")
                
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate parameters."""
        # Basic validation - could be enhanced
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }