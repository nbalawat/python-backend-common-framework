"""Base agent implementation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from commons_llm import LLMProvider, Message
from commons_core.logging import get_logger

from .tools import Tool, ToolResult
from .role import Role

logger = get_logger(__name__)


@dataclass
class AgentConfig:
    """Agent configuration."""
    
    name: str
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    max_iterations: int = 10
    temperature: float = 0.7
    verbose: bool = False
    memory_enabled: bool = True
    allow_parallel_tools: bool = False
    

@dataclass
class AgentResponse:
    """Agent response."""
    
    content: str
    tool_calls: List[ToolResult] = field(default_factory=list)
    iterations: int = 0
    thinking: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """Check if response was successful."""
        return not any(t.error for t in self.tool_calls)


class Agent(ABC):
    """Base agent class."""
    
    def __init__(
        self,
        llm: LLMProvider,
        tools: Optional[List[Tool]] = None,
        config: Optional[AgentConfig] = None,
        role: Optional[Role] = None,
    ) -> None:
        self.llm = llm
        self.tools = tools or []
        self.config = config or AgentConfig(name="Agent")
        self.role = role
        self._tool_map = {tool.name: tool for tool in self.tools}
        
    @abstractmethod
    async def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Run agent on a task."""
        pass
        
    async def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Agent thinking/reasoning step."""
        prompt = self._build_thinking_prompt(task, context)
        
        messages = [
            Message(role="system", content=self._get_system_prompt()),
            Message(role="user", content=prompt),
        ]
        
        response = await self.llm.chat(messages)
        return response.content
        
    async def act(self, action: str, tool_name: str, tool_input: Any) -> ToolResult:
        """Execute action using tool."""
        if tool_name not in self._tool_map:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool '{tool_name}' not found",
            )
            
        tool = self._tool_map[tool_name]
        
        try:
            if self.config.verbose:
                logger.info(f"Executing tool: {tool_name}", input=tool_input)
                
            result = await tool.execute(tool_input)
            
            if self.config.verbose:
                logger.info(f"Tool result: {result}")
                
            return result
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}", tool=tool_name)
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
            )
            
    def _get_system_prompt(self) -> str:
        """Get system prompt for agent."""
        if self.config.system_prompt:
            return self.config.system_prompt
            
        prompt = f"You are {self.config.name}"
        
        if self.config.description:
            prompt += f", {self.config.description}"
            
        if self.role:
            prompt += f"\n\nRole: {self.role.description}"
            prompt += f"\nGoals: {', '.join(self.role.goals)}"
            
        if self.tools:
            prompt += "\n\nAvailable tools:"
            for tool in self.tools:
                prompt += f"\n- {tool.name}: {tool.description}"
                
        return prompt
        
    def _build_thinking_prompt(self, task: str, context: Optional[Dict[str, Any]]) -> str:
        """Build thinking prompt."""
        prompt = f"Task: {task}"
        
        if context:
            prompt += "\n\nContext:"
            for key, value in context.items():
                prompt += f"\n- {key}: {value}"
                
        prompt += "\n\nThink step by step about how to complete this task."
        
        return prompt
        
    def add_tool(self, tool: Tool) -> None:
        """Add tool to agent."""
        self.tools.append(tool)
        self._tool_map[tool.name] = tool
        
    def remove_tool(self, tool_name: str) -> None:
        """Remove tool from agent."""
        if tool_name in self._tool_map:
            tool = self._tool_map.pop(tool_name)
            self.tools.remove(tool)