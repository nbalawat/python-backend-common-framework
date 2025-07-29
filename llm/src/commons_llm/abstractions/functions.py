"""LLM function definitions."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class FunctionParameter(BaseModel):
    """Function parameter definition."""
    name: str
    type: str
    description: Optional[str] = None
    required: bool = True

class Function(BaseModel):
    """Function definition for LLM."""
    name: str
    description: str
    parameters: List[FunctionParameter] = []

class FunctionCall(BaseModel):
    """Function call from LLM."""
    name: str
    arguments: Dict[str, Any] = {}
