"""Workflow step definition."""

from typing import Any, Dict

class WorkflowStep:
    """Workflow step."""
    
    def __init__(self, name: str, action: str, parameters: Dict[str, Any] = None):
        self.name = name
        self.action = action
        self.parameters = parameters or {}
