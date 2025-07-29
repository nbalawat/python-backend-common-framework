"""Workflow state."""

from typing import Any, Dict

class WorkflowState:
    """Workflow execution state."""
    
    def __init__(self, status: str, current_step: str = None, context: Dict[str, Any] = None):
        self.status = status
        self.current_step = current_step
        self.context = context or {}
