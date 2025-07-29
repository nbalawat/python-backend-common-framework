"""Workflow activity."""

from typing import Any, Dict

class Activity:
    """Workflow activity."""
    
    def __init__(self, name: str, activity_type: str, parameters: Dict[str, Any] = None):
        self.name = name
        self.activity_type = activity_type
        self.parameters = parameters or {}
