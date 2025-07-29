"""Workflow definition."""

from typing import List, Dict, Any

class Workflow:
    """Workflow definition."""
    
    def __init__(self, name: str, steps: List[Any] = None):
        self.name = name
        self.steps = steps or []
