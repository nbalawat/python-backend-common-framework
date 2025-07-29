"""Query abstraction."""

from typing import Any, Dict, List, Optional

class Condition:
    """Query condition."""
    
    def __init__(self, field: str, operator: str, value: Any):
        self.field = field
        self.operator = operator
        self.value = value

class Query:
    """Database query."""
    
    def __init__(self, table: str, conditions: List[Condition] = None):
        self.table = table
        self.conditions = conditions or []

class QueryBuilder:
    """Query builder."""
    
    def __init__(self, table: str):
        self.table = table
        self.query = Query(table)
    
    def where(self, field: str, operator: str, value: Any):
        """Add where condition."""
        self.query.conditions.append(Condition(field, operator, value))
        return self
    
    def build(self) -> Query:
        """Build the query."""
        return self.query
