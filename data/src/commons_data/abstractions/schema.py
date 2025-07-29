"""Database schema abstraction."""

from typing import Any, Dict, List, Optional

class Column:
    """Database column."""
    
    def __init__(self, name: str, column_type: str, nullable: bool = True):
        self.name = name
        self.column_type = column_type
        self.nullable = nullable

class Index:
    """Database index."""
    
    def __init__(self, name: str, columns: List[str], unique: bool = False):
        self.name = name
        self.columns = columns
        self.unique = unique

class ForeignKey:
    """Foreign key constraint."""
    
    def __init__(self, column: str, references: str):
        self.column = column
        self.references = references

class Table:
    """Database table."""
    
    def __init__(self, name: str, columns: List[Column] = None):
        self.name = name
        self.columns = columns or []
        self.indexes = []
        self.foreign_keys = []
