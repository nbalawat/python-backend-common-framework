"""Agent memory implementation."""

from typing import Any, Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

class MemoryType(Enum):
    """Types of memory."""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    WORKING = "working"

@dataclass
class MemoryEntry:
    """Memory entry."""
    content: str
    memory_type: MemoryType
    timestamp: datetime
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if not isinstance(self.timestamp, datetime):
            self.timestamp = datetime.now()

class AgentMemory:
    """Agent memory system."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.memories: List[MemoryEntry] = []
    
    def add(self, content: str, memory_type: MemoryType = MemoryType.SHORT_TERM, **metadata) -> None:
        """Add memory entry."""
        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            timestamp=datetime.now(),
            metadata=metadata
        )
        
        self.memories.append(entry)
        
        # Limit size
        if len(self.memories) > self.max_size:
            self.memories = self.memories[-self.max_size:]
    
    def search(self, query: str, memory_type: Optional[MemoryType] = None) -> List[MemoryEntry]:
        """Search memories."""
        results = []
        for memory in self.memories:
            if memory_type and memory.memory_type != memory_type:
                continue
            if query.lower() in memory.content.lower():
                results.append(memory)
        return results
    
    def get_recent(self, n: int = 10, memory_type: Optional[MemoryType] = None) -> List[MemoryEntry]:
        """Get recent memories."""
        filtered = self.memories
        if memory_type:
            filtered = [m for m in self.memories if m.memory_type == memory_type]
        
        return sorted(filtered, key=lambda x: x.timestamp, reverse=True)[:n]
    
    def clear(self, memory_type: Optional[MemoryType] = None) -> None:
        """Clear memories."""
        if memory_type:
            self.memories = [m for m in self.memories if m.memory_type != memory_type]
        else:
            self.memories.clear()
    
    def count(self, memory_type: Optional[MemoryType] = None) -> int:
        """Count memories."""
        if memory_type:
            return len([m for m in self.memories if m.memory_type == memory_type])
        return len(self.memories)