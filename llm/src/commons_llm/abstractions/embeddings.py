"""LLM embeddings abstractions."""

from typing import List, Optional
from pydantic import BaseModel

class EmbeddingResponse(BaseModel):
    """Embedding response."""
    embeddings: List[List[float]]
    model: str
    usage: Optional[dict] = None

class EmbeddingProvider:
    """Embedding provider interface."""
    
    def __init__(self, model: str = "text-embedding-ada-002"):
        self.model = model
    
    async def embed(self, texts: List[str]) -> EmbeddingResponse:
        """Generate embeddings."""
        return EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3] for _ in texts],
            model=self.model
        )
