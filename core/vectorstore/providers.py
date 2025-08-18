"""Vector store providers - imports from modular structure."""

# Import everything from the providers package
from .providers import (
    VectorStoreFactory,
    StudyBotInMemoryVectorStore,
    ChromaVectorStore,
    FAISSVectorStore
)

# Keep backwards compatibility
__all__ = [
    "VectorStoreFactory",
    "StudyBotInMemoryVectorStore",
    "ChromaVectorStore",
    "FAISSVectorStore"
]