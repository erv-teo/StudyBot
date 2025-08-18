"""Vector store provider factory."""

from typing import Any, Dict
import structlog
from langchain_core.embeddings import Embeddings

from ..base import BaseVectorStore
from .inmemory import StudyBotInMemoryVectorStore
from .chroma import ChromaVectorStore
from .faiss import FAISSVectorStore

logger = structlog.get_logger(__name__)


class VectorStoreFactory:
    """Factory for creating vector store providers."""
    
    @staticmethod
    def create_store(provider_type: str, embeddings: Embeddings, **kwargs) -> BaseVectorStore:
        """Create a vector store of the specified type.
        
        Args:
            provider_type: Type of vector store ("chroma", "faiss", "inmemory")
            embeddings: Embeddings instance
            **kwargs: Arguments for the vector store
            
        Returns:
            Vector store instance
            
        Raises:
            ValueError: If provider type is not supported
        """
        if provider_type == "chroma":
            return ChromaVectorStore(embeddings=embeddings, **kwargs)
        elif provider_type == "faiss":
            return FAISSVectorStore(embeddings=embeddings, **kwargs)
        elif provider_type == "inmemory":
            return StudyBotInMemoryVectorStore(embeddings=embeddings, **kwargs)
        else:
            raise ValueError(f"Unsupported vector store provider type: {provider_type}")
    
    @staticmethod
    def get_recommended_store(use_case: str, embeddings: Embeddings, **kwargs) -> BaseVectorStore:
        """Get recommended vector store for use case.
        
        Args:
            use_case: Use case ("development", "production", "local", "scalable")
            embeddings: Embeddings instance
            **kwargs: Arguments for the vector store
            
        Returns:
            Recommended vector store
        """
        if use_case == "development":
            return StudyBotInMemoryVectorStore(embeddings=embeddings, **kwargs)
        elif use_case == "production":
            return ChromaVectorStore(embeddings=embeddings, **kwargs)
        elif use_case == "local":
            return ChromaVectorStore(embeddings=embeddings, **kwargs)
        elif use_case == "scalable":
            return ChromaVectorStore(embeddings=embeddings, **kwargs)  # Could be extended for cloud solutions
        else:
            return ChromaVectorStore(embeddings=embeddings, **kwargs)
    
    @staticmethod
    def get_available_stores() -> Dict[str, Dict[str, Any]]:
        """Get information about available vector stores.
        
        Returns:
            Dictionary with vector store information
        """
        return {
            "chroma": {
                "name": "ChromaDB",
                "description": "Persistent vector database with good performance",
                "persistence": True,
                "scalability": "Medium",
                "use_cases": ["development", "production", "local"]
            },
            "faiss": {
                "name": "FAISS",
                "description": "Facebook's similarity search library, fast but limited features",
                "persistence": True,
                "scalability": "High",
                "use_cases": ["production", "high-performance"]
            },
            "inmemory": {
                "name": "In-Memory",
                "description": "Simple in-memory storage, good for testing",
                "persistence": False,
                "scalability": "Low",
                "use_cases": ["development", "testing", "prototyping"]
            }
        }
