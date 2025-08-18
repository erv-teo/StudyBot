"""In-memory vector store implementation."""

from typing import List, Optional, Callable
import structlog
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, InMemoryVectorStore

from ..base import BaseVectorStore

logger = structlog.get_logger(__name__)


class StudyBotInMemoryVectorStore(BaseVectorStore):
    """In-memory vector store implementation."""
    
    def __init__(self, embeddings: Embeddings, **kwargs):
        super().__init__(embeddings, **kwargs)
        self._documents: List[Document] = []
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def initialize(self) -> VectorStore:
        """Initialize the in-memory vector store."""
        try:
            store = InMemoryVectorStore(self.embeddings)
            self.logger.info("Initialized in-memory vector store")
            return store
        except Exception as e:
            self.logger.error("Failed to initialize in-memory vector store", error=str(e))
            raise
    
    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """Add documents to the vector store."""
        try:
            store = self.get_store()
            ids = store.add_documents(documents, **kwargs)
            self._documents.extend(documents)
            self.logger.debug("Added documents to in-memory store", count=len(documents))
            return ids
        except Exception as e:
            self.logger.error("Failed to add documents to in-memory store", error=str(e))
            raise
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filter: Optional[Callable[[Document], bool]] = None,
        **kwargs
    ) -> List[Document]:
        """Search for similar documents."""
        try:
            store = self.get_store()
            results = store.similarity_search(query, k=k, filter=filter, **kwargs)
            self.logger.debug("Performed similarity search", query_length=len(query), results_count=len(results))
            return results
        except Exception as e:
            self.logger.error("Failed to perform similarity search", error=str(e))
            raise
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 5,
        filter: Optional[Callable[[Document], bool]] = None,
        **kwargs
    ) -> List[tuple]:
        """Search for similar documents with scores."""
        try:
            store = self.get_store()
            results = store.similarity_search_with_score(query, k=k, filter=filter, **kwargs)
            self.logger.debug("Performed similarity search with scores", 
                            query_length=len(query), 
                            results_count=len(results))
            return results
        except Exception as e:
            self.logger.error("Failed to perform similarity search with scores", error=str(e))
            raise
    
    def delete(self, ids: List[str], **kwargs) -> bool:
        """Delete documents by IDs."""
        try:
            store = self.get_store()
            if hasattr(store, 'delete'):
                store.delete(ids, **kwargs)
                self.logger.debug("Deleted documents", count=len(ids))
                return True
            else:
                self.logger.warning("Delete operation not supported by in-memory vector store")
                return False
        except Exception as e:
            self.logger.error("Failed to delete documents", error=str(e))
            raise
    
    def get_stats(self) -> dict:
        """Get vector store statistics."""
        try:
            return {
                "type": "in-memory",
                "document_count": len(self._documents),
                "initialized": self._store is not None
            }
        except Exception as e:
            self.logger.error("Failed to get stats", error=str(e))
            return {"error": str(e)}
    
    def clear(self) -> bool:
        """Clear all documents from the vector store."""
        try:
            self._documents = []
            self._store = None  # Force re-initialization
            self.logger.info("Cleared in-memory vector store")
            return True
        except Exception as e:
            self.logger.error("Failed to clear vector store", error=str(e))
            return False
