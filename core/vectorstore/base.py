"""Base classes for vector stores."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Callable, Tuple
import structlog
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

logger = structlog.get_logger(__name__)


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    
    def __init__(self, embeddings: Embeddings, **kwargs):
        self.embeddings = embeddings
        self.kwargs = kwargs
        self._store: Optional[VectorStore] = None
    
    @abstractmethod
    def initialize(self) -> VectorStore:
        """Initialize the vector store.
        
        Returns:
            Initialized vector store instance
        """
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            **kwargs: Additional arguments
            
        Returns:
            List of document IDs
        """
        pass
    
    @abstractmethod
    def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filter: Optional[Callable[[Document], bool]] = None,
        **kwargs
    ) -> List[Document]:
        """Search for similar documents.
        
        Args:
            query: Query string
            k: Number of documents to return
            filter: Optional filter function
            **kwargs: Additional search arguments
            
        Returns:
            List of similar documents
        """
        pass
    
    @abstractmethod
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 5,
        filter: Optional[Callable[[Document], bool]] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents with scores.
        
        Args:
            query: Query string
            k: Number of documents to return
            filter: Optional filter function
            **kwargs: Additional search arguments
            
        Returns:
            List of (document, score) tuples
        """
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """Delete documents by IDs.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def persist(self) -> bool:
        """Persist the vector store.
        
        Returns:
            True if successful
        """
        pass
    
    def get_store(self) -> VectorStore:
        """Get the underlying vector store instance."""
        if self._store is None:
            self._store = self.initialize()
        return self._store


class VectorStoreManager:
    """Manager for vector store operations."""
    
    def __init__(self, vector_store: BaseVectorStore):
        self.vector_store = vector_store
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def add_documents_batch(
        self, 
        documents: List[Document], 
        batch_size: int = 100
    ) -> List[str]:
        """Add documents in batches.
        
        Args:
            documents: List of documents to add
            batch_size: Size of each batch
            
        Returns:
            List of all document IDs
        """
        all_ids = []
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            try:
                ids = self.vector_store.add_documents(batch)
                all_ids.extend(ids)
                self.logger.info(
                    "Added document batch", 
                    batch=batch_num, 
                    total_batches=total_batches,
                    batch_size=len(batch)
                )
            except Exception as e:
                self.logger.error(
                    "Failed to add document batch", 
                    batch=batch_num, 
                    error=str(e)
                )
                raise
        
        self.logger.info("Added all documents", total_documents=len(documents))
        return all_ids
    
    def search_with_filter(
        self,
        query: str,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        **kwargs
    ) -> List[Document]:
        """Search with metadata filtering and score threshold.
        
        Args:
            query: Query string
            k: Number of documents to return
            metadata_filter: Dictionary of metadata key-value pairs to filter by
            score_threshold: Minimum similarity score threshold
            **kwargs: Additional search arguments
            
        Returns:
            List of filtered documents
        """
        # Create filter function from metadata filter
        filter_func = None
        if metadata_filter:
            def filter_func(doc: Document) -> bool:
                return all(
                    doc.metadata.get(key) == value 
                    for key, value in metadata_filter.items()
                )
        
        # Get results with scores if threshold is specified
        if score_threshold is not None:
            results_with_scores = self.vector_store.similarity_search_with_score(
                query, k=k, filter=filter_func, **kwargs
            )
            # Filter by score threshold
            filtered_results = [
                doc for doc, score in results_with_scores 
                if score >= score_threshold
            ]
            self.logger.debug(
                "Filtered search results by score", 
                original_count=len(results_with_scores),
                filtered_count=len(filtered_results),
                threshold=score_threshold
            )
            return filtered_results
        else:
            return self.vector_store.similarity_search(
                query, k=k, filter=filter_func, **kwargs
            )
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            store = self.vector_store.get_store()
            # This is a generic implementation; specific stores may override
            return {
                "type": type(store).__name__,
                "initialized": True,
            }
        except Exception as e:
            self.logger.error("Failed to get collection stats", error=str(e))
            return {
                "type": "unknown",
                "initialized": False,
                "error": str(e)
            }
