"""ChromaDB vector store implementation."""

from pathlib import Path
from typing import List, Optional, Dict, Any
import structlog
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from ..base import BaseVectorStore

logger = structlog.get_logger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB vector store implementation."""
    
    def __init__(self, 
                 embeddings: Embeddings,
                 persist_directory: str = "./data/vectorstore",
                 collection_name: str = "studybot_documents",
                 **kwargs):
        super().__init__(embeddings, **kwargs)
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def initialize(self) -> VectorStore:
        """Initialize the ChromaDB vector store."""
        try:
            from langchain_chroma import Chroma
            
            # Ensure persist directory exists
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            
            init_kwargs = {
                "embedding_function": self.embeddings,
                "persist_directory": self.persist_directory,
                "collection_name": self.collection_name,
            }
            
            # Add any additional kwargs
            init_kwargs.update(self.kwargs)
            
            store = Chroma(**init_kwargs)
            self.logger.info("Initialized ChromaDB vector store", 
                           persist_directory=self.persist_directory,
                           collection_name=self.collection_name)
            return store
            
        except ImportError:
            self.logger.error("langchain_chroma not installed. Install with: pip install langchain-chroma")
            raise
        except Exception as e:
            self.logger.error("Failed to initialize ChromaDB vector store", error=str(e))
            raise
    
    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """Add documents to the vector store."""
        try:
            store = self.get_store()
            ids = store.add_documents(documents, **kwargs)
            self.logger.debug("Added documents to ChromaDB", count=len(documents))
            return ids
        except Exception as e:
            self.logger.error("Failed to add documents to ChromaDB", error=str(e))
            raise
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """Search for similar documents."""
        try:
            store = self.get_store()
            results = store.similarity_search(query, k=k, filter=filter, **kwargs)
            self.logger.debug("Performed similarity search", 
                            query_length=len(query), 
                            results_count=len(results))
            return results
        except Exception as e:
            self.logger.error("Failed to perform similarity search", error=str(e))
            raise
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
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
            store.delete(ids, **kwargs)
            self.logger.debug("Deleted documents from ChromaDB", count=len(ids))
            return True
        except Exception as e:
            self.logger.error("Failed to delete documents from ChromaDB", error=str(e))
            raise
    
    def get_stats(self) -> dict:
        """Get vector store statistics."""
        try:
            store = self.get_store()
            
            # Try to get collection info
            collection_info = {}
            if hasattr(store, '_collection'):
                try:
                    collection_info = {
                        "document_count": store._collection.count(),
                        "collection_name": self.collection_name
                    }
                except Exception:
                    collection_info = {"collection_name": self.collection_name}
            
            return {
                "type": "chroma",
                "persist_directory": self.persist_directory,
                "initialized": self._store is not None,
                **collection_info
            }
        except Exception as e:
            self.logger.error("Failed to get ChromaDB stats", error=str(e))
            return {"error": str(e)}
    
    def persist(self) -> None:
        """Persist the current state of the vector store."""
        try:
            store = self.get_store()
            if store is not None:
                store.save_local(self.persist_directory, self.collection_name)
                self.logger.info("Persisted ChromaDB vector store", 
                                persist_directory=self.persist_directory,
                                collection_name=self.collection_name)
            else:
                self.logger.warning("No store initialized to persist.")
        except Exception as e:
            self.logger.error("Failed to persist ChromaDB vector store", error=str(e))
            raise
    
    def clear(self) -> bool:
        """Clear all documents from the vector store."""
        try:
            store = self.get_store()
            if hasattr(store, '_collection'):
                # Get all document IDs and delete them
                try:
                    collection = store._collection
                    all_docs = collection.get()
                    if all_docs and 'ids' in all_docs and all_docs['ids']:
                        collection.delete(ids=all_docs['ids'])
                        self.logger.info("Cleared ChromaDB vector store")
                        return True
                    else:
                        self.logger.info("ChromaDB vector store is already empty")
                        return True
                except Exception as e:
                    self.logger.warning("Failed to clear ChromaDB using collection method", error=str(e))
            
            # Fallback: reinitialize
            self._store = None
            self.logger.info("Reset ChromaDB vector store")
            return True
            
        except Exception as e:
            self.logger.error("Failed to clear ChromaDB vector store", error=str(e))
            return False
