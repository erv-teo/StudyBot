"""FAISS vector store implementation."""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import structlog
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from ..base import BaseVectorStore

logger = structlog.get_logger(__name__)


class FAISSVectorStore(BaseVectorStore):
    """FAISS vector store implementation."""
    
    def __init__(self, 
                 embeddings: Embeddings,
                 persist_directory: str = "./data/vectorstore",
                 index_name: str = "faiss_index",
                 **kwargs):
        super().__init__(embeddings, **kwargs)
        self.persist_directory = persist_directory
        self.index_name = index_name
        self.index_path = os.path.join(persist_directory, f"{index_name}.faiss")
        self.pkl_path = os.path.join(persist_directory, f"{index_name}.pkl")
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def initialize(self) -> VectorStore:
        """Initialize the FAISS vector store."""
        try:
            from langchain_community.vectorstores import FAISS
            
            # Ensure persist directory exists
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            
            # Try to load existing index
            if os.path.exists(self.index_path) and os.path.exists(self.pkl_path):
                try:
                    store = FAISS.load_local(
                        self.persist_directory, 
                        self.embeddings,
                        index_name=self.index_name,
                        allow_dangerous_deserialization=True
                    )
                    self.logger.info("Loaded existing FAISS index", 
                                   index_path=self.index_path)
                    return store
                except Exception as e:
                    self.logger.warning("Failed to load existing FAISS index, creating new one", 
                                      error=str(e))
            
            # Create new empty index
            # FAISS requires at least one document to initialize, so we'll defer creation
            # until the first document is added
            self.logger.info("FAISS vector store ready for initialization", 
                           persist_directory=self.persist_directory,
                           index_name=self.index_name)
            return None  # Will be created when first document is added
            
        except ImportError:
            self.logger.error("langchain_community not installed. Install with: pip install langchain-community")
            raise
        except Exception as e:
            self.logger.error("Failed to initialize FAISS vector store", error=str(e))
            raise
    
    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """Add documents to the vector store."""
        try:
            from langchain_community.vectorstores import FAISS
            
            store = self.get_store()
            
            if store is None:
                # Create initial index with first batch of documents
                store = FAISS.from_documents(documents, self.embeddings)
                self._store = store
                self.logger.info("Created new FAISS index with initial documents", 
                               count=len(documents))
            else:
                # Add to existing index
                store.add_documents(documents, **kwargs)
                self.logger.debug("Added documents to existing FAISS index", 
                                count=len(documents))
            
            # Save the index
            store.save_local(self.persist_directory, self.index_name)
            
            # Generate IDs (FAISS doesn't return IDs by default)
            ids = [f"faiss_{i}" for i in range(len(documents))]
            return ids
            
        except Exception as e:
            self.logger.error("Failed to add documents to FAISS", error=str(e))
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
            if store is None:
                self.logger.warning("FAISS index not initialized, returning empty results")
                return []
            
            results = store.similarity_search(query, k=k, **kwargs)
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
            if store is None:
                self.logger.warning("FAISS index not initialized, returning empty results")
                return []
            
            results = store.similarity_search_with_score(query, k=k, **kwargs)
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
            # FAISS doesn't support deletion directly
            self.logger.warning("FAISS doesn't support document deletion")
            return False
        except Exception as e:
            self.logger.error("Failed to delete documents from FAISS", error=str(e))
            return False
    
    def get_stats(self) -> dict:
        """Get vector store statistics."""
        try:
            store = self.get_store()
            
            stats = {
                "type": "faiss",
                "persist_directory": self.persist_directory,
                "index_name": self.index_name,
                "initialized": store is not None,
                "index_exists": os.path.exists(self.index_path),
                "pkl_exists": os.path.exists(self.pkl_path)
            }
            
            if store is not None and hasattr(store, 'index'):
                try:
                    stats["document_count"] = store.index.ntotal
                except Exception:
                    pass
            
            return stats
            
        except Exception as e:
            self.logger.error("Failed to get FAISS stats", error=str(e))
            return {"error": str(e)}
    
    def clear(self) -> bool:
        """Clear all documents from the vector store."""
        try:
            # Remove index files
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
            if os.path.exists(self.pkl_path):
                os.remove(self.pkl_path)
            
            self._store = None
            self.logger.info("Cleared FAISS vector store")
            return True
            
        except Exception as e:
            self.logger.error("Failed to clear FAISS vector store", error=str(e))
            return False
