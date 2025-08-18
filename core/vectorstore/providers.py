"""Concrete vector store implementations."""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable, Tuple
import structlog
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, InMemoryVectorStore

from .base import BaseVectorStore

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
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents with scores."""
        try:
            store = self.get_store()
            results = store.similarity_search_with_score(query, k=k, filter=filter, **kwargs)
            self.logger.debug("Performed similarity search with scores", query_length=len(query), results_count=len(results))
            return results
        except Exception as e:
            self.logger.error("Failed to perform similarity search with scores", error=str(e))
            raise
    
    def delete(self, ids: List[str]) -> bool:
        """Delete documents by IDs."""
        try:
            store = self.get_store()
            if hasattr(store, 'delete'):
                store.delete(ids)
            self.logger.debug("Deleted documents", ids_count=len(ids))
            return True
        except Exception as e:
            self.logger.error("Failed to delete documents", error=str(e))
            return False
    
    def persist(self) -> bool:
        """Persist the vector store (no-op for in-memory)."""
        self.logger.debug("Persist called on in-memory store (no-op)")
        return True


class ChromaVectorStore(BaseVectorStore):
    """Chroma vector store implementation."""
    
    def __init__(self, embeddings: Embeddings, persist_directory: Optional[str] = None, 
                 collection_name: str = "studybot_documents", **kwargs):
        super().__init__(embeddings, **kwargs)
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def initialize(self) -> VectorStore:
        """Initialize the Chroma vector store."""
        try:
            from langchain_chroma import Chroma
            
            init_kwargs = {
                "embedding_function": self.embeddings,
                "collection_name": self.collection_name,
            }
            
            if self.persist_directory:
                # Ensure directory exists
                Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
                init_kwargs["persist_directory"] = self.persist_directory
            
            store = Chroma(**init_kwargs)
            self.logger.info("Initialized Chroma vector store", 
                           collection=self.collection_name,
                           persist_dir=self.persist_directory)
            return store
        except ImportError:
            self.logger.error("Chroma package not available. Install with: pip install chromadb")
            raise
        except Exception as e:
            self.logger.error("Failed to initialize Chroma vector store", error=str(e))
            raise
    
    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """Add documents to the vector store."""
        try:
            store = self.get_store()
            ids = store.add_documents(documents, **kwargs)
            self.logger.debug("Added documents to Chroma store", count=len(documents))
            return ids
        except Exception as e:
            self.logger.error("Failed to add documents to Chroma store", error=str(e))
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
            search_kwargs = kwargs.copy()
            if filter:
                search_kwargs["filter"] = filter
            
            results = store.similarity_search(query, k=k, **search_kwargs)
            self.logger.debug("Performed similarity search", query_length=len(query), results_count=len(results))
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
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents with scores."""
        try:
            store = self.get_store()
            search_kwargs = kwargs.copy()
            if filter:
                search_kwargs["filter"] = filter
            
            results = store.similarity_search_with_score(query, k=k, **search_kwargs)
            self.logger.debug("Performed similarity search with scores", query_length=len(query), results_count=len(results))
            return results
        except Exception as e:
            self.logger.error("Failed to perform similarity search with scores", error=str(e))
            raise
    
    def delete(self, ids: List[str]) -> bool:
        """Delete documents by IDs."""
        try:
            store = self.get_store()
            store.delete(ids)
            self.logger.debug("Deleted documents from Chroma", ids_count=len(ids))
            return True
        except Exception as e:
            self.logger.error("Failed to delete documents from Chroma", error=str(e))
            return False
    
    def persist(self) -> bool:
        """Persist the vector store."""
        try:
            store = self.get_store()
            if hasattr(store, 'persist'):
                store.persist()
            self.logger.debug("Persisted Chroma vector store")
            return True
        except Exception as e:
            self.logger.error("Failed to persist Chroma vector store", error=str(e))
            return False


class FAISSVectorStore(BaseVectorStore):
    """FAISS vector store implementation."""
    
    def __init__(self, embeddings: Embeddings, index_path: Optional[str] = None, **kwargs):
        super().__init__(embeddings, **kwargs)
        self.index_path = index_path
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def initialize(self) -> VectorStore:
        """Initialize the FAISS vector store."""
        try:
            from langchain_community.vectorstores import FAISS
            
            # Try to load existing index
            if self.index_path and os.path.exists(self.index_path):
                store = FAISS.load_local(
                    self.index_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self.logger.info("Loaded existing FAISS index", path=self.index_path)
            else:
                # Create empty index - will be populated when documents are added
                store = None
                self.logger.info("Will create new FAISS index when documents are added")
            
            return store
        except ImportError:
            self.logger.error("FAISS package not available. Install with: pip install faiss-cpu")
            raise
        except Exception as e:
            self.logger.error("Failed to initialize FAISS vector store", error=str(e))
            raise
    
    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """Add documents to the vector store."""
        try:
            from langchain_community.vectorstores import FAISS
            
            store = self._store
            if store is None:
                # Create new index with first batch of documents
                store = FAISS.from_documents(documents, self.embeddings)
                self._store = store
                self.logger.info("Created new FAISS index", doc_count=len(documents))
            else:
                # Add to existing index
                store.add_documents(documents, **kwargs)
                self.logger.debug("Added documents to FAISS store", count=len(documents))
            
            # Return dummy IDs (FAISS doesn't return actual IDs)
            return [f"faiss_{i}" for i in range(len(documents))]
        except Exception as e:
            self.logger.error("Failed to add documents to FAISS store", error=str(e))
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
            if store is None:
                self.logger.warning("FAISS store not initialized")
                return []
            
            results = store.similarity_search(query, k=k, **kwargs)
            
            # Apply filter if provided
            if filter:
                results = [doc for doc in results if filter(doc)]
            
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
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents with scores."""
        try:
            store = self.get_store()
            if store is None:
                self.logger.warning("FAISS store not initialized")
                return []
            
            results = store.similarity_search_with_score(query, k=k, **kwargs)
            
            # Apply filter if provided
            if filter:
                results = [(doc, score) for doc, score in results if filter(doc)]
            
            self.logger.debug("Performed similarity search with scores", query_length=len(query), results_count=len(results))
            return results
        except Exception as e:
            self.logger.error("Failed to perform similarity search with scores", error=str(e))
            raise
    
    def delete(self, ids: List[str]) -> bool:
        """Delete documents by IDs (not supported by FAISS)."""
        self.logger.warning("Delete operation not supported by FAISS")
        return False
    
    def persist(self) -> bool:
        """Persist the vector store."""
        try:
            store = self.get_store()
            if store is None or not self.index_path:
                self.logger.warning("Cannot persist - store not initialized or no index path")
                return False
            
            # Ensure directory exists
            Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
            store.save_local(self.index_path)
            self.logger.debug("Persisted FAISS vector store", path=self.index_path)
            return True
        except Exception as e:
            self.logger.error("Failed to persist FAISS vector store", error=str(e))
            return False


class VectorStoreFactory:
    """Factory for creating vector stores."""
    
    @staticmethod
    def create_store(store_type: str, embeddings: Embeddings, **kwargs) -> BaseVectorStore:
        """Create a vector store of the specified type.
        
        Args:
            store_type: Type of store ("inmemory", "chroma", "faiss")
            embeddings: Embeddings instance
            **kwargs: Arguments for the store
            
        Returns:
            Vector store instance
            
        Raises:
            ValueError: If store type is not supported
        """
        if store_type == "inmemory":
            return StudyBotInMemoryVectorStore(embeddings, **kwargs)
        elif store_type == "chroma":
            return ChromaVectorStore(embeddings, **kwargs)
        elif store_type == "faiss":
            return FAISSVectorStore(embeddings, **kwargs)
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")
    
    @staticmethod
    def get_recommended_store(use_case: str, embeddings: Embeddings, **kwargs) -> BaseVectorStore:
        """Get recommended store for use case.
        
        Args:
            use_case: Use case ("development", "production", "large_scale", "persistent")
            embeddings: Embeddings instance
            **kwargs: Arguments for the store
            
        Returns:
            Recommended vector store
        """
        if use_case == "development":
            return StudyBotInMemoryVectorStore(embeddings, **kwargs)
        elif use_case == "production":
            return ChromaVectorStore(embeddings, **kwargs)
        elif use_case == "large_scale":
            return FAISSVectorStore(embeddings, **kwargs)
        elif use_case == "persistent":
            return ChromaVectorStore(embeddings, **kwargs)
        else:
            return ChromaVectorStore(embeddings, **kwargs)
