"""Standalone chunk manager that can work independently of the RAG pipeline."""

from typing import List, Dict, Any, Optional
import structlog
from pathlib import Path

from .chunk_manager import ChunkManager
from ..config.settings import get_config
from ..chunking import EmbeddingProviderFactory
from . import VectorStoreFactory, VectorStoreManager

logger = structlog.get_logger(__name__)


class StandaloneChunkManager:
    """Standalone chunk manager that initializes its own components."""
    
    def __init__(self, config=None):
        """Initialize the standalone chunk manager.
        
        Args:
            config: Optional configuration object. If None, uses default config.
        """
        self.config = config or get_config()
        self.logger = structlog.get_logger(self.__class__.__name__)
        
        # Components that will be initialized on demand
        self._embedding_provider = None
        self._vector_store = None
        self._vector_manager = None
        self._chunk_manager = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize all required components."""
        if self._initialized:
            return
        
        try:
            self.logger.info("Initializing standalone chunk manager...")
            
            # Initialize embedding provider
            self._embedding_provider = EmbeddingProviderFactory.create_provider(
                self.config.embedding.provider,
                model_name=self.config.embedding.embedding_model_name,
                model_kwargs=self.config.embedding.model_kwargs,
                encode_kwargs=self.config.embedding.encode_kwargs
            )
            
            # Initialize vector store
            embeddings = self._embedding_provider.get_embeddings()
            self._vector_store = VectorStoreFactory.create_store(
                self.config.vectorstore.provider,
                embeddings,
                persist_directory=self.config.vectorstore.persist_directory,
                collection_name=self.config.vectorstore.collection_name
            )
            
            # Initialize vector store manager
            self._vector_manager = VectorStoreManager(self._vector_store)
            
            # Initialize chunk manager
            self._chunk_manager = ChunkManager(
                self._vector_manager, 
                self._embedding_provider
            )
            
            self._initialized = True
            self.logger.info("Standalone chunk manager initialized successfully")
            
        except Exception as e:
            self.logger.error("Failed to initialize standalone chunk manager", error=str(e))
            raise
    
    def _ensure_initialized(self):
        """Ensure the manager is initialized before use."""
        if not self._initialized:
            self.initialize()
    
    # Delegate all chunk management methods to the underlying chunk manager
    
    def list_chunks(self, **kwargs) -> List[Dict[str, Any]]:
        """List chunks with filtering options."""
        self._ensure_initialized()
        return self._chunk_manager.list_chunks(**kwargs)
    
    def get_chunk_details(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific chunk."""
        self._ensure_initialized()
        return self._chunk_manager.get_chunk_details(chunk_id)
    
    def search_chunks(self, text_query: str, **kwargs) -> List[Dict[str, Any]]:
        """Search chunks by content."""
        self._ensure_initialized()
        return self._chunk_manager.search_chunks(text_query, **kwargs)
    
    def edit_chunk(self, chunk_id: str, new_content: str, **kwargs) -> bool:
        """Edit chunk content and re-embed."""
        self._ensure_initialized()
        return self._chunk_manager.edit_chunk(chunk_id, new_content, **kwargs)
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a specific chunk."""
        self._ensure_initialized()
        return self._chunk_manager.delete_chunk(chunk_id)
    
    def get_document_stats(self, source: str) -> Dict[str, Any]:
        """Get statistics about a document's chunks."""
        self._ensure_initialized()
        return self._chunk_manager.get_document_stats(source)
    
    def validate_chunks(self, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find problematic chunks."""
        self._ensure_initialized()
        return self._chunk_manager.validate_chunks(source)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get overall collection statistics."""
        self._ensure_initialized()
        return self._vector_manager.get_collection_stats()
    
    def get_status(self) -> Dict[str, Any]:
        """Get the status of the chunk manager."""
        return {
            "initialized": self._initialized,
            "config": {
                "embedding_provider": self.config.embedding.provider,
                "embedding_model": self.config.embedding.embedding_model_name,
                "vector_store": self.config.vectorstore.provider,
                "persist_directory": self.config.vectorstore.persist_directory,
                "collection_name": self.config.vectorstore.collection_name,
            }
        }
    
    @classmethod
    def from_config(cls, config=None) -> "StandaloneChunkManager":
        """Create chunk manager from configuration.
        
        Args:
            config: Optional configuration object
            
        Returns:
            Configured chunk manager
        """
        manager = cls(config)
        manager.initialize()
        return manager
