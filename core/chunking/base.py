"""Base classes for text chunking and embeddings."""

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict
import structlog
import uuid
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = structlog.get_logger(__name__)


class BaseTextSplitter(ABC):
    """Abstract base class for text splitters."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, **kwargs):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.kwargs = kwargs
    
    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        pass
    
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        pass


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
    
    @abstractmethod
    def get_embeddings(self) -> Embeddings:
        """Get the embeddings instance.
        
        Returns:
            Embeddings instance
        """
        pass
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        pass


class ChunkProcessor:
    """Processes document chunks and adds metadata."""
    
    def __init__(self):
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def process_chunks(self, chunks: List[Document], section_strategy: str = "position") -> List[Document]:
        """Process chunks and add section metadata.
        
        Args:
            chunks: List of document chunks
            section_strategy: Strategy for determining sections ("position", "content", "none")
            
        Returns:
            Processed chunks with section metadata
        """
        if section_strategy == "position":
            return self._add_position_sections(chunks)
        elif section_strategy == "content":
            return self._add_content_sections(chunks)
        else:
            return chunks
    
    def _add_position_sections(self, chunks: List[Document]) -> List[Document]:
        """Add section metadata based on position in document."""
        total_chunks = len(chunks)
        if total_chunks == 0:
            return chunks
        
        third = total_chunks // 3
        
        for i, chunk in enumerate(chunks):
            if i < third:
                chunk.metadata["section"] = "beginning"
            elif i < 2 * third:
                chunk.metadata["section"] = "middle"
            else:
                chunk.metadata["section"] = "end"
            
            # Add chunk metadata with unique UUID
            chunk.metadata.update({
                "chunk_id": str(uuid.uuid4()),  # Unique identifier for each chunk
                "chunk_index": i,
                "total_chunks": total_chunks,
                "chunk_size": len(chunk.page_content),
            })
        
        self.logger.info("Added position-based sections", total_chunks=total_chunks)
        return chunks
    
    def _add_content_sections(self, chunks: List[Document]) -> List[Document]:
        """Add section metadata based on content analysis."""
        # Simple content-based sectioning
        # This could be enhanced with more sophisticated NLP
        
        for i, chunk in enumerate(chunks):
            content = chunk.page_content.lower()
            
            # Simple heuristics for section detection
            if any(keyword in content for keyword in ["introduction", "overview", "summary", "abstract"]):
                section = "introduction"
            elif any(keyword in content for keyword in ["conclusion", "summary", "results", "findings"]):
                section = "conclusion"
            elif any(keyword in content for keyword in ["method", "approach", "implementation", "algorithm"]):
                section = "methodology"
            else:
                section = "content"
            
            chunk.metadata.update({
                "section": section,
                "chunk_id": str(uuid.uuid4()),  # Unique identifier for each chunk
                "chunk_index": i,
                "chunk_size": len(chunk.page_content),
            })
        
        self.logger.info("Added content-based sections", total_chunks=len(chunks))
        return chunks
