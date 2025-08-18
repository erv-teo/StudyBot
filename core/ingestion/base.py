"""Base classes for document ingestion."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pathlib import Path
import structlog
from langchain_core.documents import Document

logger = structlog.get_logger(__name__)


class BaseDocumentLoader(ABC):
    """Abstract base class for document loaders."""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    @abstractmethod
    def load(self, source: str) -> List[Document]:
        """Load documents from a source.
        
        Args:
            source: The source to load from (file path, URL, etc.)
            
        Returns:
            List of loaded documents
        """
        pass
    
    @abstractmethod
    def supports(self, source: str) -> bool:
        """Check if this loader supports the given source.
        
        Args:
            source: The source to check
            
        Returns:
            True if this loader can handle the source
        """
        pass


class DocumentProcessor:
    """Processes and enriches documents after loading."""
    
    def __init__(self):
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def process(self, documents: List[Document], metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Process documents and add metadata.
        
        Args:
            documents: List of documents to process
            metadata: Additional metadata to add
            
        Returns:
            Processed documents
        """
        processed_docs = []
        
        for i, doc in enumerate(documents):
            # Add base metadata
            doc.metadata.update({
                "doc_id": f"doc_{i}",
                "processing_timestamp": self._get_timestamp(),
                "source_type": self._detect_source_type(doc),
            })
            
            # Add custom metadata if provided
            if metadata:
                doc.metadata.update(metadata)
            
            # Clean and normalize content
            doc.page_content = self._clean_content(doc.page_content)
            
            processed_docs.append(doc)
            
        self.logger.info("Processed documents", count=len(processed_docs))
        return processed_docs
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _detect_source_type(self, doc: Document) -> str:
        """Detect the source type of a document."""
        source = doc.metadata.get("source", "")
        if source.startswith("http"):
            return "web"
        elif Path(source).suffix in [".pdf", ".docx", ".txt", ".md"]:
            return "file"
        return "unknown"
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize document content."""
        # Remove excessive whitespace
        content = " ".join(content.split())
        
        # Remove empty lines
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        
        return "\n".join(lines)
