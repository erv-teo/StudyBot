"""Factory for creating document loaders."""

from typing import List
import structlog
from langchain_core.documents import Document

from ..base import BaseDocumentLoader
from .web import WebDocumentLoader
from .pdf import PDFDocumentLoader
from .text import TextDocumentLoader

logger = structlog.get_logger(__name__)


class DocumentLoaderFactory:
    """Factory for creating appropriate document loaders."""
    
    def __init__(self):
        self.loaders = [
            WebDocumentLoader(),
            PDFDocumentLoader(),
            TextDocumentLoader(),
        ]
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def get_loader(self, source: str) -> BaseDocumentLoader:
        """Get the appropriate loader for a source.
        
        Args:
            source: The source to load from
            
        Returns:
            Appropriate document loader
            
        Raises:
            ValueError: If no suitable loader is found
        """
        for loader in self.loaders:
            if loader.supports(source):
                self.logger.debug("Found loader", source=source, loader=loader.__class__.__name__)
                return loader
        
        raise ValueError(f"No suitable loader found for source: {source}")
    
    def load_document(self, source: str, **kwargs) -> List[Document]:
        """Load documents using the appropriate loader.
        
        Args:
            source: The source to load from
            **kwargs: Additional arguments for the loader
            
        Returns:
            List of loaded documents
        """
        loader = self.get_loader(source)
        # Update loader with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(loader, key):
                setattr(loader, key, value)
        
        return loader.load(source)
