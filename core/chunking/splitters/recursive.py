"""Recursive character text splitter."""

from typing import List, Optional
import structlog
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..base import BaseTextSplitter

logger = structlog.get_logger(__name__)


class RecursiveTextSplitter(BaseTextSplitter):
    """Recursive character text splitter."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, 
                 separators: Optional[List[str]] = None, **kwargs):
        super().__init__(chunk_size, chunk_overlap, **kwargs)
        self.separators = separators
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            **kwargs
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        try:
            chunks = self._splitter.split_documents(documents)
            logger.info("Split documents recursively", 
                       input_docs=len(documents), 
                       output_chunks=len(chunks))
            return chunks
        except Exception as e:
            logger.error("Failed to split documents recursively", error=str(e))
            raise
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        try:
            chunks = self._splitter.split_text(text)
            logger.debug("Split text recursively", 
                        input_length=len(text), 
                        output_chunks=len(chunks))
            return chunks
        except Exception as e:
            logger.error("Failed to split text recursively", error=str(e))
            raise
