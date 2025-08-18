"""Simple character text splitter."""

from typing import List
import structlog
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

from ..base import BaseTextSplitter

logger = structlog.get_logger(__name__)


class SimpleTextSplitter(BaseTextSplitter):
    """Simple character text splitter."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, 
                 separator: str = "\n\n", **kwargs):
        super().__init__(chunk_size, chunk_overlap, **kwargs)
        self.separator = separator
        self._splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator,
            **kwargs
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        try:
            chunks = self._splitter.split_documents(documents)
            logger.info("Split documents with character splitter", 
                       input_docs=len(documents), 
                       output_chunks=len(chunks))
            return chunks
        except Exception as e:
            logger.error("Failed to split documents with character splitter", error=str(e))
            raise
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        try:
            chunks = self._splitter.split_text(text)
            logger.debug("Split text with character splitter", 
                        input_length=len(text), 
                        output_chunks=len(chunks))
            return chunks
        except Exception as e:
            logger.error("Failed to split text with character splitter", error=str(e))
            raise
