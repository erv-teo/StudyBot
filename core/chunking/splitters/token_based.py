"""Token-based text splitter."""

from typing import List
import structlog
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter

from ..base import BaseTextSplitter

logger = structlog.get_logger(__name__)


class TokenBasedSplitter(BaseTextSplitter):
    """Token-based text splitter."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, 
                 encoding_name: str = "gpt2", **kwargs):
        super().__init__(chunk_size, chunk_overlap, **kwargs)
        self.encoding_name = encoding_name
        self._splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding_name,
            **kwargs
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        try:
            chunks = self._splitter.split_documents(documents)
            logger.info("Split documents with token splitter", 
                       input_docs=len(documents), 
                       output_chunks=len(chunks))
            return chunks
        except Exception as e:
            logger.error("Failed to split documents with token splitter", error=str(e))
            raise
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        try:
            chunks = self._splitter.split_text(text)
            logger.debug("Split text with token splitter", 
                        input_length=len(text), 
                        output_chunks=len(chunks))
            return chunks
        except Exception as e:
            logger.error("Failed to split text with token splitter", error=str(e))
            raise
