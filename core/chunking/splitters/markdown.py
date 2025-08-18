"""Markdown header-based text splitter."""

from typing import List, Optional
import structlog
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from ..base import BaseTextSplitter

logger = structlog.get_logger(__name__)


class MarkdownSplitter(BaseTextSplitter):
    """Markdown header-based text splitter."""
    
    def __init__(self, headers_to_split_on: Optional[List[tuple]] = None, **kwargs):
        super().__init__(**kwargs)
        self.headers_to_split_on = headers_to_split_on or [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self._splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        try:
            all_chunks = []
            for doc in documents:
                chunks = self._splitter.split_text(doc.page_content)
                # Convert to Document objects
                for chunk in chunks:
                    new_doc = Document(
                        page_content=chunk.page_content,
                        metadata={**doc.metadata, **chunk.metadata}
                    )
                    all_chunks.append(new_doc)
            
            logger.info("Split documents with markdown splitter", 
                       input_docs=len(documents), 
                       output_chunks=len(all_chunks))
            return all_chunks
        except Exception as e:
            logger.error("Failed to split documents with markdown splitter", error=str(e))
            raise
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        try:
            chunks = self._splitter.split_text(text)
            # Extract page content from Document objects
            text_chunks = [chunk.page_content for chunk in chunks]
            logger.debug("Split text with markdown splitter", 
                        input_length=len(text), 
                        output_chunks=len(text_chunks))
            return text_chunks
        except Exception as e:
            logger.error("Failed to split text with markdown splitter", error=str(e))
            raise
