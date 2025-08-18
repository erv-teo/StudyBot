"""Text document loader."""

import os
from pathlib import Path
from typing import List
import structlog
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader

from ..base import BaseDocumentLoader

logger = structlog.get_logger(__name__)


class TextDocumentLoader(BaseDocumentLoader):
    """Loader for text documents."""
    
    def __init__(self, encoding: str = "utf-8", **kwargs):
        super().__init__(**kwargs)
        self.encoding = encoding
    
    def load(self, source: str) -> List[Document]:
        """Load documents from a text file."""
        try:
            loader = TextLoader(source, encoding=self.encoding)
            docs = loader.load()
            logger.info("Loaded text document", file=source, doc_count=len(docs))
            return docs
        except Exception as e:
            logger.error("Failed to load text document", file=source, error=str(e))
            raise
    
    def supports(self, source: str) -> bool:
        """Check if source is a text file."""
        text_extensions = {".txt", ".md", ".rst", ".py", ".js", ".ts", ".html", ".css"}
        return Path(source).suffix.lower() in text_extensions and os.path.exists(source)
