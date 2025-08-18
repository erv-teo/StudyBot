"""PDF document loader."""

import os
from pathlib import Path
from typing import List
import structlog
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from ..base import BaseDocumentLoader

logger = structlog.get_logger(__name__)


class PDFDocumentLoader(BaseDocumentLoader):
    """Loader for PDF documents."""
    
    def load(self, source: str) -> List[Document]:
        """Load documents from a PDF file."""
        try:
            loader = PyPDFLoader(source)
            docs = loader.load()
            logger.info("Loaded PDF document", file=source, page_count=len(docs))
            return docs
        except Exception as e:
            logger.error("Failed to load PDF document", file=source, error=str(e))
            raise
    
    def supports(self, source: str) -> bool:
        """Check if source is a PDF file."""
        return Path(source).suffix.lower() == ".pdf" and os.path.exists(source)
