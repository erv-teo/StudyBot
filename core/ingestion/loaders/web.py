"""Web document loader."""

import structlog
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
import bs4

from ..base import BaseDocumentLoader

logger = structlog.get_logger(__name__)


class WebDocumentLoader(BaseDocumentLoader):
    """Loader for web documents."""
    
    def __init__(self, bs_kwargs: Dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        self.bs_kwargs = bs_kwargs or {}
    
    def load(self, source: str) -> List[Document]:
        """Load documents from a web URL."""
        try:
            loader = WebBaseLoader(
                web_paths=[source],
                bs_kwargs=self.bs_kwargs
            )
            docs = loader.load()
            logger.info("Loaded web document", url=source, doc_count=len(docs))
            return docs
        except Exception as e:
            logger.error("Failed to load web document", url=source, error=str(e))
            raise
    
    def supports(self, source: str) -> bool:
        """Check if source is a web URL."""
        return source.startswith(("http://", "https://"))
