"""Sentence Transformers embedding provider."""

from typing import List, Dict, Any, Optional
import structlog
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings as SentenceTransformerEmbeddings

from ..base import BaseEmbeddingProvider

logger = structlog.get_logger(__name__)


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """Sentence Transformers embedding provider."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", 
                 model_kwargs: Optional[Dict[str, Any]] = None,
                 encode_kwargs: Optional[Dict[str, Any]] = None,
                 **kwargs):
        super().__init__(model_name, **kwargs)
        self.model_kwargs = model_kwargs or {}
        self.encode_kwargs = encode_kwargs or {}
        self._embeddings = None
    
    def get_embeddings(self) -> Embeddings:
        """Get the embeddings instance."""
        if self._embeddings is None:
            try:
                self._embeddings = SentenceTransformerEmbeddings(
                    model_name=self.model_name,
                    model_kwargs=self.model_kwargs,
                    encode_kwargs=self.encode_kwargs
                )
                logger.info("Initialized SentenceTransformer embeddings", 
                           model=self.model_name)
            except Exception as e:
                logger.error("Failed to initialize SentenceTransformer embeddings", 
                           model=self.model_name, error=str(e))
                raise
        
        return self._embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        try:
            embeddings = self.get_embeddings()
            vectors = embeddings.embed_documents(texts)
            logger.debug("Embedded documents", count=len(texts))
            return vectors
        except Exception as e:
            logger.error("Failed to embed documents", error=str(e))
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        try:
            embeddings = self.get_embeddings()
            vector = embeddings.embed_query(text)
            logger.debug("Embedded query", text_length=len(text))
            return vector
        except Exception as e:
            logger.error("Failed to embed query", error=str(e))
            raise
