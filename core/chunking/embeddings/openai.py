"""OpenAI embedding provider."""

from typing import List, Optional
import structlog
from langchain_core.embeddings import Embeddings

from ..base import BaseEmbeddingProvider

logger = structlog.get_logger(__name__)


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider."""
    
    def __init__(self, model_name: str = "text-embedding-ada-002",
                 api_key: Optional[str] = None,
                 **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key
        self._embeddings = None
    
    def get_embeddings(self) -> Embeddings:
        """Get the embeddings instance."""
        if self._embeddings is None:
            try:
                from langchain_openai import OpenAIEmbeddings
                
                init_kwargs = {"model": self.model_name}
                if self.api_key:
                    init_kwargs["api_key"] = self.api_key
                
                self._embeddings = OpenAIEmbeddings(**init_kwargs)
                logger.info("Initialized OpenAI embeddings", model=self.model_name)
            except ImportError:
                logger.error("OpenAI package not available. Install with: pip install langchain-openai")
                raise
            except Exception as e:
                logger.error("Failed to initialize OpenAI embeddings", 
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
