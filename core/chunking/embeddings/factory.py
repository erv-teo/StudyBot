"""Factory for creating embedding providers."""

from ..base import BaseEmbeddingProvider
from .sentence_transformer import SentenceTransformerProvider
from .huggingface import HuggingFaceProvider
from .openai import OpenAIEmbeddingProvider


class EmbeddingProviderFactory:
    """Factory for creating embedding providers."""
    
    @staticmethod
    def create_provider(provider_type: str, **kwargs) -> BaseEmbeddingProvider:
        """Create an embedding provider of the specified type.
        
        Args:
            provider_type: Type of provider ("sentence-transformers", "huggingface", "openai")
            **kwargs: Arguments for the provider
            
        Returns:
            Embedding provider instance
            
        Raises:
            ValueError: If provider type is not supported
        """
        if provider_type == "sentence-transformers":
            return SentenceTransformerProvider(**kwargs)
        elif provider_type == "huggingface":
            return HuggingFaceProvider(**kwargs)
        elif provider_type == "openai":
            return OpenAIEmbeddingProvider(**kwargs)
        else:
            raise ValueError(f"Unsupported embedding provider type: {provider_type}")
    
    @staticmethod
    def get_recommended_provider(use_case: str, **kwargs) -> BaseEmbeddingProvider:
        """Get recommended provider for use case.
        
        Args:
            use_case: Use case ("local", "cloud", "multilingual", "fast")
            **kwargs: Arguments for the provider
            
        Returns:
            Recommended embedding provider
        """
        if use_case == "local":
            return SentenceTransformerProvider(
                model_name="all-MiniLM-L6-v2", **kwargs
            )
        elif use_case == "cloud":
            return OpenAIEmbeddingProvider(**kwargs)
        elif use_case == "multilingual":
            return SentenceTransformerProvider(
                model_name="paraphrase-multilingual-MiniLM-L12-v2", **kwargs
            )
        elif use_case == "fast":
            return SentenceTransformerProvider(
                model_name="all-MiniLM-L6-v2", **kwargs
            )
        else:
            return SentenceTransformerProvider(**kwargs)
