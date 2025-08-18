"""LLM provider factory."""

from typing import Any, Dict
import structlog

from ..base import BaseLLMProvider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .ollama import OllamaProvider
from .google import GoogleProvider

logger = structlog.get_logger(__name__)


class LLMProviderFactory:
    """Factory for creating LLM providers."""
    
    @staticmethod
    def create_provider(provider_type: str, **kwargs) -> BaseLLMProvider:
        """Create an LLM provider of the specified type.
        
        Args:
            provider_type: Type of provider ("openai", "anthropic", "ollama", "google")
            **kwargs: Arguments for the provider
            
        Returns:
            LLM provider instance
            
        Raises:
            ValueError: If provider type is not supported
        """
        if provider_type == "openai":
            return OpenAIProvider(**kwargs)
        elif provider_type == "anthropic":
            return AnthropicProvider(**kwargs)
        elif provider_type == "ollama":
            return OllamaProvider(**kwargs)
        elif provider_type == "google":
            return GoogleProvider(**kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider type: {provider_type}")
    
    @staticmethod
    def get_recommended_provider(use_case: str, **kwargs) -> BaseLLMProvider:
        """Get recommended provider for use case.
        
        Args:
            use_case: Use case ("local", "cloud", "fast", "quality")
            **kwargs: Arguments for the provider
            
        Returns:
            Recommended LLM provider
        """
        if use_case == "local":
            return OllamaProvider(**kwargs)
        elif use_case == "cloud":
            return OpenAIProvider(**kwargs)
        elif use_case == "fast":
            return OpenAIProvider(model_name="gpt-3.5-turbo", **kwargs)
        elif use_case == "quality":
            return AnthropicProvider(**kwargs)
        else:
            return OpenAIProvider(**kwargs)
    
    @staticmethod
    def get_available_providers() -> Dict[str, Dict[str, Any]]:
        """Get information about available providers.
        
        Returns:
            Dictionary with provider information
        """
        return {
            "openai": {
                "name": "OpenAI",
                "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                "requires_api_key": True,
                "description": "OpenAI's GPT models"
            },
            "anthropic": {
                "name": "Anthropic",
                "models": ["claude-3-sonnet-20240229", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
                "requires_api_key": True,
                "description": "Anthropic's Claude models"
            },
            "ollama": {
                "name": "Ollama",
                "models": ["llama2", "mistral", "codellama", "neural-chat"],
                "requires_api_key": False,
                "description": "Local LLM models via Ollama"
            },
            "google": {
                "name": "Google Gemini",
                "models": ["gemini-pro", "gemini-pro-vision", "gemini-1.5-pro", "gemini-1.5-flash"],
                "requires_api_key": True,
                "description": "Google's Gemini models"
            }
        }
