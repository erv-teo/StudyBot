"""Concrete LLM provider implementations."""

from typing import Any, Dict, List, Optional
import structlog
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage

from .base import BaseLLMProvider

logger = structlog.get_logger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider."""
    
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo",
                 api_key: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def initialize(self) -> BaseLanguageModel:
        """Initialize the OpenAI LLM."""
        try:
            from langchain_openai import ChatOpenAI
            
            init_kwargs = {
                "model": self.model_name,
                "temperature": self.temperature,
            }
            
            if self.api_key:
                init_kwargs["api_key"] = self.api_key
            
            if self.max_tokens:
                init_kwargs["max_tokens"] = self.max_tokens
            
            # Add any additional kwargs
            init_kwargs.update(self.kwargs)
            
            llm = ChatOpenAI(**init_kwargs)
            logger.info("Initialized OpenAI LLM", model=self.model_name)
            return llm
            
        except ImportError:
            logger.error("OpenAI package not available. Install with: pip install langchain-openai")
            raise
        except Exception as e:
            logger.error("Failed to initialize OpenAI LLM", model=self.model_name, error=str(e))
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        try:
            llm = self.get_llm()
            response = llm.invoke(prompt, **kwargs)
            
            # Extract content from response
            if hasattr(response, 'content'):
                result = response.content
            else:
                result = str(response)
            
            logger.debug("Generated text with OpenAI", 
                        prompt_length=len(prompt),
                        response_length=len(result))
            return result
            
        except Exception as e:
            logger.error("Failed to generate text with OpenAI", error=str(e))
            raise
    
    def generate_with_messages(self, messages: List[BaseMessage], **kwargs) -> str:
        """Generate text from a list of messages."""
        try:
            llm = self.get_llm()
            response = llm.invoke(messages, **kwargs)
            
            # Extract content from response
            if hasattr(response, 'content'):
                result = response.content
            else:
                result = str(response)
            
            logger.debug("Generated text with messages", 
                        messages_count=len(messages),
                        response_length=len(result))
            return result
            
        except Exception as e:
            logger.error("Failed to generate text with messages", error=str(e))
            raise


class AnthropicProvider(BaseLLMProvider):
    """Anthropic LLM provider."""
    
    def __init__(self, 
                 model_name: str = "claude-3-sonnet-20240229",
                 api_key: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def initialize(self) -> BaseLanguageModel:
        """Initialize the Anthropic LLM."""
        try:
            from langchain_anthropic import ChatAnthropic
            
            init_kwargs = {
                "model": self.model_name,
                "temperature": self.temperature,
            }
            
            if self.api_key:
                init_kwargs["api_key"] = self.api_key
            
            if self.max_tokens:
                init_kwargs["max_tokens"] = self.max_tokens
            
            # Add any additional kwargs
            init_kwargs.update(self.kwargs)
            
            llm = ChatAnthropic(**init_kwargs)
            logger.info("Initialized Anthropic LLM", model=self.model_name)
            return llm
            
        except ImportError:
            logger.error("Anthropic package not available. Install with: pip install langchain-anthropic")
            raise
        except Exception as e:
            logger.error("Failed to initialize Anthropic LLM", model=self.model_name, error=str(e))
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        try:
            llm = self.get_llm()
            response = llm.invoke(prompt, **kwargs)
            
            # Extract content from response
            if hasattr(response, 'content'):
                result = response.content
            else:
                result = str(response)
            
            logger.debug("Generated text with Anthropic", 
                        prompt_length=len(prompt),
                        response_length=len(result))
            return result
            
        except Exception as e:
            logger.error("Failed to generate text with Anthropic", error=str(e))
            raise
    
    def generate_with_messages(self, messages: List[BaseMessage], **kwargs) -> str:
        """Generate text from a list of messages."""
        try:
            llm = self.get_llm()
            response = llm.invoke(messages, **kwargs)
            
            # Extract content from response
            if hasattr(response, 'content'):
                result = response.content
            else:
                result = str(response)
            
            logger.debug("Generated text with messages", 
                        messages_count=len(messages),
                        response_length=len(result))
            return result
            
        except Exception as e:
            logger.error("Failed to generate text with messages", error=str(e))
            raise


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider for local models."""
    
    def __init__(self, 
                 model_name: str = "llama2",
                 base_url: str = "http://localhost:11434",
                 temperature: float = 0.7,
                 **kwargs):
        super().__init__(model_name, **kwargs)
        self.base_url = base_url
        self.temperature = temperature
    
    def initialize(self) -> BaseLanguageModel:
        """Initialize the Ollama LLM."""
        try:
            from langchain_ollama import ChatOllama
            
            init_kwargs = {
                "model": self.model_name,
                "base_url": self.base_url,
                "temperature": self.temperature,
            }
            
            # Add any additional kwargs
            init_kwargs.update(self.kwargs)
            
            llm = ChatOllama(**init_kwargs)
            logger.info("Initialized Ollama LLM", model=self.model_name, base_url=self.base_url)
            return llm
            
        except ImportError:
            logger.error("Ollama package not available. Install with: pip install langchain-ollama")
            raise
        except Exception as e:
            logger.error("Failed to initialize Ollama LLM", model=self.model_name, error=str(e))
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        try:
            llm = self.get_llm()
            response = llm.invoke(prompt, **kwargs)
            
            # Extract content from response
            if hasattr(response, 'content'):
                result = response.content
            else:
                result = str(response)
            
            logger.debug("Generated text with Ollama", 
                        prompt_length=len(prompt),
                        response_length=len(result))
            return result
            
        except Exception as e:
            logger.error("Failed to generate text with Ollama", error=str(e))
            raise
    
    def generate_with_messages(self, messages: List[BaseMessage], **kwargs) -> str:
        """Generate text from a list of messages."""
        try:
            llm = self.get_llm()
            response = llm.invoke(messages, **kwargs)
            
            # Extract content from response
            if hasattr(response, 'content'):
                result = response.content
            else:
                result = str(response)
            
            logger.debug("Generated text with messages", 
                        messages_count=len(messages),
                        response_length=len(result))
            return result
            
        except Exception as e:
            logger.error("Failed to generate text with messages", error=str(e))
            raise


class LLMProviderFactory:
    """Factory for creating LLM providers."""
    
    @staticmethod
    def create_provider(provider_type: str, **kwargs) -> BaseLLMProvider:
        """Create an LLM provider of the specified type.
        
        Args:
            provider_type: Type of provider ("openai", "anthropic", "ollama")
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
