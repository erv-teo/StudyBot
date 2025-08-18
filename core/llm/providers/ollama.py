"""Ollama LLM provider implementation."""

from typing import List, Optional
import structlog
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage

from ..base import BaseLLMProvider

logger = structlog.get_logger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider."""
    
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
            logger.error("langchain_ollama not installed. Install with: pip install langchain-ollama")
            raise
        except Exception as e:
            logger.error("Failed to initialize Ollama LLM", error=str(e))
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        try:
            if not self._llm:
                self._llm = self.initialize()
            
            result = self._llm.invoke(prompt, **kwargs)
            logger.debug("Generated text with Ollama", 
                        input_length=len(prompt), 
                        output_length=len(result.content))
            return result.content
            
        except Exception as e:
            logger.error("Failed to generate text with Ollama", error=str(e))
            raise
    
    def generate_with_messages(self, messages: List[BaseMessage], **kwargs) -> str:
        """Generate text from messages."""
        try:
            if not self._llm:
                self._llm = self.initialize()
            
            # For Ollama, we might need to handle the response differently
            # depending on the model and version
            result = self._llm.invoke(messages, **kwargs)
            
            # Handle different response types
            if hasattr(result, 'content'):
                response_text = result.content
            else:
                response_text = str(result)
            
            logger.debug("Generated text with messages", 
                        messages_count=len(messages),
                        response_length=len(response_text))
            return response_text
            
        except Exception as e:
            logger.error("Failed to generate text with messages", error=str(e))
            raise
