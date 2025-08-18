"""OpenAI LLM provider implementation."""

from typing import List, Optional
import structlog
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage

from ..base import BaseLLMProvider

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
            logger.error("langchain_openai not installed. Install with: pip install langchain-openai")
            raise
        except Exception as e:
            logger.error("Failed to initialize OpenAI LLM", error=str(e))
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        try:
            if not self._llm:
                self._llm = self.initialize()
            
            result = self._llm.invoke(prompt, **kwargs)
            logger.debug("Generated text with OpenAI", 
                        input_length=len(prompt), 
                        output_length=len(result.content))
            return result.content
            
        except Exception as e:
            logger.error("Failed to generate text with OpenAI", error=str(e))
            raise
    
    def generate_with_messages(self, messages: List[BaseMessage], **kwargs) -> str:
        """Generate text from messages."""
        try:
            if not self._llm:
                self._llm = self.initialize()
            
            result = self._llm.invoke(messages, **kwargs)
            logger.debug("Generated text with messages", 
                        messages_count=len(messages),
                        response_length=len(result.content))
            return result.content
            
        except Exception as e:
            logger.error("Failed to generate text with messages", error=str(e))
            raise
