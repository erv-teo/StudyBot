"""Anthropic LLM provider implementation."""

from typing import List, Optional
import structlog
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage

from ..base import BaseLLMProvider

logger = structlog.get_logger(__name__)


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
            logger.error("langchain_anthropic not installed. Install with: pip install langchain-anthropic")
            raise
        except Exception as e:
            logger.error("Failed to initialize Anthropic LLM", error=str(e))
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        try:
            if not self._llm:
                self._llm = self.initialize()
            
            result = self._llm.invoke(prompt, **kwargs)
            logger.debug("Generated text with Anthropic", 
                        input_length=len(prompt), 
                        output_length=len(result.content))
            return result.content
            
        except Exception as e:
            logger.error("Failed to generate text with Anthropic", error=str(e))
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
