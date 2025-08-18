"""Google Gemini LLM provider implementation."""

from typing import List, Optional
import structlog
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage

from ..base import BaseLLMProvider

logger = structlog.get_logger(__name__)


class GoogleProvider(BaseLLMProvider):
    """Google Gemini LLM provider."""
    
    def __init__(self, 
                 model_name: str = "gemini-pro",
                 api_key: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def initialize(self) -> BaseLanguageModel:
        """Initialize the Google Gemini LLM."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
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
            
            llm = ChatGoogleGenerativeAI(**init_kwargs)
            logger.info("Initialized Google Gemini LLM", model=self.model_name)
            return llm
            
        except ImportError:
            logger.error("langchain_google_genai not installed. Install with: pip install langchain-google-genai")
            raise
        except Exception as e:
            logger.error("Failed to initialize Google Gemini LLM", error=str(e))
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt using Google Gemini."""
        try:
            if not self._llm:
                self._llm = self.initialize()
            
            result = self._llm.invoke(prompt, **kwargs)
            logger.debug("Generated text using Google Gemini", 
                        input_length=len(prompt), 
                        output_length=len(result.content))
            return result.content
            
        except Exception as e:
            logger.error("Failed to generate text with Google Gemini", error=str(e))
            raise
    
    def generate_with_messages(self, messages: List[BaseMessage], **kwargs) -> str:
        """Generate text using Google Gemini with messages."""
        try:
            if not self._llm:
                self._llm = self.initialize()
            
            result = self._llm.invoke(messages, **kwargs)
            logger.debug("Generated text using Google Gemini", 
                        input_length=len(str(messages)), 
                        output_length=len(result.content))
            return result.content
            
        except Exception as e:
            logger.error("Failed to generate text with Google Gemini", error=str(e))
            raise
