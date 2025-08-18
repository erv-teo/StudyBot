"""Base classes for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import structlog
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

logger = structlog.get_logger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self._llm: Optional[BaseLanguageModel] = None
    
    @abstractmethod
    def initialize(self) -> BaseLanguageModel:
        """Initialize the LLM.
        
        Returns:
            Initialized LLM instance
        """
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def generate_with_messages(self, messages: List[BaseMessage], **kwargs) -> str:
        """Generate text from a list of messages.
        
        Args:
            messages: List of messages
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        pass
    
    def get_llm(self) -> BaseLanguageModel:
        """Get the LLM instance."""
        if self._llm is None:
            self._llm = self.initialize()
        return self._llm
    
    def create_messages(self, 
                       system_prompt: Optional[str] = None,
                       user_message: str = "",
                       conversation_history: Optional[List[Dict[str, str]]] = None) -> List[BaseMessage]:
        """Create message list for conversation.
        
        Args:
            system_prompt: Optional system prompt
            user_message: Current user message
            conversation_history: Previous conversation messages
            
        Returns:
            List of formatted messages
        """
        messages = []
        
        # Add system message if provided
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        # Add conversation history
        if conversation_history:
            for msg in conversation_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        
        # Add current user message
        if user_message:
            messages.append(HumanMessage(content=user_message))
        
        return messages


class LLMManager:
    """Manager for LLM operations and conversation handling."""
    
    def __init__(self, llm_provider: BaseLLMProvider):
        self.llm_provider = llm_provider
        self.conversation_history: List[Dict[str, str]] = []
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def chat(self, 
             user_message: str,
             system_prompt: Optional[str] = None,
             reset_history: bool = False,
             **kwargs) -> str:
        """Have a conversation with the LLM.
        
        Args:
            user_message: User's message
            system_prompt: Optional system prompt
            reset_history: Whether to reset conversation history
            **kwargs: Additional generation parameters
            
        Returns:
            LLM's response
        """
        try:
            if reset_history:
                self.conversation_history = []
            
            # Create messages for this turn
            messages = self.llm_provider.create_messages(
                system_prompt=system_prompt,
                user_message=user_message,
                conversation_history=self.conversation_history
            )
            
            # Generate response
            response = self.llm_provider.generate_with_messages(messages, **kwargs)
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            self.logger.debug("Generated chat response", 
                            user_msg_length=len(user_message),
                            response_length=len(response),
                            history_length=len(self.conversation_history))
            
            return response
            
        except Exception as e:
            self.logger.error("Failed to generate chat response", error=str(e))
            raise
    
    def ask(self, question: str, context: Optional[str] = None, **kwargs) -> str:
        """Ask a one-off question (without conversation history).
        
        Args:
            question: Question to ask
            context: Optional context
            **kwargs: Additional generation parameters
            
        Returns:
            LLM's answer
        """
        try:
            if context:
                prompt = f"Context: {context}\n\nQuestion: {question}"
            else:
                prompt = question
            
            response = self.llm_provider.generate(prompt, **kwargs)
            
            self.logger.debug("Generated answer", 
                            question_length=len(question),
                            response_length=len(response))
            
            return response
            
        except Exception as e:
            self.logger.error("Failed to generate answer", error=str(e))
            raise
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history."""
        return self.conversation_history.copy()
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
        self.logger.debug("Cleared conversation history")
    
    def save_conversation(self, file_path: str) -> bool:
        """Save conversation history to file.
        
        Args:
            file_path: Path to save the conversation
            
        Returns:
            True if successful
        """
        try:
            import json
            from pathlib import Path
            
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
            
            self.logger.info("Saved conversation", file_path=file_path)
            return True
            
        except Exception as e:
            self.logger.error("Failed to save conversation", file_path=file_path, error=str(e))
            return False
    
    def load_conversation(self, file_path: str) -> bool:
        """Load conversation history from file.
        
        Args:
            file_path: Path to load the conversation from
            
        Returns:
            True if successful
        """
        try:
            import json
            
            with open(file_path, 'r') as f:
                self.conversation_history = json.load(f)
            
            self.logger.info("Loaded conversation", 
                           file_path=file_path,
                           messages_count=len(self.conversation_history))
            return True
            
        except Exception as e:
            self.logger.error("Failed to load conversation", file_path=file_path, error=str(e))
            return False
