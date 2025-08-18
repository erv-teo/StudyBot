"""Factory for creating text splitters."""

from ..base import BaseTextSplitter
from .recursive import RecursiveTextSplitter
from .simple import SimpleTextSplitter
from .token_based import TokenBasedSplitter
from .markdown import MarkdownSplitter


class TextSplitterFactory:
    """Factory for creating text splitters."""
    
    @staticmethod
    def create_splitter(splitter_type: str, **kwargs) -> BaseTextSplitter:
        """Create a text splitter of the specified type.
        
        Args:
            splitter_type: Type of splitter ("recursive", "simple", "token", "markdown")
            **kwargs: Arguments for the splitter
            
        Returns:
            Text splitter instance
            
        Raises:
            ValueError: If splitter type is not supported
        """
        if splitter_type == "recursive":
            return RecursiveTextSplitter(**kwargs)
        elif splitter_type == "simple":
            return SimpleTextSplitter(**kwargs)
        elif splitter_type == "token":
            return TokenBasedSplitter(**kwargs)
        elif splitter_type == "markdown":
            return MarkdownSplitter(**kwargs)
        else:
            raise ValueError(f"Unsupported splitter type: {splitter_type}")
    
    @staticmethod
    def get_recommended_splitter(content_type: str, **kwargs) -> BaseTextSplitter:
        """Get recommended splitter for content type.
        
        Args:
            content_type: Type of content ("markdown", "code", "general", "academic")
            **kwargs: Arguments for the splitter
            
        Returns:
            Recommended text splitter
        """
        if content_type == "markdown":
            return MarkdownSplitter(**kwargs)
        elif content_type == "code":
            return RecursiveTextSplitter(
                separators=["\n\n", "\n", " ", ""],
                **kwargs
            )
        elif content_type == "academic":
            return TokenBasedSplitter(**kwargs)
        else:  # general
            return RecursiveTextSplitter(**kwargs)
