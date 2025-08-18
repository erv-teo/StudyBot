"""Chunking and embeddings module for StudyBot."""

from .base import BaseTextSplitter, BaseEmbeddingProvider, ChunkProcessor
from .splitters import (
    TextSplitterFactory,
    RecursiveTextSplitter,
    SimpleTextSplitter,
    TokenBasedSplitter,
    MarkdownSplitter,
)
from .embeddings import (
    EmbeddingProviderFactory,
    SentenceTransformerProvider,
    HuggingFaceProvider,
    OpenAIEmbeddingProvider,
)

__all__ = [
    "BaseTextSplitter",
    "BaseEmbeddingProvider",
    "ChunkProcessor",
    "TextSplitterFactory",
    "RecursiveTextSplitter",
    "SimpleTextSplitter",
    "TokenBasedSplitter",
    "MarkdownSplitter",
    "EmbeddingProviderFactory",
    "SentenceTransformerProvider",
    "HuggingFaceProvider",
    "OpenAIEmbeddingProvider",
]
