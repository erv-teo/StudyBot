"""Configuration settings for StudyBot RAG system."""

import warnings
from typing import Any, Dict, Optional
from pydantic import Field
from pydantic_settings import BaseSettings

# Clean warning suppression (only for issues that can't be fixed via env vars)
import urllib3
urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)


class EmbeddingConfig(BaseSettings):
    """Configuration for embedding models."""
    
    provider: str = Field(default="sentence-transformers", description="Embedding provider")
    embedding_model_name: str = Field(default="all-MiniLM-L6-v2", description="Embedding model name")
    model_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Additional model kwargs")
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Additional encode kwargs")
    
    model_config = {"protected_namespaces": ("settings_",)}


class LLMConfig(BaseSettings):
    """Configuration for LLM models."""
    
    provider: str = Field(default="openai", description="LLM provider (openai, anthropic, ollama)")
    llm_model_name: str = Field(default="gpt-3.5-turbo", description="LLM model name")
    temperature: float = Field(default=0.7, description="Temperature for generation")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens for generation")
    api_key: Optional[str] = Field(default=None, description="API key for the LLM provider")
    base_url: Optional[str] = Field(default=None, description="Base URL for custom endpoints")
    
    model_config = {"protected_namespaces": ("settings_",)}


class VectorStoreConfig(BaseSettings):
    """Configuration for vector stores."""
    
    provider: str = Field(default="chroma", description="Vector store provider (chroma, faiss, inmemory)")
    persist_directory: Optional[str] = Field(default="./data/vectorstore", description="Directory to persist vector store")
    collection_name: str = Field(default="studybot_documents", description="Collection name")


class ChunkingConfig(BaseSettings):
    """Configuration for text chunking."""
    
    chunk_size: int = Field(default=1000, description="Size of text chunks")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    separators: Optional[list] = Field(default=None, description="Custom separators for chunking")


class TelegramConfig(BaseSettings):
    """Configuration for Telegram bot."""
    
    bot_token: Optional[str] = Field(default=None, description="Telegram bot token")
    webhook_url: Optional[str] = Field(default=None, description="Webhook URL for the bot")
    webhook_port: int = Field(default=8080, description="Port for webhook")
    polling: bool = Field(default=True, description="Use polling instead of webhooks")


class RAGConfig(BaseSettings):
    """Configuration for RAG pipeline."""
    
    retrieval_k: int = Field(default=5, description="Number of documents to retrieve")
    score_threshold: Optional[float] = Field(default=None, description="Minimum similarity score threshold")
    enable_query_analysis: bool = Field(default=True, description="Enable query analysis step")


class StudyBotConfig(BaseSettings):
    """Main configuration for StudyBot."""
    
    # Component configurations
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    
    # General settings
    log_level: str = Field(default="INFO", description="Logging level")
    data_directory: str = Field(default="./data", description="Base data directory")
    
    # System settings (these are read from .env but don't need to be used in code)
    tokenizers_parallelism: str = Field(default="false", description="HuggingFace tokenizers parallelism")
    user_agent: str = Field(default="StudyBot/1.0", description="User agent for web requests")
    anonymized_telemetry: str = Field(default="False", description="ChromaDB telemetry setting")
    chroma_telemetry: str = Field(default="False", description="ChromaDB telemetry setting")
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        case_sensitive = False


# Singleton instance
_config: Optional[StudyBotConfig] = None


def get_config() -> StudyBotConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = StudyBotConfig()
    return _config


def update_config(**kwargs) -> StudyBotConfig:
    """Update the global configuration with new values."""
    global _config
    _config = StudyBotConfig(**kwargs)
    return _config
