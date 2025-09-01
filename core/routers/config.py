"""Configuration endpoints for StudyBot API."""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import structlog

logger = structlog.get_logger(__name__)
router = APIRouter()


# Response Models
class EmbeddingConfigResponse(BaseModel):
    """Response model for embedding configuration."""
    provider: str = Field(..., description="Embedding provider")
    model_name: str = Field(..., description="Embedding model name")


class LLMConfigResponse(BaseModel):
    """Response model for LLM configuration."""
    provider: str = Field(..., description="LLM provider")
    model_name: str = Field(..., description="LLM model name")
    temperature: float = Field(..., description="Temperature for generation")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens for generation")


class VectorStoreConfigResponse(BaseModel):
    """Response model for vector store configuration."""
    provider: str = Field(..., description="Vector store provider")
    collection_name: str = Field(..., description="Collection name")


class ChunkingConfigResponse(BaseModel):
    """Response model for chunking configuration."""
    chunk_size: int = Field(..., description="Size of text chunks")
    chunk_overlap: int = Field(..., description="Overlap between chunks")


class RAGConfigResponse(BaseModel):
    """Response model for RAG configuration."""
    retrieval_k: int = Field(..., description="Number of documents to retrieve")
    score_threshold: Optional[float] = Field(default=None, description="Minimum similarity score threshold")
    enable_query_analysis: bool = Field(..., description="Enable query analysis step")


class SystemConfigResponse(BaseModel):
    """Response model for system configuration."""
    log_level: str = Field(..., description="Logging level")
    data_directory: str = Field(..., description="Base data directory")


class ConfigResponse(BaseModel):
    """Response model for complete configuration."""
    embedding: EmbeddingConfigResponse = Field(..., description="Embedding configuration")
    llm: LLMConfigResponse = Field(..., description="LLM configuration")
    vectorstore: VectorStoreConfigResponse = Field(..., description="Vector store configuration")
    chunking: ChunkingConfigResponse = Field(..., description="Chunking configuration")
    rag: RAGConfigResponse = Field(..., description="RAG configuration")
    system: SystemConfigResponse = Field(..., description="System configuration")


# Dependency to get configuration
def get_config():
    """Dependency to get the current configuration."""
    from core.main import config
    if not config:
        raise HTTPException(status_code=503, detail="Configuration not initialized")
    return config


@router.get("/", response_model=ConfigResponse)
async def get_configuration(config = Depends(get_config)) -> ConfigResponse:
    """Get the current system configuration."""
    try:
        return ConfigResponse(
            embedding=EmbeddingConfigResponse(
                provider=config.embedding.provider,
                model_name=config.embedding.embedding_model_name
            ),
            llm=LLMConfigResponse(
                provider=config.llm.provider,
                model_name=config.llm.llm_model_name,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens
            ),
            vectorstore=VectorStoreConfigResponse(
                provider=config.vectorstore.provider,
                collection_name=config.vectorstore.collection_name
            ),
            chunking=ChunkingConfigResponse(
                chunk_size=config.chunking.chunk_size,
                chunk_overlap=config.chunking.chunk_overlap
            ),
            rag=RAGConfigResponse(
                retrieval_k=config.rag.retrieval_k,
                score_threshold=config.rag.score_threshold,
                enable_query_analysis=config.rag.enable_query_analysis
            ),
            system=SystemConfigResponse(
                log_level=config.log_level,
                data_directory=config.data_directory
            )
        )
        
    except Exception as e:
        logger.error("Failed to get configuration", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get configuration: {str(e)}")


@router.get("/summary", response_model=Dict[str, Any])
async def get_config_summary(config = Depends(get_config)) -> Dict[str, Any]:
    """Get a summary of the current configuration."""
    try:
        return {
            "embedding_provider": config.embedding.provider,
            "embedding_model": config.embedding.embedding_model_name,
            "llm_provider": config.llm.provider,
            "llm_model": config.llm.llm_model_name,
            "vectorstore_provider": config.vectorstore.provider,
            "chunk_size": config.chunking.chunk_size,
            "chunk_overlap": config.chunking.chunk_overlap,
            "retrieval_k": config.rag.retrieval_k,
            "log_level": config.log_level
        }
        
    except Exception as e:
        logger.error("Failed to get configuration summary", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get configuration summary: {str(e)}")
