"""RAG endpoints for StudyBot API."""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import structlog

from core.rag.pipeline import RAGPipeline

logger = structlog.get_logger(__name__)
router = APIRouter()


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for RAG query."""
    question: str = Field(..., description="The question to ask")
    k: Optional[int] = Field(default=None, description="Number of documents to retrieve")
    score_threshold: Optional[float] = Field(default=None, description="Minimum similarity score threshold")
    metadata_filter: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filter for retrieval")


class QueryResponse(BaseModel):
    """Response model for RAG query."""
    answer: str = Field(..., description="Generated answer")
    question: str = Field(..., description="Original question")
    retrieved_docs_count: Optional[int] = Field(default=None, description="Number of documents retrieved")
    processing_time_ms: Optional[float] = Field(default=None, description="Processing time in milliseconds")


class RAGStatusResponse(BaseModel):
    """Response model for RAG status."""
    initialized: bool = Field(..., description="Whether RAG pipeline is initialized")
    components: Dict[str, bool] = Field(..., description="Status of RAG components")
    config: Dict[str, str] = Field(..., description="Current configuration")


# Dependency to get RAG pipeline
def get_rag_pipeline():
    """Dependency to get the initialized RAG pipeline."""
    from core.main import rag_pipeline
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    return rag_pipeline


@router.post("/query", response_model=QueryResponse)
async def query_knowledge_base(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
) -> QueryResponse:
    """Query the knowledge base and generate an answer."""
    try:
        import time
        start_time = time.time()
        
        logger.info("Processing RAG query", question_length=len(request.question))
        
        # Prepare query parameters
        query_params = {}
        if request.k is not None:
            query_params['k'] = request.k
        if request.score_threshold is not None:
            query_params['score_threshold'] = request.score_threshold
        if request.metadata_filter is not None:
            query_params['metadata_filter'] = request.metadata_filter
        
        # Execute query
        answer = pipeline.query(request.question, **query_params)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        logger.info(
            "RAG query completed",
            question_length=len(request.question),
            answer_length=len(answer),
            processing_time_ms=processing_time
        )
        
        return QueryResponse(
            answer=answer,
            question=request.question,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error("Failed to process RAG query", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")


@router.get("/status", response_model=RAGStatusResponse)
async def get_rag_status(
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
) -> RAGStatusResponse:
    """Get RAG pipeline status."""
    try:
        status = pipeline.get_status()
        
        return RAGStatusResponse(
            initialized=status.get("initialized", False),
            components=status.get("components", {}),
            config=status.get("config", {})
        )
        
    except Exception as e:
        logger.error("Failed to get RAG status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get RAG status: {str(e)}")
