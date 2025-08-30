"""Chunk management endpoints for StudyBot API."""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import structlog

logger = structlog.get_logger(__name__)
router = APIRouter()


# Request/Response Models
class ChunkCreateRequest(BaseModel):
    """Request model for creating a chunk."""
    content: str = Field(..., description="Chunk content")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Chunk metadata")


class ChunkUpdateRequest(BaseModel):
    """Request model for updating a chunk."""
    content: Optional[str] = Field(default=None, description="New chunk content")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Updated metadata")


class ChunkResponse(BaseModel):
    """Response model for chunk data."""
    chunk_id: str = Field(..., description="Chunk ID")
    content: str = Field(..., description="Chunk content")
    content_preview: Optional[str] = Field(default=None, description="Content preview")
    metadata: Dict[str, Any] = Field(..., description="Chunk metadata")
    content_length: int = Field(..., description="Content length")
    created_at: Optional[str] = Field(default=None, description="Creation timestamp")


class ChunkListResponse(BaseModel):
    """Response model for chunk list."""
    chunks: List[ChunkResponse] = Field(..., description="List of chunks")
    total_count: int = Field(..., description="Total number of chunks")
    limit: int = Field(..., description="Limit applied")
    source_filter: Optional[str] = Field(default=None, description="Source filter applied")


class ChunkSearchRequest(BaseModel):
    """Request model for searching chunks."""
    query: str = Field(..., description="Search query")
    semantic: bool = Field(default=True, description="Use semantic search")
    metadata_filter: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filter")
    limit: int = Field(default=10, description="Maximum results")


class ChunkSearchResponse(BaseModel):
    """Response model for chunk search."""
    results: List[ChunkResponse] = Field(..., description="Search results")
    query: str = Field(..., description="Search query")
    relevance_type: str = Field(..., description="Type of relevance (semantic/text_match)")


# Dependency to get chunk manager
def get_chunk_manager():
    """Dependency to get the initialized chunk manager."""
    from core.main import chunk_manager
    if not chunk_manager:
        raise HTTPException(status_code=503, detail="Chunk manager not initialized")
    return chunk_manager


@router.get("/", response_model=ChunkListResponse)
async def list_chunks(
    source_filter: Optional[str] = Query(None, description="Filter by source path/URL"),
    doc_id_filter: Optional[str] = Query(None, description="Filter by document ID"),
    section_filter: Optional[str] = Query(None, description="Filter by section"),
    content_preview_length: int = Query(100, description="Length of content preview"),
    limit: int = Query(50, description="Maximum number of chunks to return"),
    manager = Depends(get_chunk_manager)
) -> ChunkListResponse:
    """List chunks with optional filtering."""
    try:
        chunks_data = manager.list_chunks(
            source_filter=source_filter,
            doc_id_filter=doc_id_filter,
            section_filter=section_filter,
            content_preview_length=content_preview_length,
            limit=limit
        )
        
        chunks = []
        for chunk_data in chunks_data:
            chunks.append(ChunkResponse(
                chunk_id=chunk_data.get('chunk_id', 'unknown'),
                content=chunk_data.get('content', ''),
                content_preview=chunk_data.get('content_preview'),
                metadata=chunk_data.get('metadata', {}),
                content_length=len(chunk_data.get('content', '')),
                created_at=chunk_data.get('created_at')
            ))
        
        return ChunkListResponse(
            chunks=chunks,
            total_count=len(chunks),
            limit=limit,
            source_filter=source_filter
        )
        
    except Exception as e:
        logger.error("Failed to list chunks", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list chunks: {str(e)}")


@router.get("/{chunk_id}", response_model=ChunkResponse)
async def get_chunk(
    chunk_id: str,
    manager = Depends(get_chunk_manager)
) -> ChunkResponse:
    """Get a specific chunk by ID."""
    try:
        chunk_data = manager.get_chunk_details(chunk_id)
        
        if not chunk_data:
            raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found")
        
        return ChunkResponse(
            chunk_id=chunk_data.get('chunk_id', chunk_id),
            content=chunk_data.get('content', ''),
            metadata=chunk_data.get('metadata', {}),
            content_length=chunk_data.get('content_length', 0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get chunk", chunk_id=chunk_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get chunk: {str(e)}")


@router.post("/", response_model=ChunkResponse, status_code=201)
async def create_chunk(
    request: ChunkCreateRequest,
    manager = Depends(get_chunk_manager)
) -> ChunkResponse:
    """Create a new chunk manually."""
    try:
        # This would need to be implemented in the chunk manager
        # For now, we'll use the RAG pipeline's add_text method
        from core.main import rag_pipeline
        
        success = rag_pipeline.add_text(request.content, request.metadata)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to create chunk")
        
        # Get the created chunk (this is a simplified approach)
        # In a real implementation, you'd return the actual created chunk
        return ChunkResponse(
            chunk_id="new_chunk",  # This would be the actual ID
            content=request.content,
            metadata=request.metadata or {},
            content_length=len(request.content)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create chunk", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create chunk: {str(e)}")


@router.patch("/{chunk_id}", response_model=ChunkResponse)
async def update_chunk(
    chunk_id: str,
    request: ChunkUpdateRequest,
    manager = Depends(get_chunk_manager)
) -> ChunkResponse:
    """Update a chunk's content and/or metadata."""
    try:
        if not request.content and not request.metadata:
            raise HTTPException(status_code=400, detail="Must provide content or metadata to update")
        
        # Get current chunk
        current_chunk = manager.get_chunk_details(chunk_id)
        if not current_chunk:
            raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found")
        
        # Prepare update data
        new_content = request.content if request.content is not None else current_chunk.get('content', '')
        new_metadata = current_chunk.get('metadata', {}).copy()
        if request.metadata:
            new_metadata.update(request.metadata)
        
        # Update chunk
        success = manager.edit_chunk(chunk_id, new_content, new_metadata)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update chunk")
        
        # Get updated chunk
        updated_chunk = manager.get_chunk_details(chunk_id)
        
        return ChunkResponse(
            chunk_id=updated_chunk.get('chunk_id', chunk_id),
            content=updated_chunk.get('content', ''),
            metadata=updated_chunk.get('metadata', {}),
            content_length=updated_chunk.get('content_length', 0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update chunk", chunk_id=chunk_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to update chunk: {str(e)}")


@router.delete("/{chunk_id}", status_code=204)
async def delete_chunk(
    chunk_id: str,
    manager = Depends(get_chunk_manager)
):
    """Delete a chunk."""
    try:
        success = manager.delete_chunk(chunk_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found or could not be deleted")
        
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete chunk", chunk_id=chunk_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete chunk: {str(e)}")


@router.post("/search", response_model=ChunkSearchResponse)
async def search_chunks(
    request: ChunkSearchRequest,
    manager = Depends(get_chunk_manager)
) -> ChunkSearchResponse:
    """Search chunks by content."""
    try:
        results_data = manager.search_chunks(
            text_query=request.query,
            semantic=request.semantic,
            metadata_filter=request.metadata_filter,
            limit=request.limit
        )
        
        results = []
        for result_data in results_data:
            results.append(ChunkResponse(
                chunk_id=result_data.get('chunk_id', 'unknown'),
                content=result_data.get('content', ''),
                metadata=result_data.get('metadata', {}),
                content_length=len(result_data.get('content', ''))
            ))
        
        return ChunkSearchResponse(
            results=results,
            query=request.query,
            relevance_type=results_data[0].get('relevance_type', 'semantic') if results_data else 'semantic'
        )
        
    except Exception as e:
        logger.error("Failed to search chunks", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to search chunks: {str(e)}")


@router.get("/document/{source}", response_model=ChunkListResponse)
async def get_document_chunks(
    source: str,
    manager = Depends(get_chunk_manager)
) -> ChunkListResponse:
    """Get all chunks for a specific document."""
    try:
        chunks_data = manager.get_document_chunks(source)
        
        chunks = []
        for chunk_data in chunks_data:
            chunks.append(ChunkResponse(
                chunk_id=chunk_data.get('chunk_id', 'unknown'),
                content=chunk_data.get('content', ''),
                content_preview=chunk_data.get('content_preview'),
                metadata=chunk_data.get('metadata', {}),
                content_length=len(chunk_data.get('content', ''))
            ))
        
        return ChunkListResponse(
            chunks=chunks,
            total_count=len(chunks),
            limit=len(chunks),
            source_filter=source
        )
        
    except Exception as e:
        logger.error("Failed to get document chunks", source=source, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get document chunks: {str(e)}")
