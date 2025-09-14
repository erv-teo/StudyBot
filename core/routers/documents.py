"""Document ingestion endpoints for StudyBot API."""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import structlog
import tempfile
import os



logger = structlog.get_logger(__name__)
router = APIRouter()


# Request/Response Models
class DocumentAddRequest(BaseModel):
    """Request model for adding a document from URL or path."""
    source: str = Field(..., description="Document source (URL or file path)")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class TextAddRequest(BaseModel):
    """Request model for adding raw text."""
    text: str = Field(..., description="Text content to add")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class DocumentResponse(BaseModel):
    """Response model for document operations."""
    success: bool = Field(..., description="Operation success status")
    source: str = Field(..., description="Document source")
    message: str = Field(..., description="Operation message")
    chunks_added: Optional[int] = Field(default=None, description="Number of chunks added")


class DocumentListResponse(BaseModel):
    """Response model for document list."""
    documents: List[Dict[str, Any]] = Field(..., description="List of documents")
    total_count: int = Field(..., description="Total number of documents")


class DocumentStatsResponse(BaseModel):
    """Response model for document statistics."""
    source: str = Field(..., description="Document source")
    total_chunks: int = Field(..., description="Total number of chunks")
    total_characters: int = Field(..., description="Total characters")
    average_chunk_size: int = Field(..., description="Average chunk size")
    sections: Dict[str, int] = Field(..., description="Chunk sections breakdown")
    doc_id: Optional[str] = Field(default=None, description="Document ID")


# Dependency to get RAG pipeline
def get_rag_pipeline():
    """Dependency to get the initialized RAG pipeline."""
    from core.main import rag_pipeline
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    return rag_pipeline


@router.post("/", response_model=DocumentResponse, status_code=201)
async def add_document(
    request: DocumentAddRequest,
    pipeline = Depends(get_rag_pipeline)
) -> DocumentResponse:
    """Add a document from URL or file path."""
    try:
        logger.info("Adding document", source=request.source)
        
        success = pipeline.add_document(request.source, request.metadata)
        
        if success:
            return DocumentResponse(
                success=True,
                source=request.source,
                message="Document added successfully"
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to add document")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to add document", source=request.source, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to add document: {str(e)}")


@router.post("/upload", response_model=DocumentResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None, description="JSON metadata string"),
    pipeline = Depends(get_rag_pipeline)
) -> DocumentResponse:
    """Upload and add a document file."""
    try:
        import json
        
        # Parse metadata if provided
        parsed_metadata = None
        if metadata:
            try:
                parsed_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON metadata")
        
        # Initialize metadata if None
        if parsed_metadata is None:
            parsed_metadata = {}
        
        # Add document name to metadata
        parsed_metadata['original_filename'] = file.filename
        parsed_metadata['content_type'] = file.content_type or 'application/octet-stream'
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            logger.info("Uploading document", filename=file.filename, size=len(content))
            
            # Add document using the temporary file path with metadata including filename
            success = pipeline.add_document(temp_file_path, parsed_metadata)
            
            if success:
                return DocumentResponse(
                    success=True,
                    source=file.filename,  # Use filename as source for display
                    message="Document uploaded and added successfully"
                )
            else:
                raise HTTPException(status_code=400, detail="Failed to process uploaded document")
                
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass  # File might already be deleted
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to upload document", filename=file.filename, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {str(e)}")


@router.post("/text", response_model=DocumentResponse, status_code=201)
async def add_text(
    request: TextAddRequest,
    pipeline = Depends(get_rag_pipeline)
) -> DocumentResponse:
    """Add raw text to the knowledge base."""
    try:
        logger.info("Adding text", text_length=len(request.text))
        
        success = pipeline.add_text(request.text, request.metadata)
        
        if success:
            return DocumentResponse(
                success=True,
                source="direct_text",
                message="Text added successfully"
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to add text")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to add text", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to add text: {str(e)}")


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    pipeline = Depends(get_rag_pipeline)
) -> DocumentListResponse:
    """List all ingested documents."""
    try:
        from core.main import document_store
        
        if not document_store:
            raise HTTPException(status_code=503, detail="Document store not initialized")
        
        # Get documents from document store
        documents_data = document_store.list_documents()
        
        # Convert to response format
        documents = []
        for doc in documents_data:
            documents.append({
                'doc_id': doc['doc_id'],
                'source': doc['source'],
                'original_filename': doc['original_filename'],
                'chunks_count': doc['total_chunks'],
                'total_chars': doc['total_chars'],
                'content_type': doc['content_type'],
                'created_at': doc['created_at']
            })
        
        return DocumentListResponse(
            documents=documents,
            total_count=len(documents)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list documents", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.get("/{source}/stats", response_model=DocumentStatsResponse)
async def get_document_stats(
    source: str,
    pipeline = Depends(get_rag_pipeline)
) -> DocumentStatsResponse:
    """Get statistics for a specific document."""
    try:
        from core.main import chunk_manager
        
        if not chunk_manager:
            raise HTTPException(status_code=503, detail="Chunk manager not initialized")
        
        stats = chunk_manager.get_document_stats(source)
        
        if 'error' in stats:
            raise HTTPException(status_code=404, detail=stats['error'])
        
        return DocumentStatsResponse(
            source=stats['source'],
            total_chunks=stats['total_chunks'],
            total_characters=stats['total_characters'],
            average_chunk_size=stats['average_chunk_size'],
            sections=stats['sections'],
            doc_id=stats.get('doc_id')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get document stats", source=source, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get document stats: {str(e)}")


@router.delete("/{source}", status_code=204)
async def delete_document(
    source: str,
    pipeline = Depends(get_rag_pipeline)
):
    """Delete a document and all its chunks."""
    try:
        from core.main import chunk_manager
        
        if not chunk_manager:
            raise HTTPException(status_code=503, detail="Chunk manager not initialized")
        
        # Get all chunks for this document
        chunks = chunk_manager.get_document_chunks(source)
        
        if not chunks:
            raise HTTPException(status_code=404, detail=f"Document {source} not found")
        
        # Delete all chunks
        deleted_count = 0
        for chunk in chunks:
            chunk_id = chunk.get('chunk_id')
            if chunk_id and chunk_manager.delete_chunk(chunk_id):
                deleted_count += 1
        
        logger.info("Deleted document", source=source, chunks_deleted=deleted_count)
        
        if deleted_count == 0:
            raise HTTPException(status_code=500, detail="Failed to delete any chunks")
        
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete document", source=source, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@router.post("/clear", status_code=204)
async def clear_knowledge_base(
    pipeline = Depends(get_rag_pipeline)
):
    """Clear all documents from the knowledge base."""
    try:
        logger.info("Clearing knowledge base")
        
        success = pipeline.clear_knowledge_base()
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to clear knowledge base")
        
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to clear knowledge base", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to clear knowledge base: {str(e)}")
