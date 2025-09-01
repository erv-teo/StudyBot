"""FastAPI application entry point for StudyBot RAG system."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import structlog
import datetime
from typing import Dict, Any

from .config.settings import get_config
from .rag.pipeline import RAGPipeline
from .vectorstore.chunk_manager import ChunkManager
from .vectorstore.document_store import DocumentStore
from .vectorstore import VectorStoreManager
from .vectorstore.providers.factory import VectorStoreFactory
from .chunking.embeddings.factory import EmbeddingProviderFactory

# Configure logging
logger = structlog.get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="StudyBot RAG API",
    description="REST API for StudyBot RAG (Retrieval-Augmented Generation) system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (initialized at startup)
rag_pipeline: RAGPipeline = None
chunk_manager: ChunkManager = None
document_store: DocumentStore = None
config = None


@app.on_event("startup")
async def startup_event():
    """Initialize all services on startup."""
    global rag_pipeline, chunk_manager, document_store, config
    
    try:
        logger.info("Starting StudyBot RAG API...")
        
        # Load configuration
        config = get_config()
        logger.info("Configuration loaded", config_summary={
            "embedding_provider": config.embedding.provider,
            "llm_provider": config.llm.provider,
            "vectorstore_provider": config.vectorstore.provider
        })
        
        # Initialize all components
        from core.chunking import TextSplitterFactory, ChunkProcessor
        from core.ingestion import DocumentProcessor
        from core.llm import LLMProviderFactory, LLMManager
        
        # Initialize embedding provider
        embedding_provider = EmbeddingProviderFactory.create_provider(
            config.embedding.provider,
            model_name=config.embedding.embedding_model_name,
            model_kwargs=config.embedding.model_kwargs,
            encode_kwargs=config.embedding.encode_kwargs
        )
        
        # Initialize vector store
        embeddings = embedding_provider.get_embeddings()
        vector_store = VectorStoreFactory.create_store(
            config.vectorstore.provider,
            embeddings,
            persist_directory=config.vectorstore.persist_directory,
            collection_name=config.vectorstore.collection_name
        )
        
        # Initialize vector store manager
        vector_manager = VectorStoreManager(vector_store)
        
        # Initialize LLM
        llm_provider = LLMProviderFactory.create_provider(
            config.llm.provider,
            model_name=config.llm.llm_model_name,
            api_key=config.llm.api_key,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            base_url=config.llm.base_url
        )
        llm_manager = LLMManager(llm_provider)
        
        # Initialize document processing components
        document_processor = DocumentProcessor()
        chunk_processor = ChunkProcessor()
        text_splitter = TextSplitterFactory.create_splitter(
            "recursive",
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap
        )
        
        # Initialize document store
        document_store = DocumentStore(persist_directory=config.vectorstore.persist_directory)
        logger.info("Document store initialized")
        
        # Initialize RAG pipeline with all components
        rag_pipeline = RAGPipeline(
            config=config,
            vector_manager=vector_manager,
            llm_manager=llm_manager,
            document_processor=document_processor,
            chunk_processor=chunk_processor,
            text_splitter=text_splitter,
            document_store=document_store
        )
        logger.info("RAG pipeline initialized with all components")
        
        # Initialize chunk manager using already-created components
        chunk_manager = ChunkManager(vector_manager, embedding_provider)
        logger.info("Chunk manager initialized")
        
        logger.info("StudyBot RAG API startup completed successfully")
        
    except Exception as e:
        logger.error("Failed to initialize StudyBot RAG API", error=str(e))
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down StudyBot RAG API...")


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    try:
        # Check if all components are initialized
        rag_healthy = rag_pipeline is not None
        chunk_manager_healthy = chunk_manager is not None
        vector_store_healthy = chunk_manager is not None  # Simplified check
        
        overall_status = "healthy" if all([rag_healthy, chunk_manager_healthy, vector_store_healthy]) else "unhealthy"
        
        return {
            "status": overall_status,
            "timestamp": datetime.datetime.now().isoformat(),
            "version": "1.0.0",
            "services": {
                "rag_pipeline": rag_healthy,
                "chunk_manager": chunk_manager_healthy,
                "vector_store": vector_store_healthy
            }
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "version": "1.0.0",
            "services": {
                "rag_pipeline": False,
                "chunk_manager": False,
                "vector_store": False
            }
        }


@app.get("/status")
async def get_status() -> Dict[str, Any]:
    """Get detailed system status."""
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        
        status = rag_pipeline.get_status()
        status["api_status"] = "running"
        status["chunk_manager_available"] = chunk_manager is not None
        
        return status
        
    except Exception as e:
        logger.error("Failed to get status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


# Import and include routers
from core.routers import rag, chunks, documents, config

app.include_router(rag.router, prefix="/rag", tags=["RAG"])
app.include_router(chunks.router, prefix="/chunks", tags=["Chunks"])
app.include_router(documents.router, prefix="/documents", tags=["Documents"])
app.include_router(config.router, prefix="/config", tags=["Configuration"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
