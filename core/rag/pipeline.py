"""RAG Pipeline - orchestrates document ingestion, retrieval, and generation."""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import structlog
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from ..config.settings import StudyBotConfig, get_config
from ..ingestion import DocumentLoaderFactory, DocumentProcessor
from ..chunking import (
    TextSplitterFactory, 
    EmbeddingProviderFactory, 
    ChunkProcessor
)
from ..vectorstore import VectorStoreFactory, VectorStoreManager
from ..llm import LLMProviderFactory, LLMManager

logger = structlog.get_logger(__name__)


class RAGPipeline:
    """Main RAG pipeline that orchestrates all components."""
    
    def __init__(self, 
                 config: Optional[StudyBotConfig] = None,
                 vector_manager: Optional[VectorStoreManager] = None,
                 llm_manager: Optional[LLMManager] = None,
                 document_processor: Optional[DocumentProcessor] = None,
                 chunk_processor: Optional[ChunkProcessor] = None,
                 text_splitter: Optional = None,
                 document_store: Optional = None):
        """
        Initialize RAG pipeline with pre-initialized components.
        
        Args:
            config: Configuration object
            vector_manager: Pre-initialized vector store manager
            llm_manager: Pre-initialized LLM manager
            document_processor: Pre-initialized document processor
            chunk_processor: Pre-initialized chunk processor
            text_splitter: Pre-initialized text splitter
        """
        self.config = config or get_config()
        self.logger = structlog.get_logger(self.__class__.__name__)
        
        # Use pre-initialized components if provided
        self._vector_manager = vector_manager
        self._llm_manager = llm_manager
        self._document_processor = document_processor
        self._chunk_processor = chunk_processor
        self._text_splitter = text_splitter
        self._document_store = document_store
        
        # Track initialization
        self._initialized = self._check_initialization()
        
        if self._initialized:
            self.logger.info("RAG pipeline initialized with pre-configured components")
        else:
            self.logger.warning("RAG pipeline initialized with missing components - some operations may fail")
    
    def _check_initialization(self) -> bool:
        """Check if all required components are available."""
        required_components = [
            self._vector_manager,
            self._llm_manager,
            self._document_processor,
            self._chunk_processor,
            self._text_splitter,
            self._document_store
        ]
        return all(component is not None for component in required_components)
    
    def _ensure_initialized(self) -> None:
        """Ensure pipeline is properly initialized before operations."""
        if not self._initialized:
            raise RuntimeError("RAG pipeline not properly initialized. All components must be provided.")
    
    def add_document(self, source: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a document to the knowledge base.
        
        Args:
            source: Path to document or URL
            metadata: Additional metadata for the document
            
        Returns:
            True if successful
        """
        try:
            self._ensure_initialized()
            
            self.logger.info("Adding document to knowledge base", source=source)
            
            # Create document record
            original_filename = metadata.get('original_filename') if metadata else None
            content_type = metadata.get('content_type') if metadata else None
            
            doc_id = self._document_store.create_document(
                source=source,
                original_filename=original_filename,
                content_type=content_type,
                metadata=metadata
            )
            
            # Load document
            loader_factory = DocumentLoaderFactory()
            documents = loader_factory.load_document(source)
            
            if not documents:
                self.logger.warning("No documents loaded", source=source)
                return False
            
            # Process documents and add document ID to metadata
            if metadata is None:
                metadata = {}
            metadata['doc_id'] = doc_id
            
            for doc in documents:
                doc.metadata.update(metadata)
            
            processed_docs = self._document_processor.process(documents, metadata)
            
            # Split into chunks
            chunks = self._text_splitter.split_documents(processed_docs)
            
            # Process chunks with metadata
            processed_chunks = self._chunk_processor.process_chunks(chunks)
            
            # Add to vector store
            vector_doc_ids = self._vector_manager.add_documents_batch(processed_chunks)
            
            # Update document stats
            total_chars = sum(len(chunk.page_content) for chunk in processed_chunks)
            self._document_store.update_document_stats(doc_id, len(processed_chunks), total_chars)
            
            # Persist if supported
            self._vector_manager.vector_store.persist()
            
            self.logger.info(
                "Successfully added document",
                doc_id=doc_id,
                source=source,
                chunks=len(processed_chunks),
                vector_doc_ids=len(vector_doc_ids)
            )
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to add document", source=source, error=str(e))
            return False
    
    def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add raw text to the knowledge base.
        
        Args:
            text: Text content to add
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        try:
            self._ensure_initialized()
            
            # Create document from text
            doc_metadata = metadata or {}
            doc_metadata.update({"source": "direct_text", "content_type": "text"})
            
            document = Document(page_content=text, metadata=doc_metadata)
            
            # Process and add
            processed_docs = self._document_processor.process([document], metadata)
            chunks = self._text_splitter.split_documents(processed_docs)
            processed_chunks = self._chunk_processor.process_chunks(chunks)
            
            doc_ids = self._vector_manager.add_documents_batch(processed_chunks)
            self._vector_manager.vector_store.persist()
            
            self.logger.info("Added text to knowledge base", chunks=len(processed_chunks))
            return True
            
        except Exception as e:
            self.logger.error("Failed to add text", error=str(e))
            return False
    
    def query(self, question: str, **kwargs) -> str:
        """Query the knowledge base and generate an answer.
        
        Args:
            question: User question
            **kwargs: Additional query parameters
            
        Returns:
            Generated answer
        """
        try:
            self._ensure_initialized()
            
            self.logger.info("Processing query", question_length=len(question))
            
            # Retrieve relevant documents
            k = kwargs.get('k', self.config.rag.retrieval_k)
            score_threshold = kwargs.get('score_threshold', self.config.rag.score_threshold)
            metadata_filter = kwargs.get('metadata_filter')
            
            retrieved_docs = self._vector_manager.search_with_filter(
                question,
                k=k,
                metadata_filter=metadata_filter,
                score_threshold=score_threshold
            )
            
            if not retrieved_docs:
                self.logger.warning("No relevant documents found")
                return "I couldn't find any relevant information to answer your question."
            
            # Create context from retrieved documents
            context = self._create_context(retrieved_docs)
            
            # Generate answer using LLM
            answer = self._generate_answer(question, context)
            
            self.logger.info(
                "Generated answer",
                question_length=len(question),
                context_length=len(context),
                answer_length=len(answer),
                retrieved_docs=len(retrieved_docs)
            )
            
            return answer
            
        except Exception as e:
            self.logger.error("Failed to process query", error=str(e))
            return "I encountered an error while processing your question. Please try again."
    
    def _create_context(self, documents: List[Document]) -> str:
        """Create context string from retrieved documents."""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content.strip()
            
            context_parts.append(f"Source {i} ({source}):\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using LLM with retrieved context."""
        
        # Create RAG prompt
        rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer the question based on the context provided
- If the context doesn't contain enough information to answer the question, say so
- Be concise but comprehensive
- Cite specific parts of the context when relevant

Answer:""")
        
        # Format prompt with context and question
        messages = rag_prompt.format_messages(
            context=context,
            question=question
        )
        
        # Generate answer
        return self._llm_manager.llm_provider.generate_with_messages(messages)
    
    def clear_knowledge_base(self) -> bool:
        """Clear all documents from the knowledge base.
        
        Returns:
            True if successful
        """
        try:
            self._ensure_initialized()
            
            self.logger.info("Clearing knowledge base")
            
            # Clear all documents from the vector store
            success = self._vector_manager.clear_all()
            
            if success:
                self.logger.info("Successfully cleared knowledge base")
            else:
                self.logger.error("Failed to clear knowledge base")
            
            return success
            
        except Exception as e:
            self.logger.error("Failed to clear knowledge base", error=str(e))
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get RAG pipeline status.
        
        Returns:
            Dictionary with status information
        """
        try:
            status = {
                "initialized": self._initialized,
                "components": {
                    "vector_manager": self._vector_manager is not None,
                    "llm_manager": self._llm_manager is not None,
                    "document_processor": self._document_processor is not None,
                    "chunk_processor": self._chunk_processor is not None,
                    "text_splitter": self._text_splitter is not None,
                },
                "config": {
                    "embedding_provider": self.config.embedding.provider,
                    "embedding_model": self.config.embedding.embedding_model_name,
                    "llm_provider": self.config.llm.provider,
                    "llm_model": self.config.llm.llm_model_name,
                    "vector_store": self.config.vectorstore.provider,
                }
            }
            
            if self._vector_manager:
                status["vector_store_stats"] = self._vector_manager.get_collection_stats()
            
            return status
            
        except Exception as e:
            return {
                "initialized": False,
                "error": str(e)
            }
    
    @classmethod
    def from_config(cls, config: Optional[StudyBotConfig] = None) -> "RAGPipeline":
        """Create RAG pipeline from configuration.
        
        Args:
            config: Optional configuration object
            
        Returns:
            Configured RAG pipeline
        """
        # This method is kept for backward compatibility but now requires
        # components to be initialized externally
        pipeline = cls(config)
        return pipeline
