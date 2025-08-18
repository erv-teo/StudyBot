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
    
    def __init__(self, config: Optional[StudyBotConfig] = None):
        self.config = config or get_config()
        self.logger = structlog.get_logger(self.__class__.__name__)
        
        # Initialize components
        self._embedding_provider = None
        self._vector_store = None
        self._vector_manager = None
        self._llm_manager = None
        self._document_processor = None
        self._chunk_processor = None
        self._text_splitter = None
        
        # Track initialization
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize all RAG components."""
        if self._initialized:
            return
        
        try:
            self.logger.info("Initializing RAG pipeline...")
            
            # Initialize embedding provider
            self._embedding_provider = EmbeddingProviderFactory.create_provider(
                self.config.embedding.provider,
                model_name=self.config.embedding.embedding_model_name,
                model_kwargs=self.config.embedding.model_kwargs,
                encode_kwargs=self.config.embedding.encode_kwargs
            )
            
            # Initialize vector store
            embeddings = self._embedding_provider.get_embeddings()
            self._vector_store = VectorStoreFactory.create_store(
                self.config.vectorstore.provider,
                embeddings,
                persist_directory=self.config.vectorstore.persist_directory,
                collection_name=self.config.vectorstore.collection_name
            )
            
            # Initialize vector store manager
            self._vector_manager = VectorStoreManager(self._vector_store)
            
            # Initialize LLM
            llm_provider = LLMProviderFactory.create_provider(
                self.config.llm.provider,
                model_name=self.config.llm.llm_model_name,
                api_key=self.config.llm.api_key,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                base_url=self.config.llm.base_url
            )
            self._llm_manager = LLMManager(llm_provider)
            
            # Initialize document processing components
            self._document_processor = DocumentProcessor()
            self._chunk_processor = ChunkProcessor()
            self._text_splitter = TextSplitterFactory.create_splitter(
                "recursive",
                chunk_size=self.config.chunking.chunk_size,
                chunk_overlap=self.config.chunking.chunk_overlap
            )
            
            self._initialized = True
            self.logger.info("RAG pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error("Failed to initialize RAG pipeline", error=str(e))
            raise
    
    def add_document(self, source: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a document to the knowledge base.
        
        Args:
            source: Path to document or URL
            metadata: Additional metadata for the document
            
        Returns:
            True if successful
        """
        try:
            self.initialize()
            
            self.logger.info("Adding document to knowledge base", source=source)
            
            # Load document
            loader_factory = DocumentLoaderFactory()
            documents = loader_factory.load_document(source)
            
            if not documents:
                self.logger.warning("No documents loaded", source=source)
                return False
            
            # Process documents
            if metadata:
                for doc in documents:
                    doc.metadata.update(metadata)
            
            processed_docs = self._document_processor.process(documents, metadata)
            
            # Split into chunks
            chunks = self._text_splitter.split_documents(processed_docs)
            
            # Process chunks with metadata
            processed_chunks = self._chunk_processor.process_chunks(chunks)
            
            # Add to vector store
            doc_ids = self._vector_manager.add_documents_batch(processed_chunks)
            
            # Persist if supported
            self._vector_store.persist()
            
            self.logger.info(
                "Successfully added document",
                source=source,
                chunks=len(processed_chunks),
                doc_ids=len(doc_ids)
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
            self.initialize()
            
            # Create document from text
            doc_metadata = metadata or {}
            doc_metadata.update({"source": "direct_text", "content_type": "text"})
            
            document = Document(page_content=text, metadata=doc_metadata)
            
            # Process and add
            processed_docs = self._document_processor.process([document], metadata)
            chunks = self._text_splitter.split_documents(processed_docs)
            processed_chunks = self._chunk_processor.process_chunks(chunks)
            
            doc_ids = self._vector_manager.add_documents_batch(processed_chunks)
            self._vector_store.persist()
            
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
            self.initialize()
            
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
            self.initialize()
            
            # Note: This is a simplified implementation
            # In practice, you'd need to track document IDs for proper deletion
            self.logger.info("Clearing knowledge base")
            
            # Re-initialize vector store (effectively clearing it)
            embeddings = self._embedding_provider.get_embeddings()
            self._vector_store = VectorStoreFactory.create_store(
                self.config.vectorstore.provider,
                embeddings,
                persist_directory=self.config.vectorstore.persist_directory,
                collection_name=self.config.vectorstore.collection_name + "_new"
            )
            self._vector_manager = VectorStoreManager(self._vector_store)
            
            return True
            
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
                    "embedding_provider": self._embedding_provider is not None,
                    "vector_store": self._vector_store is not None,
                    "llm_manager": self._llm_manager is not None,
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
        pipeline = cls(config)
        pipeline.initialize()
        return pipeline
