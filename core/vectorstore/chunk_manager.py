"""Chunk management utilities for viewing and editing vector store content."""

from typing import List, Dict, Any, Optional, Tuple
import structlog
from langchain_core.documents import Document
from datetime import datetime
import sqlite3

from .base import VectorStoreManager

logger = structlog.get_logger(__name__)


class ChunkManager:
    """Enhanced chunk viewing and editing capabilities for vector stores."""
    
    def __init__(self, vector_manager: VectorStoreManager, embedding_provider=None):
        self.vector_manager = vector_manager
        self.embedding_provider = embedding_provider
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def list_chunks(self, 
                   source_filter: Optional[str] = None,
                   doc_id_filter: Optional[str] = None, 
                   section_filter: Optional[str] = None,
                   content_preview_length: int = 100,
                   limit: int = 50) -> List[Dict]:
        """List chunks with metadata and content preview.
        
        Args:
            source_filter: Filter by source path/URL
            doc_id_filter: Filter by document ID
            section_filter: Filter by section (beginning, middle, end)
            content_preview_length: Length of content preview
            limit: Maximum number of chunks to return
            
        Returns:
            List of chunk information dictionaries
        """
        try:
            # Get underlying vector store
            store = self.vector_manager.vector_store.get_store()
            
            # For ChromaDB, we can query the SQLite database directly
            if hasattr(store, '_client') and hasattr(store._client, '_db'):
                return self._list_chunks_chromadb(
                    source_filter, doc_id_filter, section_filter, 
                    content_preview_length, limit
                )
            else:
                # Fallback: use similarity search with dummy query
                return self._list_chunks_fallback(
                    source_filter, doc_id_filter, section_filter,
                    content_preview_length, limit
                )
                
        except Exception as e:
            self.logger.error("Failed to list chunks", error=str(e))
            return []
    
    def _list_chunks_chromadb(self, source_filter, doc_id_filter, section_filter, 
                             content_preview_length, limit) -> List[Dict]:
        """List chunks using direct ChromaDB SQLite access."""
        try:
            # Get the SQLite database path
            store = self.vector_manager.vector_store.get_store()
            persist_dir = self.vector_manager.vector_store.persist_directory
            db_path = f"{persist_dir}/chroma.sqlite3"
            
            # Build query with filters
            query = """
            SELECT DISTINCT e.embedding_id, e.created_at
            FROM embeddings e
            JOIN embedding_metadata em ON e.id = em.id
            WHERE 1=1
            """
            params = []
            
            if source_filter:
                query += " AND EXISTS (SELECT 1 FROM embedding_metadata em2 WHERE em2.id = e.id AND em2.key = 'source' AND em2.string_value LIKE ?)"
                params.append(f"%{source_filter}%")
            
            if doc_id_filter:
                query += " AND EXISTS (SELECT 1 FROM embedding_metadata em3 WHERE em3.id = e.id AND em3.key = 'doc_id' AND em3.string_value = ?)"
                params.append(doc_id_filter)
                
            if section_filter:
                query += " AND EXISTS (SELECT 1 FROM embedding_metadata em4 WHERE em4.id = e.id AND em4.key = 'section' AND em4.string_value = ?)"
                params.append(section_filter)
            
            query += f" ORDER BY e.created_at DESC LIMIT {limit}"
            
            # Execute query
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute(query, params)
                embedding_ids = cursor.fetchall()
            
            # Get full metadata for each embedding
            chunks = []
            for embedding_id, created_at in embedding_ids:
                chunk_info = self.get_chunk_details(embedding_id)
                if chunk_info:
                    # Add preview
                    content = chunk_info.get('content', '')
                    chunk_info['content_preview'] = (
                        content[:content_preview_length] + "..." 
                        if len(content) > content_preview_length 
                        else content
                    )
                    chunk_info['created_at'] = created_at
                    chunks.append(chunk_info)
            
            return chunks
            
        except Exception as e:
            self.logger.error("Failed to list chunks via ChromaDB", error=str(e))
            return []
    
    def _list_chunks_fallback(self, source_filter, doc_id_filter, section_filter,
                             content_preview_length, limit) -> List[Dict]:
        """Fallback method using similarity search."""
        try:
            # Use a generic query to get some results
            docs = self.vector_manager.vector_store.similarity_search("", k=limit)
            
            chunks = []
            for doc in docs:
                # Apply filters
                if source_filter and source_filter.lower() not in doc.metadata.get('source', '').lower():
                    continue
                if doc_id_filter and doc.metadata.get('doc_id') != doc_id_filter:
                    continue
                if section_filter and doc.metadata.get('section') != section_filter:
                    continue
                
                chunk_info = {
                    'chunk_id': doc.metadata.get('chunk_id', 'unknown'),
                    'content': doc.page_content,
                    'content_preview': (
                        doc.page_content[:content_preview_length] + "..." 
                        if len(doc.page_content) > content_preview_length 
                        else doc.page_content
                    ),
                    'metadata': doc.metadata
                }
                chunks.append(chunk_info)
            
            return chunks
            
        except Exception as e:
            self.logger.error("Failed to list chunks via fallback", error=str(e))
            return []
    
    def get_chunk_details(self, chunk_id: str) -> Optional[Dict]:
        """Get full details of a specific chunk.
        
        Args:
            chunk_id: The chunk ID or embedding ID
            
        Returns:
            Dictionary with full chunk information
        """
        try:
            store = self.vector_manager.vector_store.get_store()
            
            # For ChromaDB, query the metadata directly
            if hasattr(store, '_client'):
                return self._get_chunk_details_chromadb(chunk_id)
            else:
                return self._get_chunk_details_fallback(chunk_id)
                
        except Exception as e:
            self.logger.error("Failed to get chunk details", chunk_id=chunk_id, error=str(e))
            return None
    
    def _get_chunk_details_chromadb(self, chunk_id: str) -> Optional[Dict]:
        """Get chunk details from ChromaDB SQLite."""
        try:
            persist_dir = self.vector_manager.vector_store.persist_directory
            db_path = f"{persist_dir}/chroma.sqlite3"
            
            # First, find the embedding by chunk_id or embedding_id
            with sqlite3.connect(db_path) as conn:
                # Try to find by chunk_id first
                cursor = conn.execute("""
                    SELECT e.embedding_id, e.id 
                    FROM embeddings e
                    JOIN embedding_metadata em ON e.id = em.id
                    WHERE em.key = 'chunk_id' AND em.string_value = ?
                """, (chunk_id,))
                
                result = cursor.fetchone()
                if not result:
                    # Try to find by embedding_id
                    cursor = conn.execute("""
                        SELECT embedding_id, id FROM embeddings WHERE embedding_id = ?
                    """, (chunk_id,))
                    result = cursor.fetchone()
                
                if not result:
                    return None
                
                embedding_id, db_id = result
                
                # Get all metadata for this embedding
                cursor = conn.execute("""
                    SELECT key, string_value, int_value, float_value, bool_value
                    FROM embedding_metadata 
                    WHERE id = ?
                """, (db_id,))
                
                metadata = {}
                content = ""
                
                for key, str_val, int_val, float_val, bool_val in cursor.fetchall():
                    if key == "chroma:document":
                        content = str_val
                    elif str_val is not None:
                        metadata[key] = str_val
                    elif int_val is not None:
                        metadata[key] = int_val
                    elif float_val is not None:
                        metadata[key] = float_val
                    elif bool_val is not None:
                        metadata[key] = bool(bool_val)
                
                return {
                    'embedding_id': embedding_id,
                    'chunk_id': metadata.get('chunk_id', embedding_id),
                    'content': content,
                    'metadata': metadata,
                    'content_length': len(content),
                    'chunk_size': metadata.get('chunk_size', len(content))
                }
                
        except Exception as e:
            self.logger.error("Failed to get chunk details from ChromaDB", error=str(e))
            return None
    
    def _get_chunk_details_fallback(self, chunk_id: str) -> Optional[Dict]:
        """Fallback method for getting chunk details."""
        try:
            # Search for the chunk using metadata filter
            docs = self.vector_manager.search_with_filter(
                "", k=100, metadata_filter={'chunk_id': chunk_id}
            )
            
            if docs:
                doc = docs[0]
                return {
                    'chunk_id': chunk_id,
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'content_length': len(doc.page_content),
                    'chunk_size': doc.metadata.get('chunk_size', len(doc.page_content))
                }
            
            return None
            
        except Exception as e:
            self.logger.error("Failed to get chunk details via fallback", error=str(e))
            return None
    
    def search_chunks(self, 
                     text_query: str,
                     semantic: bool = True,
                     metadata_filter: Optional[Dict] = None,
                     limit: int = 10) -> List[Dict]:
        """Search chunks by content.
        
        Args:
            text_query: Text to search for
            semantic: Use semantic search (True) or text search (False)
            metadata_filter: Optional metadata filters
            limit: Maximum results
            
        Returns:
            List of matching chunks with relevance scores
        """
        try:
            if semantic and text_query.strip():
                # Use vector similarity search
                docs = self.vector_manager.search_with_filter(
                    text_query, k=limit, metadata_filter=metadata_filter
                )
                
                results = []
                for doc in docs:
                    results.append({
                        'chunk_id': doc.metadata.get('chunk_id', 'unknown'),
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'relevance_type': 'semantic'
                    })
                
                return results
            else:
                # Use text-based search (requires different approach)
                return self._text_search_chunks(text_query, metadata_filter, limit)
                
        except Exception as e:
            self.logger.error("Failed to search chunks", error=str(e))
            return []
    
    def _text_search_chunks(self, text_query: str, metadata_filter: Optional[Dict], limit: int) -> List[Dict]:
        """Text-based search in chunks."""
        try:
            # For ChromaDB, we can use the FTS table
            persist_dir = self.vector_manager.vector_store.persist_directory
            db_path = f"{persist_dir}/chroma.sqlite3"
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("""
                    SELECT em.id, em.string_value
                    FROM embedding_metadata em
                    WHERE em.key = 'chroma:document' 
                    AND em.string_value LIKE ?
                    LIMIT ?
                """, (f"%{text_query}%", limit))
                
                results = []
                for db_id, content in cursor.fetchall():
                    # Get chunk metadata
                    meta_cursor = conn.execute("""
                        SELECT key, string_value, int_value 
                        FROM embedding_metadata 
                        WHERE id = ? AND key != 'chroma:document'
                    """, (db_id,))
                    
                    metadata = {}
                    for key, str_val, int_val in meta_cursor.fetchall():
                        metadata[key] = str_val if str_val is not None else int_val
                    
                    results.append({
                        'chunk_id': metadata.get('chunk_id', f'db_{db_id}'),
                        'content': content,
                        'metadata': metadata,
                        'relevance_type': 'text_match'
                    })
                
                return results
                
        except Exception as e:
            self.logger.error("Failed to perform text search", error=str(e))
            return []
    
    def edit_chunk(self, 
                   chunk_id: str, 
                   new_content: str,
                   update_metadata: Optional[Dict] = None) -> bool:
        """Edit chunk content and trigger re-embedding.
        
        Args:
            chunk_id: ID of chunk to edit
            new_content: New content for the chunk
            update_metadata: Optional metadata updates
            
        Returns:
            True if successful
        """
        try:
            if not self.embedding_provider:
                raise ValueError("Embedding provider required for chunk editing")
            
            # Get current chunk details
            chunk_details = self.get_chunk_details(chunk_id)
            if not chunk_details:
                raise ValueError(f"Chunk {chunk_id} not found")
            
            # Create updated document
            metadata = chunk_details['metadata'].copy()
            metadata['chunk_size'] = len(new_content)
            metadata['last_edited'] = datetime.now().isoformat()
            
            if update_metadata:
                metadata.update(update_metadata)
            
            # Create new document with updated content
            updated_doc = Document(page_content=new_content, metadata=metadata)
            
            # Delete old chunk and add new one
            # Note: This is a simplified approach - in production you'd want more atomic operations
            self.delete_chunk(chunk_id)
            
            # Add updated chunk
            self.vector_manager.add_documents_batch([updated_doc])
            
            self.logger.info("Successfully edited chunk", chunk_id=chunk_id)
            return True
            
        except Exception as e:
            self.logger.error("Failed to edit chunk", chunk_id=chunk_id, error=str(e))
            return False
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """Remove a specific chunk.
        
        Args:
            chunk_id: ID of chunk to delete
            
        Returns:
            True if successful
        """
        try:
            # Get the embedding ID for this chunk
            chunk_details = self.get_chunk_details(chunk_id)
            if not chunk_details:
                self.logger.warning("Chunk not found for deletion", chunk_id=chunk_id)
                return False
            
            embedding_id = chunk_details['embedding_id']
            
            # Delete from vector store
            success = self.vector_manager.vector_store.delete([embedding_id])
            
            if success:
                self.logger.info("Successfully deleted chunk", chunk_id=chunk_id)
            
            return success
            
        except Exception as e:
            self.logger.error("Failed to delete chunk", chunk_id=chunk_id, error=str(e))
            return False
    
    def get_document_chunks(self, source: str) -> List[Dict]:
        """Get all chunks for a specific document.
        
        Args:
            source: Source path/URL of the document
            
        Returns:
            List of chunks for the document
        """
        return self.list_chunks(source_filter=source, limit=1000)
    
    def get_document_stats(self, source: str) -> Dict:
        """Get statistics about document chunks.
        
        Args:
            source: Source path/URL of the document
            
        Returns:
            Dictionary with document statistics
        """
        try:
            chunks = self.get_document_chunks(source)
            
            if not chunks:
                return {'error': 'Document not found', 'source': source}
            
            total_chunks = len(chunks)
            total_chars = sum(len(chunk.get('content', '')) for chunk in chunks)
            sections = {}
            
            for chunk in chunks:
                section = chunk.get('metadata', {}).get('section', 'unknown')
                sections[section] = sections.get(section, 0) + 1
            
            return {
                'source': source,
                'total_chunks': total_chunks,
                'total_characters': total_chars,
                'average_chunk_size': total_chars // total_chunks if total_chunks > 0 else 0,
                'sections': sections,
                'doc_id': chunks[0].get('metadata', {}).get('doc_id', 'unknown') if chunks else None
            }
            
        except Exception as e:
            self.logger.error("Failed to get document stats", source=source, error=str(e))
            return {'error': str(e), 'source': source}
    
    def validate_chunks(self, source: Optional[str] = None) -> List[Dict]:
        """Find problematic chunks.
        
        Args:
            source: Optional source filter
            
        Returns:
            List of problematic chunks with issues
        """
        try:
            chunks = self.list_chunks(source_filter=source, limit=1000)
            problems = []
            
            for chunk in chunks:
                issues = []
                content = chunk.get('content', '')
                metadata = chunk.get('metadata', {})
                
                # Check for issues
                if len(content) < 50:
                    issues.append('too_short')
                if len(content) > 2000:
                    issues.append('too_long')
                if not content.strip():
                    issues.append('empty')
                if len(content.split()) < 5:
                    issues.append('too_few_words')
                if not metadata.get('chunk_id'):
                    issues.append('missing_chunk_id')
                if not metadata.get('source'):
                    issues.append('missing_source')
                
                if issues:
                    problems.append({
                        'chunk_id': chunk.get('chunk_id', 'unknown'),
                        'issues': issues,
                        'content_length': len(content),
                        'source': metadata.get('source', 'unknown')
                    })
            
            return problems
            
        except Exception as e:
            self.logger.error("Failed to validate chunks", error=str(e))
            return []
