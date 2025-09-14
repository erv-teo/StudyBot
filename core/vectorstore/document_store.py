"""Document store for managing document metadata and IDs."""

import uuid
import json
import sqlite3
from typing import Dict, List, Optional, Any
from datetime import datetime
import structlog
from pathlib import Path

logger = structlog.get_logger(__name__)


class DocumentStore:
    """Manages document metadata and provides document ID tracking."""
    
    def __init__(self, persist_directory: str = "./data"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.db_path = self.persist_directory / "documents.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database for document storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    original_filename TEXT,
                    content_type TEXT,
                    total_chunks INTEGER DEFAULT 0,
                    total_chars INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            conn.commit()
    
    def create_document(self, 
                       source: str, 
                       original_filename: Optional[str] = None,
                       content_type: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new document and return its ID.
        
        Args:
            source: Original source (file path, URL, etc.)
            original_filename: Original filename if from file upload
            content_type: Type of content
            metadata: Additional metadata
            
        Returns:
            Document ID
        """
        doc_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO documents 
                (doc_id, source, original_filename, content_type, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_id,
                source,
                original_filename,
                content_type,
                now,
                now,
                json.dumps(metadata or {})
            ))
            conn.commit()
        
        logger.info("Created document", doc_id=doc_id, source=source, original_filename=original_filename)
        return doc_id
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document metadata or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT doc_id, source, original_filename, content_type, 
                       total_chunks, total_chars, created_at, updated_at, metadata
                FROM documents WHERE doc_id = ?
            """, (doc_id,))
            row = cursor.fetchone()
            
            if row:
                return {
                    'doc_id': row[0],
                    'source': row[1],
                    'original_filename': row[2],
                    'content_type': row[3],
                    'total_chunks': row[4],
                    'total_chars': row[5],
                    'created_at': row[6],
                    'updated_at': row[7],
                    'metadata': json.loads(row[8]) if row[8] else {}
                }
            return None
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents.
        
        Returns:
            List of document metadata
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT doc_id, source, original_filename, content_type, 
                       total_chunks, total_chars, created_at, updated_at, metadata
                FROM documents ORDER BY created_at DESC
            """)
            
            documents = []
            for row in cursor.fetchall():
                documents.append({
                    'doc_id': row[0],
                    'source': row[1],
                    'original_filename': row[2],
                    'content_type': row[3],
                    'total_chunks': row[4],
                    'total_chars': row[5],
                    'created_at': row[6],
                    'updated_at': row[7],
                    'metadata': json.loads(row[8]) if row[8] else {}
                })
            
            return documents
    
    def update_document_stats(self, doc_id: str, total_chunks: int, total_chars: int):
        """Update document statistics.
        
        Args:
            doc_id: Document ID
            total_chunks: Total number of chunks
            total_chars: Total number of characters
        """
        now = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE documents 
                SET total_chunks = ?, total_chars = ?, updated_at = ?
                WHERE doc_id = ?
            """, (total_chunks, total_chars, now, doc_id))
            conn.commit()
        
        logger.info("Updated document stats", doc_id=doc_id, total_chunks=total_chunks, total_chars=total_chars)
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if successful
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            conn.commit()
            
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info("Deleted document", doc_id=doc_id)
            
            return deleted
    
    def get_document_by_source(self, source: str) -> Optional[Dict[str, Any]]:
        """Get document by source path.
        
        Args:
            source: Source path
            
        Returns:
            Document metadata or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT doc_id, source, original_filename, content_type, 
                       total_chunks, total_chars, created_at, updated_at, metadata
                FROM documents WHERE source = ?
            """, (source,))
            row = cursor.fetchone()
            
            if row:
                return {
                    'doc_id': row[0],
                    'source': row[1],
                    'original_filename': row[2],
                    'content_type': row[3],
                    'total_chunks': row[4],
                    'total_chars': row[5],
                    'created_at': row[6],
                    'updated_at': row[7],
                    'metadata': json.loads(row[8]) if row[8] else {}
                }
            return None
