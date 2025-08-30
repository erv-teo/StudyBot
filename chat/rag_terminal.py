#!/usr/bin/env python3
"""
Terminal RAG interface - upload documents and ask questions about them.

Usage:
    python rag_terminal.py [--provider openai|anthropic|ollama|google] [--model MODEL_NAME]

Features:
    - Upload documents (PDF, Word, Text, Web URLs)
    - Ask questions about uploaded documents
    - View document collection status
    - Clear knowledge base
"""

# Clean imports - no hacky environment variable setting

import argparse
import sys
import os
import requests
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Add the project root to Python path  
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class RAGTerminal:
    """Terminal interface for RAG system using FastAPI backend."""
    
    def __init__(self, backend_url: str = None):
        self.backend_url = backend_url or os.getenv('BACKEND_API_URL', 'http://localhost:8000')
        self.session = requests.Session()
        
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Make HTTP request to backend API."""
        url = f"{self.backend_url}{endpoint}"
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json() if response.content else None
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to backend at {self.backend_url}")
            print("💡 Make sure the FastAPI backend is running with: python -m core.main")
            return None
        except requests.exceptions.HTTPError as e:
            if response.status_code == 503:
                print("❌ Backend service not ready. Please wait for initialization to complete.")
            else:
                print(f"❌ HTTP Error: {e}")
                if response.content:
                    try:
                        error_detail = response.json()
                        print(f"   Details: {error_detail.get('detail', 'Unknown error')}")
                    except:
                        print(f"   Response: {response.text}")
            return None
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return None
    
    def check_backend_health(self) -> bool:
        """Check if backend is healthy and ready."""
        result = self._make_request('GET', '/health')
        if result:
            print(f"✅ Backend is healthy: {result.get('message', 'OK')}")
            return True
        return False
    
    def run(self):
        """Run the terminal RAG interface."""
        print("📚 StudyBot RAG Terminal")
        print("=" * 50)
        print(f"Backend URL: {self.backend_url}")
        
        print("\nCommands:")
        print("  /upload [path]    - Upload document or enter path interactively")
        print("  /url [url]        - Add web page to knowledge base")
        print("  /text [text]      - Add raw text to knowledge base")
        print("  /status           - Show knowledge base status")
        print("  /clear            - Clear knowledge base")
        print("  /help             - Show this help")
        print("  /quit             - Exit")
        print("\nType your question to search the knowledge base!")
        print("=" * 50)
        
        # Check backend health
        if not self.check_backend_health():
            print("\n💡 To start the backend, run: python -m core.main")
            return
        
        print("\n✅ Backend connection established!")
        self.show_status()
        print("\n💡 Start by uploading some documents with /upload")
        
        while True:
            try:
                user_input = input("\n📚 RAG: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if user_input == '/quit':
                        print("\n👋 Goodbye!")
                        break
                    elif user_input == '/help':
                        self.show_help()
                    elif user_input == '/status':
                        self.show_status()
                    elif user_input == '/clear':
                        self.clear_knowledge_base()
                    elif user_input.startswith('/upload'):
                        self.upload_document(user_input[7:].strip())
                    elif user_input.startswith('/url'):
                        self.add_url(user_input[4:].strip())
                    elif user_input.startswith('/text'):
                        self.add_text(user_input[5:].strip())
                    else:
                        print("❓ Unknown command. Type /help for available commands.")
                    continue
                
                # Process as query
                self.process_query(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break
    
    def upload_document(self, path: str = ""):
        """Upload a document to the knowledge base."""
        if not path:
            path = input("📄 Enter document path: ").strip()
        
        if not path:
            print("❌ No path provided")
            return
        
        path_obj = Path(path)
        if not path_obj.exists():
            print(f"❌ File not found: {path}")
            return
        
        print(f"📤 Uploading document: {path}")
        
        try:
            with open(path_obj, 'rb') as f:
                files = {'file': (path_obj.name, f, 'application/octet-stream')}
                result = self._make_request('POST', '/documents/upload', files=files)
            
            if result and result.get('success'):
                print(f"✅ Successfully uploaded document: {path_obj.name}")
            else:
                print(f"❌ Failed to upload document: {path_obj.name}")
        except Exception as e:
            print(f"❌ Error uploading document: {e}")
    
    def add_url(self, url: str = ""):
        """Add a web page to the knowledge base."""
        if not url:
            url = input("🌐 Enter URL: ").strip()
        
        if not url:
            print("❌ No URL provided")
            return
        
        if not url.startswith(('http://', 'https://')):
            print("❌ URL must start with http:// or https://")
            return
        
        print(f"📤 Adding web page: {url}")
        
        try:
            data = {'source': url}
            result = self._make_request('POST', '/documents/', json=data)
            
            if result and result.get('success'):
                print(f"✅ Successfully added web page")
            else:
                print(f"❌ Failed to add web page")
        except Exception as e:
            print(f"❌ Error adding web page: {e}")
    
    def add_text(self, text: str = ""):
        """Add raw text to the knowledge base."""
        if not text:
            print("📝 Enter text (press Ctrl+D when done):")
            lines = []
            try:
                while True:
                    line = input()
                    lines.append(line)
            except EOFError:
                text = "\n".join(lines)
        
        if not text.strip():
            print("❌ No text provided")
            return
        
        print("📤 Adding text to knowledge base...")
        
        try:
            data = {'text': text}
            result = self._make_request('POST', '/documents/text', json=data)
            
            if result and result.get('success'):
                print("✅ Successfully added text")
            else:
                print("❌ Failed to add text")
        except Exception as e:
            print(f"❌ Error adding text: {e}")
    
    def process_query(self, question: str):
        """Process a question against the knowledge base."""
        print("\n🔍 Searching knowledge base...")
        
        try:
            data = {'question': question}
            result = self._make_request('POST', '/rag/query', json=data)
            
            if result:
                answer = result.get('answer', 'No answer generated')
                processing_time = result.get('processing_time_ms', 0)
                
                print(f"\n🤖 Answer:")
                print("-" * 40)
                print(answer)
                print("-" * 40)
                if processing_time:
                    print(f"⏱️  Processing time: {processing_time:.1f}ms")
            else:
                print("❌ Failed to get answer from backend")
        except Exception as e:
            print(f"❌ Error processing query: {e}")
    
    def show_status(self):
        """Show knowledge base status."""
        try:
            result = self._make_request('GET', '/status')
            
            if result:
                print("\n📊 Backend Status:")
                print("-" * 30)
                print(f"API Status: {'✅' if result.get('api_status') == 'running' else '❌'}")
                
                if 'initialized' in result:
                    print(f"RAG Pipeline: {'✅' if result['initialized'] else '❌'}")
                
                if 'components' in result:
                    components = result['components']
                    print(f"Embedding Provider: {'✅' if components.get('embedding_provider') else '❌'}")
                    print(f"Vector Store: {'✅' if components.get('vector_store') else '❌'}")
                    print(f"LLM Manager: {'✅' if components.get('llm_manager') else '❌'}")
                
                if 'config' in result:
                    config = result['config']
                    print(f"\nConfiguration:")
                    print(f"  LLM: {config.get('llm_provider', 'Unknown')} ({config.get('llm_model', 'Unknown')})")
                    print(f"  Embeddings: {config.get('embedding_provider', 'Unknown')} ({config.get('embedding_model', 'Unknown')})")
                    print(f"  Vector Store: {config.get('vector_store', 'Unknown')}")
                
                # Get document count
                docs_result = self._make_request('GET', '/documents/')
                if docs_result:
                    print(f"\nDocuments: {docs_result.get('total_count', 0)} documents in knowledge base")
            
        except Exception as e:
            print(f"❌ Error getting status: {e}")
    
    def clear_knowledge_base(self):
        """Clear the knowledge base."""
        response = input("⚠️  Are you sure you want to clear the knowledge base? (y/N): ")
        if response.lower() != 'y':
            print("❌ Cancelled")
            return
        
        try:
            result = self._make_request('POST', '/documents/clear')
            if result is not None:  # 204 status code returns None
                print("✅ Knowledge base cleared")
            else:
                print("❌ Failed to clear knowledge base")
        except Exception as e:
            print(f"❌ Error clearing knowledge base: {e}")
    
    def show_help(self):
        """Show help information."""
        print("\n📖 Help:")
        print("  /upload [path]    - Upload document (PDF, Word, Text)")
        print("  /url [url]        - Add web page to knowledge base")
        print("  /text [text]      - Add raw text to knowledge base")
        print("  /status           - Show knowledge base status")
        print("  /clear            - Clear knowledge base")
        print("  /help             - Show this help")
        print("  /quit             - Exit")
        print("\n💡 Usage Tips:")
        print("  - Upload documents first with /upload")
        print("  - Then ask questions about the documents")
        print("  - The system will search and provide answers based on your documents")
        print("  - Supported formats: PDF, Word, Text, Web pages")
        print(f"  - Backend URL: {self.backend_url}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Terminal RAG interface for StudyBot")
    parser.add_argument("--backend-url", 
                       default=os.getenv('BACKEND_API_URL', 'http://localhost:8000'),
                       help="Backend API URL (defaults to BACKEND_API_URL env var or localhost:8000)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run RAG interface
    rag = RAGTerminal(backend_url=args.backend_url)
    rag.run()


if __name__ == "__main__":
    main()
