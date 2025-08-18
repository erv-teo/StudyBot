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
from pathlib import Path

# Add the project root to Python path  
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.rag import RAGPipeline
from core.config.settings import get_config
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
    """Terminal interface for RAG system."""
    
    def __init__(self, provider_type: str = None, model_name: str = None):
        self.config = get_config()
        self.provider_type = provider_type or self.config.llm.provider
        self.model_name = model_name
        self.rag_pipeline = None
        
    def initialize(self):
        """Initialize the RAG pipeline."""
        try:
            # Update config with command line arguments if provided
            if self.provider_type != self.config.llm.provider:
                self.config.llm.provider = self.provider_type
            
            if self.model_name:
                self.config.llm.llm_model_name = self.model_name
            
            # Create and initialize RAG pipeline
            self.rag_pipeline = RAGPipeline(self.config)
            self.rag_pipeline.initialize()
            
            logger.info("Initialized RAG pipeline", 
                       provider=self.provider_type, 
                       model=self.config.llm.llm_model_name)
            
        except Exception as e:
            logger.error("Failed to initialize RAG pipeline", error=str(e))
            raise
    
    def run(self):
        """Run the terminal RAG interface."""
        print("📚 StudyBot RAG Terminal")
        print("=" * 50)
        print(f"LLM Provider: {self.provider_type}")
        print(f"LLM Model: {self.model_name or self.config.llm.llm_model_name}")
        print(f"Vector Store: {self.config.vectorstore.provider}")
        print(f"Embeddings: {self.config.embedding.provider}")
        
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
        
        try:
            self.initialize()
        except Exception as e:
            print(f"\n❌ Failed to initialize: {e}")
            print("\nTips:")
            print("1. Make sure you have set your API key in .env file")
            print("2. For OpenAI: LLM__API_KEY=your_openai_api_key")
            print("3. For Anthropic: LLM__API_KEY=your_anthropic_api_key")
            print("4. For Ollama: Make sure Ollama is running locally")
            return
        
        print("\n✅ RAG pipeline initialized successfully!")
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
            success = self.rag_pipeline.add_document(path)
            if success:
                print(f"✅ Successfully added document: {path_obj.name}")
            else:
                print(f"❌ Failed to add document: {path_obj.name}")
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
            success = self.rag_pipeline.add_document(url)
            if success:
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
            success = self.rag_pipeline.add_text(text)
            if success:
                print("✅ Successfully added text")
            else:
                print("❌ Failed to add text")
        except Exception as e:
            print(f"❌ Error adding text: {e}")
    
    def process_query(self, question: str):
        """Process a question against the knowledge base."""
        print("\n🔍 Searching knowledge base...")
        
        try:
            answer = self.rag_pipeline.query(question)
            print(f"\n🤖 Answer:")
            print("-" * 40)
            print(answer)
            print("-" * 40)
        except Exception as e:
            print(f"❌ Error processing query: {e}")
    
    def show_status(self):
        """Show knowledge base status."""
        try:
            status = self.rag_pipeline.get_status()
            
            print("\n📊 Knowledge Base Status:")
            print("-" * 30)
            print(f"Initialized: {'✅' if status['initialized'] else '❌'}")
            
            if status['initialized']:
                components = status['components']
                print(f"Embedding Provider: {'✅' if components['embedding_provider'] else '❌'}")
                print(f"Vector Store: {'✅' if components['vector_store'] else '❌'}")
                print(f"LLM Manager: {'✅' if components['llm_manager'] else '❌'}")
                
                config = status['config']
                print(f"\nConfiguration:")
                print(f"  LLM: {config['llm_provider']} ({config['llm_model']})")
                print(f"  Embeddings: {config['embedding_provider']} ({config['embedding_model']})")
                print(f"  Vector Store: {config['vector_store']}")
                
                if 'vector_store_stats' in status:
                    stats = status['vector_store_stats']
                    print(f"\nVector Store:")
                    print(f"  Type: {stats.get('type', 'Unknown')}")
                    print(f"  Initialized: {'✅' if stats.get('initialized', False) else '❌'}")
            
        except Exception as e:
            print(f"❌ Error getting status: {e}")
    
    def clear_knowledge_base(self):
        """Clear the knowledge base."""
        response = input("⚠️  Are you sure you want to clear the knowledge base? (y/N): ")
        if response.lower() != 'y':
            print("❌ Cancelled")
            return
        
        try:
            success = self.rag_pipeline.clear_knowledge_base()
            if success:
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


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Terminal RAG interface for StudyBot")
    parser.add_argument("--provider", choices=["openai", "anthropic", "ollama", "google"], 
                       default=None, help="LLM provider to use (defaults to .env setting)")
    parser.add_argument("--model", help="Model name to use")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get config to determine default provider
    config = get_config()
    provider = args.provider or config.llm.provider
    
    # Create and run RAG interface
    rag = RAGTerminal(provider_type=provider, model_name=args.model)
    rag.run()


if __name__ == "__main__":
    main()
