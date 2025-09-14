#!/usr/bin/env python3
"""
Discord RAG Bot - Upload documents and ask questions about them via Discord.

Usage:
    python discord_bot.py [--token BOT_TOKEN] [--backend-url BACKEND_URL]

Features:
    - Upload documents (PDF, Word, Text, Web URLs) via Discord
    - Ask questions about uploaded documents
    - View document collection status
    - Clear knowledge base
"""

import os
import sys
import requests
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
from io import BytesIO

# Add the project root to Python path  
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import discord
from discord.ext import commands
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


class RAGDiscordBot:
    """Discord bot interface for RAG system using FastAPI backend."""
    
    def __init__(self, bot_token: str, backend_url: str = None):
        self.bot_token = bot_token
        self.backend_url = backend_url or os.getenv('BACKEND_API_URL', 'http://localhost:8000')
        self.session = requests.Session()
        
        # Set up Discord bot
        intents = discord.Intents.default()
        intents.message_content = True  # Enable message content intent for better UX
        
        self.bot = commands.Bot(command_prefix='!', intents=intents)
        self.setup_commands()
        
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Make HTTP request to backend API."""
        url = f"{self.backend_url}{endpoint}"
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json() if response.content else None
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to backend at {self.backend_url}")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error: {e}")
            if response.content:
                try:
                    error_detail = response.json()
                    logger.error(f"Details: {error_detail.get('detail', 'Unknown error')}")
                except:
                    logger.error(f"Response: {response.text}")
            return None
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None
    
    def setup_commands(self):
        """Set up Discord bot commands."""
        
        @self.bot.event
        async def on_ready():
            logger.info(f'{self.bot.user} has connected to Discord!')
            logger.info(f'Backend URL: {self.backend_url}')
            
            # Check backend health
            if self.check_backend_health():
                logger.info("✅ Backend connection established!")
            else:
                logger.error("❌ Backend not available. Please start the FastAPI backend.")
        
        @self.bot.command(name='ping')
        async def ping(ctx):
            """Test if the bot is responsive."""
            await ctx.send('Pong! 🏓')
        
        @self.bot.command(name='status')
        async def status(ctx):
            """Show knowledge base status."""
            await ctx.send("🔍 Checking status...")
            
            try:
                result = self._make_request('GET', '/status')
                
                if result:
                    status_msg = "📊 **Backend Status:**\n"
                    status_msg += f"API Status: {'✅' if result.get('api_status') == 'running' else '❌'}\n"
                    
                    if 'initialized' in result:
                        status_msg += f"RAG Pipeline: {'✅' if result['initialized'] else '❌'}\n"
                    
                    if 'components' in result:
                        components = result['components']
                        status_msg += f"Embedding Provider: {'✅' if components.get('embedding_provider') else '❌'}\n"
                        status_msg += f"Vector Store: {'✅' if components.get('vector_store') else '❌'}\n"
                        status_msg += f"LLM Manager: {'✅' if components.get('llm_manager') else '❌'}\n"
                    
                    if 'config' in result:
                        config = result['config']
                        status_msg += f"\n**Configuration:**\n"
                        status_msg += f"LLM: {config.get('llm_provider', 'Unknown')} ({config.get('llm_model', 'Unknown')})\n"
                        status_msg += f"Embeddings: {config.get('embedding_provider', 'Unknown')} ({config.get('embedding_model', 'Unknown')})\n"
                        status_msg += f"Vector Store: {config.get('vector_store', 'Unknown')}\n"
                    
                    # Get document count
                    docs_result = self._make_request('GET', '/documents/')
                    if docs_result:
                        status_msg += f"\n**Documents:** {docs_result.get('total_count', 0)} documents in knowledge base"
                    
                    await ctx.send(status_msg)
                else:
                    await ctx.send("❌ Failed to get status from backend")
                    
            except Exception as e:
                await ctx.send(f"❌ Error getting status: {e}")
        
        @self.bot.command(name='upload')
        async def upload(ctx, url: str = None):
            """Upload a document or add a URL to the knowledge base."""
            if url:
                # Handle URL upload
                if not url.startswith(('http://', 'https://')):
                    await ctx.send("❌ URL must start with http:// or https://")
                    return
                
                await ctx.send(f"📤 Adding web page: {url}")
                
                try:
                    data = {'source': url}
                    result = self._make_request('POST', '/documents/', json=data)
                    
                    if result and result.get('success'):
                        await ctx.send("✅ Successfully added web page")
                    else:
                        await ctx.send("❌ Failed to add web page")
                except Exception as e:
                    await ctx.send(f"❌ Error adding web page: {e}")
            else:
                # Check for file attachment
                if not ctx.message.attachments:
                    await ctx.send("📄 Please attach a file or provide a URL. Usage: `!upload [url]` or attach a file with `!upload`")
                    return
                
                attachment = ctx.message.attachments[0]
                await ctx.send(f"📤 Uploading document: {attachment.filename}")
                
                try:
                    # Download the file
                    file_data = await attachment.read()
                    
                    # Upload to backend
                    files = {'file': (attachment.filename, BytesIO(file_data), 'application/octet-stream')}
                    result = self._make_request('POST', '/documents/upload', files=files)
                    
                    if result and result.get('success'):
                        await ctx.send(f"✅ Successfully uploaded document: {attachment.filename}")
                    else:
                        await ctx.send(f"❌ Failed to upload document: {attachment.filename}")
                except Exception as e:
                    await ctx.send(f"❌ Error uploading document: {e}")
        
        @self.bot.command(name='text')
        async def add_text(ctx, *, text: str):
            """Add raw text to the knowledge base."""
            if not text.strip():
                await ctx.send("❌ No text provided. Usage: `!text [your text here]`")
                return
            
            await ctx.send("📤 Adding text to knowledge base...")
            
            try:
                data = {'text': text}
                result = self._make_request('POST', '/documents/text', json=data)
                
                if result and result.get('success'):
                    await ctx.send("✅ Successfully added text")
                else:
                    await ctx.send("❌ Failed to add text")
            except Exception as e:
                await ctx.send(f"❌ Error adding text: {e}")
        
        @self.bot.command(name='clear')
        async def clear_knowledge_base(ctx):
            """Clear the knowledge base."""
            await ctx.send("⚠️ Are you sure you want to clear the knowledge base? Type `!confirm` to proceed.")
            
            def check(message):
                return message.author == ctx.author and message.channel == ctx.channel and message.content.lower() == '!confirm'
            
            try:
                await self.bot.wait_for('message', timeout=30.0, check=check)
                
                result = self._make_request('POST', '/documents/clear')
                if result is not None:  # 204 status code returns None
                    await ctx.send("✅ Knowledge base cleared")
                else:
                    await ctx.send("❌ Failed to clear knowledge base")
            except asyncio.TimeoutError:
                await ctx.send("❌ Confirmation timeout. Knowledge base not cleared.")
            except Exception as e:
                await ctx.send(f"❌ Error clearing knowledge base: {e}")
        
        @self.bot.command(name='help')
        async def help_command(ctx):
            """Show help information."""
            help_msg = """📖 **StudyBot RAG Commands:**

**Document Management:**
• `!upload [url]` - Add web page to knowledge base
• `!upload` (with file attachment) - Upload document
• `!text [text]` - Add raw text to knowledge base
• `!clear` - Clear knowledge base (requires confirmation)

**Information:**
• `!status` - Show knowledge base status
• `!help` - Show this help
• `!ping` - Test bot responsiveness

**Querying:**
• just type your question directly!

💡 **Usage Tips:**
• Upload documents first with `!upload`
• Then ask questions with `!ask [question]` or just type your question
• Supported formats: PDF, Word, Text, Web pages
• The system will search and provide answers based on your documents"""
            
            await ctx.send(help_msg)
        
        @self.bot.event
        async def on_message(message):
            """Handle all messages, including queries."""
            # Don't respond to bot's own messages
            if message.author == self.bot.user:
                return
            
            # Process commands first
            await self.bot.process_commands(message)
            
            # If it's not a command and not empty, treat as a query
            if not message.content.startswith('!') and message.content.strip():
                await self.process_query(message.channel, message.content.strip())
    
    def check_backend_health(self) -> bool:
        """Check if backend is healthy and ready."""
        result = self._make_request('GET', '/health')
        if result:
            logger.info(f"Backend is healthy: {result.get('message', 'OK')}")
            return True
        return False
    
    async def process_query(self, ctx_or_channel, question: str):
        """Process a question against the knowledge base."""
        # Handle both command context and channel context
        if hasattr(ctx_or_channel, 'channel'):
            channel = ctx_or_channel.channel
            send_func = ctx_or_channel.send
        else:
            channel = ctx_or_channel
            send_func = ctx_or_channel.send
        
        # Send typing indicator
        async with channel.typing():
            try:
                data = {'question': question}
                result = self._make_request('POST', '/rag/query', json=data)
                
                if result:
                    answer = result.get('answer', 'No answer generated')
                    processing_time = result.get('processing_time_ms', 0)
                    
                    # Split long responses if needed (Discord has 2000 char limit)
                    if len(answer) > 1900:
                        parts = [answer[i:i+1900] for i in range(0, len(answer), 1900)]
                        for i, part in enumerate(parts):
                            if i == 0:
                                await send_func(part)
                            else:
                                await send_func(f"(continued...)\n{part}")
                    else:
                        await send_func(answer)
                else:
                    await send_func("❌ Failed to get answer from backend")
            except Exception as e:
                await send_func(f"❌ Error processing query: {e}")
    
    def run(self):
        """Run the Discord bot."""
        logger.info("Starting Discord RAG Bot...")
        self.bot.run(self.bot_token)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Discord RAG Bot for StudyBot")
    parser.add_argument("--token", 
                       default=os.getenv('DISCORD_BOT_TOKEN'),
                       help="Discord bot token (defaults to DISCORD_BOT_TOKEN env var)")
    parser.add_argument("--backend-url", 
                       default=os.getenv('BACKEND_API_URL', 'http://localhost:8000'),
                       help="Backend API URL (defaults to BACKEND_API_URL env var or localhost:8000)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Debug: Check if token is available
    if not args.token:
        logger.error("❌ Discord bot token is required. Set DISCORD_BOT_TOKEN environment variable or use --token")
        logger.error(f"Environment variable DISCORD_BOT_TOKEN: {'SET' if os.getenv('DISCORD_BOT_TOKEN') else 'NOT SET'}")
        sys.exit(1)
    
    logger.info("✅ Discord bot token found")
    
    # Set log level
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run Discord bot
    bot = RAGDiscordBot(bot_token=args.token, backend_url=args.backend_url)
    bot.run()


if __name__ == "__main__":
    main()