# StudyBot

A modular RAG (Retrieval Augmented Generation) system for question-answering based on document collections. Upload documents and ask questions about them using advanced AI.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone and enter the repository
cd StudyBot

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy the template to create your configuration
cp env_template.txt .env

# Edit .env file with your API keys
# Required: Add your OpenAI API key for LLM functionality
```

**Edit `.env` file:**

```bash
# Replace with your actual API key
LLM__API_KEY=your_openai_api_key_here

# Optional: Change providers/models
LLM__PROVIDER=openai
LLM__LLM_MODEL_NAME=gpt-3.5-turbo

# Include Telegram api key
TELEGRAM_BOT_KEY=mock_api_key
```

### 3. Start the RAG Terminal

```bash
python3 chat/rag_terminal.py
```

## 📚 Usage

### Basic Commands

Once the RAG terminal is running, you can use these commands:

```bash
# Upload documents
/upload path/to/your/document.pdf
/upload data/raw/sample_document_1.txt

# Add web pages
/url https://en.wikipedia.org/wiki/Tetris

# Add raw text
/text "Are we cooked as Software engineers in 2027? Maybe. (Source Trust me Bro)"

# Check status
/status

# Clear knowledge base
/clear

# Get help
/help

# Exit
/quit
```

### Example Workflow

1. **Upload a document:**

   ```bash
   📚 RAG: /upload data/raw/sample_document_1.txt
   ✅ Successfully added document: sample_document_1.txt
   ```

2. **Ask questions about the document:**

   ```bash
   📚 RAG: What is StudyBot?
   🤖 Answer: StudyBot is an advanced RAG system designed for question-answering...

   📚 RAG: What file formats are supported?
   🤖 Answer: The supported formats include PDF, Word, Text, and web pages...
   ```

3. **Check knowledge base status:**
   ```bash
   📚 RAG: /status
   📊 Knowledge Base Status:
   Initialized: ✅
   LLM: openai (gpt-3.5-turbo)
   Embeddings: sentence-transformers (all-MiniLM-L6-v2)
   Vector Store: chroma
   ```

## 🔧 Configuration Options

### LLM Providers

Choose your preferred LLM provider in `.env`:

```bash
# OpenAI (requires API key)
LLM__PROVIDER=openai
LLM__LLM_MODEL_NAME=gpt-3.5-turbo
LLM__API_KEY=your_openai_key

# Anthropic (requires API key)
LLM__PROVIDER=anthropic
LLM__LLM_MODEL_NAME=claude-3-sonnet-20240229
LLM__API_KEY=your_anthropic_key

# Local Ollama (free, runs locally)
LLM__PROVIDER=ollama
LLM__LLM_MODEL_NAME=llama2
LLM__BASE_URL=http://localhost:11434
```

### Supported File Formats

- **PDF** documents
- **Word** documents (.docx, .doc)
- **Text** files (.txt, .md)
- **Web pages** (via URL)
- **Raw text** (direct input)

## 🎯 Example Use Cases

- **Research Assistant**: Upload research papers and ask about methodologies
- **Document Q&A**: Upload company docs and query policies
- **Study Helper**: Upload textbooks and ask concept questions
- **Meeting Analysis**: Upload transcripts and extract action items

## 🛠️ Development

The system uses a modular architecture:

```
StudyBot/
├── core/                 # Platform-agnostic RAG engine
│   ├── config/          # Configuration management
│   ├── rag/             # RAG pipeline orchestration
│   ├── llm/             # LLM providers (OpenAI, Anthropic, Ollama)
│   ├── vectorstore/     # Vector stores (Chroma, FAISS, In-memory)
│   ├── chunking/        # Text splitters & embeddings
│   └── ingestion/       # Document loaders (PDF, Web, Text, etc.)
├── chat/                # Platform-specific interfaces
│   ├── rag_terminal.py  # Terminal interface
│   ├── telegram.py      # Future: Telegram bot
│   └── whatsapp.py      # Future: WhatsApp bot
└── requirements.txt
```

**Key Components:**

- `core/config/` - Configuration management
- `core/ingestion/` - Document loading (PDF, Web, Text, etc.)
- `core/chunking/` - Text splitting & embeddings
- `core/vectorstore/` - Vector databases (Chroma, FAISS)
- `core/llm/` - LLM providers (OpenAI, Anthropic, Ollama)
- `core/rag/` - RAG pipeline orchestration
- `chat/` - Platform-specific interfaces (Terminal, Telegram, WhatsApp)

## 📋 Requirements

- Python 3.9+
- OpenAI API key (for GPT models) or other LLM provider
- ~500MB for sentence-transformer models (downloaded automatically)
