# 🧠 Knowledge Base System

A multi-source knowledge base with AI agent, featuring automatic content extraction and intelligent querying.

## 🚀 Main Workflow

1. **Add Source** → Provide URL or file path
2. **Auto Extract** → Content is automatically extracted and processed
3. **Query** → Ask questions about your knowledge base

## ✨ Features

- **Multi-Source Support**: Web pages, PDFs, JSON, CSV, TXT, DOCX, XLSX
- **Automatic Processing**: Add source → extract → process → store in one step
- **AI-Powered Search**: Query your knowledge base with natural language
- **Storage Options**: MongoDB + ChromaDB (with memory fallback)
- **Clean GUI**: Simple Gradio interface focused on core workflow

## 📋 Requirements

- Python 3.9+
- OpenAI API key (for AI agent)
- Optional: MongoDB and ChromaDB for persistent storage

## 🛠️ Installation

1. **Clone and install dependencies:**

```bash
git clone <repository>
cd A05_task
uv pip install -e .
```

2. **Set up environment:**

```bash
cp knowledge_base/.env.example knowledge_base/.env
# Edit .env and add your OPENAI_API_KEY
```

3. **Start the system:**

```bash
# Terminal 1: Start backend
cd knowledge_base
python main.py

# Terminal 2: Start Gradio GUI
python gradio_app.py
```

4. **Access the GUI:**

- Open http://localhost:7860
- Backend API: http://localhost:8000

## 🎯 Usage

### Add Source & Extract

1. Go to "📥 Add Source" tab
2. Enter Source ID (e.g., `wiki_python`)
3. Enter Location (e.g., `https://en.wikipedia.org/wiki/Python`)
4. Select Source Type (`web`)
5. Click "🚀 Add Source & Extract"
6. Wait for automatic extraction and processing

### Query Knowledge Base

1. Go to "🔍 Query" tab
2. Ask questions like:
   - "What is Python?"
   - "Summarize the key points"
   - "What are the main concepts?"
3. Get AI-powered answers with sources

## 🏗️ Architecture

```
gradio_app.py (GUI) → FastAPI Backend → Extractors → Processors → Storage
                                                                    ↓
                                     AI Agent ← MongoDB + ChromaDB
```

### Components

- **Gradio GUI**: Simple web interface (`gradio_app.py`)
- **FastAPI Backend**: API server (`knowledge_base/main.py`)
- **Extractors**: Content extraction from various sources
- **Processors**: Text chunking and preprocessing
- **Storage**: MongoDB (metadata) + ChromaDB (vectors) + Memory (fallback)
- **AI Agent**: OpenAI-powered query answering

## 🔧 Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your_openai_api_key

# Optional (defaults provided)
MONGODB_URI=mongodb://localhost:27017
CHROMA_PERSIST_DIRECTORY=./chroma_db
HOST=0.0.0.0
PORT=8000
```

### Supported Source Types

- **web**: Web pages and HTML content
- **pdf**: PDF documents
- **json**: JSON files and API responses
- **csv**: CSV files and structured data
- **txt**: Plain text files
- **docx**: Microsoft Word documents
- **xlsx**: Microsoft Excel files

## 🚀 Quick Start Example

1. Start the system (backend + GUI)
2. Add a Wikipedia source:
   - Source ID: `test_source`
   - Location: `https://en.wikipedia.org/wiki/Artificial_intelligence`
   - Type: `web`
3. Wait for processing (see progress in Result area)
4. Query: "What is artificial intelligence?"
5. Get AI-powered answer with sources!

## 🛟 Troubleshooting

- **Backend not responding**: Check if `python knowledge_base/main.py` is running
- **Extraction timeout**: Large sources may take 1-2 minutes to process
- **Database errors**: System falls back to memory storage automatically
- **API key errors**: Ensure OPENAI_API_KEY is set in `.env` file

## 📁 Project Structure

```
A05_task/
├── gradio_app.py              # Gradio GUI (main interface)
├── knowledge_base/
│   ├── main.py                # FastAPI backend server
│   ├── api/routes.py          # API endpoints
│   ├── extractors/            # Content extractors
│   ├── processors/            # Text processors
│   ├── storage/               # Storage backends
│   ├── agent/                 # AI agent
│   └── utils/                 # Utilities
└── pyproject.toml             # Dependencies
```

---

**🎯 Focus**: Simple, effective knowledge base with automatic extraction and AI-powered querying via clean Gradio interface.
