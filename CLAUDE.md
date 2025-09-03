# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Backend Development
```bash
# Install dependencies
uv pip install -r requirements.txt

# Run tests
pytest

# Run async tests  
pytest -v --asyncio-mode=auto

# Test specific components
python web-ui/backend/test_hyperrag_api.py  # Test HyperRAG API
python web-ui/backend/test_file_api.py       # Test file operations
python web-ui/backend/test_websocket_logs.py # Test WebSocket logging

# Start FastAPI backend development server
cd web-ui/backend
fastapi dev main.py
```

### Frontend Development
```bash
# Navigate to frontend
cd web-ui/frontend

# Install dependencies  
pnpm install

# Start development server
pnpm run dev

# Build for production
pnpm run build

# Lint JavaScript/TypeScript code
pnpm run lint:script

# Lint styles
pnpm run lint:style
```

### Quick Demo
```bash
# Run the toy example
python examples/hyperrag_demo.py

# Or run the complete pipeline
python reproduce/Step_0.py          # Preprocess data
python reproduce/Step_1.py          # Build knowledge hypergraphs
python reproduce/Step_2_extract_question.py  # Extract questions
python reproduce/Step_3_response_question.py # Generate responses
```

## Architecture

### Core System (`/hyperrag`)
The main Hyper-RAG implementation with three key retrieval modes:
- **hyper**: Full hypergraph-based retrieval using beyond-pairwise correlations
- **hyper-lite**: Lightweight variant with 2x retrieval speed  
- **naive**: Standard RAG baseline for comparison

Key modules:
- `hyperrag.py`: Main HyperRAG class with configuration and orchestration
- `llm.py`: LLM interfaces with rate limiting:
  - RetryableGeminiClient: Automatic API key rotation for Gemini
  - AsyncLimiter integration for rate control (40-80% of API limits)
  - Support for OpenAI, Azure OpenAI, and Gemini models
- `storage.py`: Storage backends (JsonKVStorage, NanoVectorDBStorage, HypergraphStorage)
- `operate.py`: Core operations (chunking, entity extraction, query methods)
- `base.py`: Abstract base classes and interfaces
- `utils.py`: Utilities for embeddings, hashing, async operations

### Web Interface (`/web-ui`)
Full-stack application with React frontend and FastAPI backend:

**Backend** (`/web-ui/backend`):
- `main.py`: FastAPI application with key endpoints:
  - `/db/*`: Hypergraph CRUD operations
  - `/hyperrag/*`: Query and document insertion
  - `/files/*`: File upload and embedding with WebSocket progress
  - `/settings`: Configuration management
- `file_manager.py`: Document upload and embedding pipeline
- `db.py`: Database operations and hypergraph queries
- `hyperdb/`: Hypergraph database implementation
- `translations.py`: i18n support for multiple languages

**Frontend** (`/web-ui/frontend`):
- React 18 with Ant Design Pro components
- AntV G6 for hypergraph visualization
- WebSocket integration for real-time progress tracking
- Multi-language support (i18n)

### Configuration
1. Copy `config_temp.py` to `my_config.py` and configure:
   - LLM settings (OpenAI, Gemini, or compatible APIs)
   - Embedding model configuration
   - API keys and endpoints (supports multiple keys for load balancing)

2. The system uses:
   - Hypergraph-DB for native hypergraph storage
   - Nano-vectordb for efficient vector search
   - Configurable chunking (default: 1200 tokens with 100 overlap)
   - Settings persistence in `web-ui/backend/settings.json`

### Key Design Patterns
- **Async-first**: Extensive use of asyncio for concurrent operations
- **Storage abstraction**: Pluggable storage backends via base classes
- **Rate limiting**: 
  - Built-in retry with exponential backoff (tenacity)
  - Per-model rate limiters using aiolimiter
  - Automatic API key rotation on rate limit errors
- **Modular queries**: Separate implementations for different RAG strategies
- **Real-time monitoring**: WebSocket-based progress tracking for long operations
- **Multi-database support**: Dynamic database switching with persistent state