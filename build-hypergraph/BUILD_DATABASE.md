# HyperRAG Database Builder Guide

This script builds a HyperRAG knowledge database from your local files using Google's Gemini API for embeddings and entity extraction.

## Setup

### 1. Install Dependencies

```bash
# Install the required Google Gemini package
uv add google-genai

# Or install all dependencies from requirements file
uv pip install -r requirements_builder.txt
```

### 2. Get Gemini API Key

Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

Set it as environment variable:
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Or for Vertex AI:
```bash
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
```

## Usage

### Basic Usage

Build database from a directory:
```bash
uv run python build_hyperrag_db.py /path/to/your/project
```

Build from a single file:
```bash
uv run python build_hyperrag_db.py /path/to/document.txt
```

### Dry Run (Preview Files)

See which files will be processed without building:
```bash
uv run python build_hyperrag_db.py /path/to/project --dry-run
```

### Advanced Options

```bash
uv run python build_hyperrag_db.py /path/to/project \
  --output ./my_knowledge_base \
  --api-key YOUR_API_KEY \
  --generation-model gemini-2.5-flash-lite \
  --embedding-model gemini-embedding-001 \
  --embedding-dim 768 \
  --temperature 0.1 \
  --max-file-size 20 \
  --batch-size 5
```

### Specifying Additional Ignore Patterns

```bash
# Ignore specific patterns via command line
uv run python build_hyperrag_db.py /path/to/project \
  --ignore "*.test.js" \
  --ignore "temp_*" \
  --ignore "build/"

# Use a custom ignore file
uv run python build_hyperrag_db.py /path/to/project \
  --ignore-file ./my-ignore-patterns.txt

# Combine multiple methods
uv run python build_hyperrag_db.py /path/to/project \
  --ignore "*.tmp" \
  --ignore-file ./extra-ignores.txt
```

### Using .hyperragignore Files

Create `.hyperragignore` files in any directory (just like `.gitignore`):

```bash
# .hyperragignore in project root
*.test.js
*.spec.ts
test/
__tests__/
*.mock.*
coverage/
.pytest_cache/
```

The script automatically:
1. Finds all `.hyperragignore` files in the directory tree
2. Applies patterns relative to where the file is located
3. Combines them with `.gitignore` patterns
4. Works in subdirectories too

### Using Vertex AI

```bash
uv run python build_hyperrag_db.py /path/to/project \
  --use-vertex \
  --project-id your-gcp-project \
  --location us-central1
```

## Features

### Automatic File Filtering

The script automatically:
- **Respects .gitignore**: Reads all .gitignore files in the directory tree
- **Respects .hyperragignore**: Automatically reads all .hyperragignore files in the directory tree
- **Filters binary files**: Skips images, videos, executables, etc.
- **Size limits**: Skips files larger than specified size (default 10MB)
- **Text detection**: Intelligently detects text files by extension and content

### Supported File Types

Automatically processes:
- **Code**: `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.go`, `.rs`, etc.
- **Markup**: `.md`, `.rst`, `.html`, `.xml`, `.tex`
- **Config**: `.json`, `.yaml`, `.toml`, `.ini`, `.env`
- **Data**: `.csv`, `.tsv`, `.sql`
- **Docs**: `.txt`, `.log`
- And many more text formats

### Default Ignore Patterns

Always ignores:
- Version control: `.git`, `.svn`, `.hg`
- Dependencies: `node_modules`, `venv`, `__pycache__`
- Build artifacts: `*.pyc`, `*.so`, `*.exe`
- Media files: images, videos, audio
- Archives: `*.zip`, `*.tar.gz`
- System files: `.DS_Store`, `Thumbs.db`

## Examples

### Build knowledge base from a Python project
```bash
uv run python build_hyperrag_db.py ~/my_python_project --dry-run
# Review the files that will be processed
uv run python build_hyperrag_db.py ~/my_python_project -o ./python_project_kb
```

### Build from documentation folder
```bash
uv run python build_hyperrag_db.py ./docs -o ./docs_knowledge_base \
  --max-file-size 5 \
  --batch-size 8
```

### Process a codebase with custom model
```bash
uv run python build_hyperrag_db.py ./src \
  --generation-model gemini-2.5-flash-lite \
  --embedding-model gemini-embedding-001 \
  --embedding-dim 768 \
  --temperature 0.1
```

## Using the Built Database

Once built, use your database in code:

```python
from hyperrag import HyperRAG, QueryParam
from hyperrag.utils import EmbeddingFunc
import numpy as np
from google import genai

# Initialize Gemini client
client = genai.Client(api_key="your-key")

# Define functions for HyperRAG
async def llm_func(prompt, **kwargs):
    response = client.models.generate_content(
        model='gemini-2.5-flash-lite',
        contents=prompt
    )
    return response.text

async def embed_func(texts):
    response = client.models.embed_content(
        model='gemini-embedding-001',
        contents=texts
    )
    embeddings = [e.values for e in response.embeddings]
    return np.array(embeddings)

# Load your database
rag = HyperRAG(
    working_dir="./your_database_dir",
    llm_model_func=llm_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        func=embed_func
    )
)

# Query your knowledge base
result = rag.query(
    "How does the authentication system work?",
    param=QueryParam(mode="hyper")
)
print(result)
```

## Tips

1. **Start with dry-run**: Always do a dry-run first to see what files will be processed
2. **Batch size**: Lower batch size (5-10) for large files, higher (20-50) for small files
3. **Rate limits**: Gemini has rate limits; the script handles retries automatically
4. **Incremental builds**: HyperRAG deduplicates content, so you can safely re-run on updated files
5. **Model selection**:
   - Use `gemini-2.5-flash-lite` for faster, cheaper processing
   - Use `gemini-2.5-flash` for better quality extraction
   - Embedding dimension should match your model (768 recommended for gemini-embedding-001)

## Troubleshooting

### "API key not found"
Set your API key: `export GOOGLE_API_KEY="your-key"`

### "Rate limit exceeded"
Reduce `--batch-size` or add delays between batches

### "File too large"
Increase `--max-file-size` or split large files

### "Out of memory"
Process in smaller batches or reduce `--batch-size`

### Dry run not showing expected files
Check your .gitignore patterns and verify file extensions are in the supported list