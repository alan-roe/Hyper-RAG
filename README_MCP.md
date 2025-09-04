# HyperRAG MCP Server

This is a Model Context Protocol (MCP) server that exposes HyperRAG's context retrieval functionality to LLMs via FastMCP.

## Overview

The HyperRAG MCP Server provides tools for querying hypergraph-based RAG databases to retrieve enriched context. Unlike typical RAG systems, HyperRAG captures beyond-pairwise relationships using hypergraphs, enabling more comprehensive context retrieval.

The server returns enriched context including:
- Extracted entities with descriptions
- Multi-hop relationships between entities
- Relevant text chunks
- Hypergraph-based correlations

It does **not** include the final LLM response generation, allowing the calling LLM to use the context as needed.

## Installation

### Prerequisites

1. Install FastMCP:
```bash
pip install fastmcp
```

2. Ensure HyperRAG is installed with all dependencies:
```bash
pip install -r requirements.txt
```

### Configuration

The server uses settings from `web-ui/backend/settings.json` for:
- LLM model configuration
- Embedding model settings
- API keys and endpoints

## Available Tools

### 1. `list_databases`

Lists all available HyperRAG databases.

**Parameters:** None

**Returns:** List of database names

**Example:**
```python
databases = await list_databases()
# Returns: ["documents_db", "research_db", "notes_db"]
```

### 2. `query_context`

Queries a HyperRAG database for enriched context.

**Parameters:**
- `query` (str, required): The question or query to search for context
- `database` (str, required): The name of the database to query

**Returns:** Dictionary containing enriched context with entities, relationships, and text units

**Example:**
```python
context = await query_context(
    query="What is the architecture of HyperRAG?",
    database="documents_db"
)
```

## Usage

### Running the Server

#### Standalone Mode (stdio transport):
```bash
python mcp_server.py
```

#### Using FastMCP CLI:
```bash
fastmcp run mcp_server.py:mcp
```

#### HTTP Transport:
```bash
fastmcp run mcp_server.py:mcp --transport http --port 8000
```

### Integration with Claude Desktop

1. Edit your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "hyperrag": {
      "command": "python",
      "args": ["/path/to/Hyper-RAG/mcp_server.py"]
    }
  }
}
```

2. Restart Claude Desktop

3. The HyperRAG tools will be available in Claude's tool list

### Integration with Other MCP Clients

For HTTP transport:
```python
from fastmcp.client import FastMCPClient

client = FastMCPClient(url="http://localhost:8000")

# List databases
databases = await client.call_tool("list_databases", {})

# Query for context
context = await client.call_tool("query_context", {
    "query": "Explain hypergraph-based retrieval",
    "database": "research_db"
})
```

## How It Works

1. **Database Discovery**: Use `list_databases` to find available HyperRAG databases in the `hyperrag_cache` directory

2. **Context Retrieval**: Query a specific database to get enriched context. The server:
   - Initializes or retrieves a cached HyperRAG instance for the database
   - Processes the query through the hypergraph retrieval pipeline
   - Returns structured context without generating a final response

3. **Caching**: HyperRAG instances are cached per database to avoid re-initialization overhead

## Database Structure

HyperRAG databases are stored in the `hyperrag_cache/` directory. Each database is a folder containing:
- Hypergraph storage files
- Vector embeddings
- Entity and relationship indices
- Text chunk storage

## Troubleshooting

### No databases found
- Ensure databases exist in `hyperrag_cache/` directory
- Check directory permissions

### Query errors
- Verify the database name exists (use `list_databases` first)
- Check `settings.json` for proper LLM/embedding configuration
- Review logs for detailed error messages

### Performance
- First query to a database may be slower due to initialization
- Subsequent queries use cached instances for better performance

## Development

To modify or extend the server:

1. Edit `mcp_server.py` to add new tools or modify behavior
2. Update `fastmcp.json` if dependencies change
3. Test with: `fastmcp run mcp_server.py:mcp`

## License

Same as the main HyperRAG project.