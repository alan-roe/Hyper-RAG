"""
FastMCP Server for HyperRAG Context Retrieval

This MCP server exposes HyperRAG's context retrieval functionality,
allowing LLMs to query HyperRAG databases for enriched context.
"""

import sys
import os
import json
import asyncio
import logging
import traceback
from pathlib import Path
from typing import Dict, List

# Configure logging before any imports to suppress non-critical output
# but allow error messages for debugging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

# Suppress info/debug messages from HyperRAG
logging.getLogger("hyper_rag").setLevel(logging.ERROR)
logging.getLogger("hyperrag").setLevel(logging.ERROR)

from fastmcp import FastMCP

# Import HyperRAG components
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hyperrag import HyperRAG
from hyperrag.base import QueryParam
from hyperrag.llm import openai_complete_if_cache, openai_embedding
from hyperrag.llm_structured_wrapper import create_smart_llm_func


mcp = FastMCP("HyperRAG Context Server")

# Global configuration
SETTINGS_FILE = Path(__file__).parent / "web-ui" / "backend" / "settings.json"
HYPERRAG_CACHE_DIR = Path(__file__).parent / "web-ui" / "backend" / "hyperrag_cache"

# Cache for HyperRAG instances
hyperrag_instances: Dict[str, HyperRAG] = {}


def load_settings():
    """Load settings from settings.json"""
    try:
        if not SETTINGS_FILE.exists():
            return {}
        
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        sys.stderr.write(f"ERROR: Failed to load settings: {e}\n")
        return {}


def get_or_create_hyperrag(database: str) -> HyperRAG:
    """
    Get or create HyperRAG instance for specified database
    """
    global hyperrag_instances
    
    try:
        # Check if instance already exists
        if database in hyperrag_instances:
            return hyperrag_instances[database]
        
        # Load settings
        settings = load_settings()
        
        # Determine working directory
        db_working_dir = HYPERRAG_CACHE_DIR / database
        db_working_dir.mkdir(parents=True, exist_ok=True)
        
        # Get embedding configuration
        embedding_provider = settings.get("embeddingProvider", settings.get("modelProvider", "openai"))
        embedding_model = settings.get("embeddingModel", "text-embedding-3-small")
        
        # Determine embedding dimensions
        embedding_dim = 1536  # Default for OpenAI
        if embedding_provider == "local":
            embedding_dim = 768  # Default for local models
        elif embedding_model == "text-embedding-3-large":
            embedding_dim = 3072
        
        # Create embedding function
        if embedding_provider == "local":
            from hyperrag.llm import local_embedding
            embedding_func = local_embedding
        else:
            embedding_func = openai_embedding
        
        # Create LLM function
        llm_func = None
        if settings.get("useStructuredOutput", False):
            llm_func = create_smart_llm_func(settings)
        else:
            async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
                final_kwargs = {
                    "base_url": settings.get("baseUrl"),
                    "api_key": settings.get("apiKey"),
                    **kwargs
                }
                if isinstance(final_kwargs.get("api_key"), list):
                    final_kwargs["api_key"] = final_kwargs["api_key"][0] if final_kwargs["api_key"] else None
                
                return await openai_complete_if_cache(
                    model=settings.get("modelName", "gpt-4o"),
                    prompt=prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    **final_kwargs
                )
        
        # Create HyperRAG instance
        rag = HyperRAG(
            working_dir=str(db_working_dir),
            embedding_func=embedding_func,
            llm_model_func=llm_func,
            vector_db_storage_cls_kwargs={"embedding_dim": embedding_dim}
        )
        
        # Cache the instance
        hyperrag_instances[database] = rag
        
        return rag
    except Exception as e:
        sys.stderr.write(f"ERROR: Failed to create HyperRAG instance for database '{database}': {e}\n")
        traceback.print_exc(file=sys.stderr)
        raise


@mcp.tool
async def list_databases() -> List[str]:
    """
    List all available HyperRAG databases
    
    Returns:
        List of database names available in the hyperrag_cache directory
    """
    databases = []
    
    # Check if cache directory exists
    if not HYPERRAG_CACHE_DIR.exists():
        return databases
    
    try:
        for item in HYPERRAG_CACHE_DIR.iterdir():
            if item.is_dir():
                databases.append(item.name)
    except OSError as e:
        sys.stderr.write(f"ERROR: Failed to list databases: {e}\n")
        # Return empty list on error rather than failing completely
    
    return databases


@mcp.tool
async def query_context(query: str, database: str) -> Dict:
    """
    Query a HyperRAG database for enriched context
    
    This returns the full context including entities, relationships, 
    and relevant text chunks, but without the final LLM response generation.
    
    Args:
        query: The question or query to search for context
        database: The name of the database to query
    
    Returns:
        Dictionary containing the enriched context with entities, 
        relationships, and text units
    """
    try:
        # Get or create HyperRAG instance
        rag = get_or_create_hyperrag(database)
        
        # Create query parameters
        param = QueryParam(
            mode="hyper",  # Use full hypergraph mode by default
            only_need_context=True,  # Return context without final LLM response
            top_k=60,
            max_token_for_text_unit=1600,
            max_token_for_entity_context=300,
            max_token_for_relation_context=1600,
            return_type='json'
        )
        
        # Execute query
        result = await rag.aquery(query, param)
        
        # Ensure result is in JSON format
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                # If not JSON, wrap in a simple structure
                result = {"context": result}
        
        return result
        
    except Exception as e:
        error_msg = f"Failed to query database '{database}': {str(e)}"
        sys.stderr.write(f"ERROR: {error_msg}\n")
        sys.stderr.write(f"Query: {query}\n")
        traceback.print_exc(file=sys.stderr)
        
        return {
            "error": str(e),
            "database": database,
            "query": query
        }


# Run the server
if __name__ == "__main__":
    # Run with HTTP transport for better debugging and network access
    mcp.run(
        transport="http",
        host="192.168.1.228",
        port=8765,
    )
