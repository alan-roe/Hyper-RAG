"""
FastMCP Server for HyperRAG Context Retrieval

This MCP server exposes HyperRAG's context retrieval functionality,
allowing LLMs to query HyperRAG databases for enriched context.
"""

import sys
import os
import json
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

from fastmcp import FastMCP, Context
from fastmcp.utilities.logging import get_logger

# Import HyperRAG components
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hyperrag import HyperRAG
from hyperrag.base import QueryParam
from hyperrag.llm import openai_complete_if_cache, openai_embedding
from hyperrag.llm_structured_wrapper import create_smart_llm_func
from hyperrag.utils import encode_string_by_tiktoken


mcp = FastMCP("HyperRAG Context Server")

# Create module logger for server-side logging
logger = get_logger(__name__)

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
async def query_context(query: str, database: str, max_tokens, ctx: Context = None) -> Dict:
    """
    Query a HyperRAG database for enriched context
    
    This returns the full context including entities, relationships, 
    and relevant text chunks, but without the final LLM response generation.
    
    Args:
        query: The question or query to search for context
        database: The name of the database to query
        max_tokens: Maximum tokens for context (controls retrieval size)
        ctx: FastMCP context for client logging
    
    Returns:
        Dictionary containing the enriched context with entities, 
        relationships, and text units
    """
    try:
        # Log incoming request
        if ctx:
            await ctx.info(f"Query request received for database '{database}'", extra={
                "query": query[:100] if len(query) > 100 else query,
                "database": database,
                "max_tokens": max_tokens,
            })
        
        # Get or create HyperRAG instance
        if ctx:
            await ctx.debug(f"Getting HyperRAG instance for database '{database}'")
        rag = get_or_create_hyperrag(database)
        
        # Dynamically adjust parameters based on max_tokens to prevent huge responses
        # Convert max_tokens to int if it's a string
        if max_tokens is not None:
            try:
                max_tokens = int(max_tokens)
            except (ValueError, TypeError):
                max_tokens = 4000  # Default if conversion fails
                if ctx:
                    await ctx.warning(f"Invalid max_tokens value, using default: 4000")
        else:
            max_tokens = 4000  # Set a reasonable default
        
        # Scale parameters based on max_tokens
        # These ratios help balance the context distribution
        if max_tokens <= 1000:
            # Very small context - minimal retrieval
            default_top_k = 10
            max_token_for_text = 300
            max_token_for_entity = 100
            max_token_for_relation = 300
            default_mode = "hyper" 
        elif max_tokens <= 2000:
            # Small context - reduced retrieval
            default_top_k = 20
            max_token_for_text = 500
            max_token_for_entity = 150
            max_token_for_relation = 500
            default_mode = "hyper"
        elif max_tokens <= 4000:
            # Medium context - balanced retrieval
            default_top_k = 30
            max_token_for_text = 800
            max_token_for_entity = 200
            max_token_for_relation = 800
            default_mode = "hyper"
        else:
            # Large context - fuller retrieval (but still bounded)
            default_top_k = 40
            max_token_for_text = 1200
            max_token_for_entity = 250
            max_token_for_relation = 1200
            default_mode = "hyper"
        
        # Use provided parameters or defaults
        final_mode = default_mode
        final_top_k = default_top_k
        
        # Create query parameters with scaled values
        param = QueryParam(
            mode=final_mode,
            top_k=final_top_k,
            max_token_for_text_unit=max_token_for_text,
            max_token_for_entity_context=max_token_for_entity,
            max_token_for_relation_context=max_token_for_relation
        )
        
        # Log the parameters being used for debugging
        sys.stderr.write(f"Query with max_tokens={max_tokens}, using mode={final_mode}, top_k={final_top_k}\n")
        logger.info(f"Query parameters: max_tokens={max_tokens}, mode={final_mode}, top_k={final_top_k}")
        
        if ctx:
            await ctx.debug(f"Starting context extraction with mode='{final_mode}', top_k={final_top_k}")
        
        # Use the new aget_context method to get context without response generation
        # Pass max_tokens if specified
        result = await rag.aget_context(query, param, max_tokens)
        
        # Count tokens in the response
        total_tokens = 0
        if result and isinstance(result, dict):
            # Convert result to JSON string to count tokens
            result_json = json.dumps(result, ensure_ascii=False)
            total_tokens = len(encode_string_by_tiktoken(result_json))
            
            # Log token count to stderr for debugging
            sys.stderr.write(f"Response contains {total_tokens} tokens (requested max: {max_tokens})\n")
            logger.info(f"Response token count: {total_tokens} (requested max: {max_tokens})")
        
        # Log the results
        if ctx:
            if result and isinstance(result, dict):
                entities_count = len(result.get("entities", []))
                hyperedges_count = len(result.get("hyperedges", []))
                text_units_count = len(result.get("text_units", []))
                
                if entities_count == 0 and hyperedges_count == 0 and text_units_count == 0:
                    await ctx.warning("Query returned empty context - no matching entities, hyperedges, or text units found")
                    await ctx.debug("This may indicate a keyword extraction failure or no relevant content in the database")
                else:
                    await ctx.info(f"Context retrieved successfully", extra={
                        "entities": entities_count,
                        "hyperedges": hyperedges_count,
                        "text_units": text_units_count,
                        "total_tokens": total_tokens,
                        "requested_max_tokens": max_tokens
                    })
            else:
                await ctx.warning(f"Unexpected result format: {type(result)}")
        
        # Result should already be in the correct format from aget_context
        return result
        
    except Exception as e:
        error_msg = f"Failed to query database '{database}': {str(e)}"
        sys.stderr.write(f"ERROR: {error_msg}\n")
        sys.stderr.write(f"Query: {query}\n")
        traceback.print_exc(file=sys.stderr)
        logger.error(f"Query failed: {error_msg}", exc_info=True)
        
        if ctx:
            await ctx.error(f"Query failed: {str(e)}", extra={
                "database": database,
                "query": query[:100] if len(query) > 100 else query,
                "error_type": type(e).__name__
            })
        
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
