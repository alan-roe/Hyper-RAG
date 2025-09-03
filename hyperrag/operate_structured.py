"""
Enhanced query operations with structured output support for HyperRAG.
This module provides improved versions of the query functions that use structured outputs.
"""

import json
from typing import Optional
from .base import BaseKVStorage, BaseVectorStorage, BaseHypergraphStorage, TextChunkSchema, QueryParam
from .prompt import PROMPTS
from .structured_outputs import KeywordExtractionResponse
from .utils import logger


async def hyper_query_with_structured_output(
    query,
    knowledge_hypergraph_inst: BaseHypergraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    """
    Enhanced hyper_query that uses structured outputs when available.
    Falls back to JSON parsing if structured output is not configured.
    """
    entity_context = None
    relation_context = None
    use_model_func = global_config["llm_model_func"]
    
    # Check if structured output is available
    use_structured_output = global_config.get("use_structured_output", False)
    structured_model_func = global_config.get("structured_model_func", None)
    
    entity_keywords = None
    relation_keywords = None
    
    # Try structured output first if available
    if use_structured_output and structured_model_func:
        try:
            kw_prompt_temp = PROMPTS["keywords_extraction"]
            kw_prompt = kw_prompt_temp.format(query=query)
            
            # Use structured output with the configured model
            result = await structured_model_func(
                model=global_config.get("structured_model_name", "gpt-4o-2024-08-06"),
                prompt=kw_prompt,
                response_model=KeywordExtractionResponse,
                base_url=global_config.get("base_url", None),
                api_key=global_config.get("api_key", None),
            )
            
            entity_keywords = ", ".join(result.low_level_keywords)
            relation_keywords = ", ".join(result.high_level_keywords)
            
            logger.info(f"Successfully used structured output for keyword extraction")
            
        except Exception as e:
            logger.warning(f"Structured output failed, falling back to JSON parsing: {e}")
            use_structured_output = False
    
    # Fall back to traditional JSON parsing if structured output is not available or failed
    if not use_structured_output or entity_keywords is None:
        kw_prompt_temp = PROMPTS["keywords_extraction"]
        kw_prompt = kw_prompt_temp.format(query=query)
        
        result = await use_model_func(kw_prompt)
        
        try:
            keywords_data = json.loads(result)
            entity_keywords = keywords_data.get("low_level_keywords", [])
            relation_keywords = keywords_data.get("high_level_keywords", [])
            entity_keywords = ", ".join(entity_keywords)
            relation_keywords = ", ".join(relation_keywords)
        except json.JSONDecodeError:
            try:
                result = (
                    result.replace(kw_prompt[:-1], "")
                    .replace("user", "")
                    .replace("model", "")
                    .strip()
                )
                result = "{" + result.split("{")[1].split("}")[0] + "}"
                keywords_data = json.loads(result)
                relation_keywords = keywords_data.get("high_level_keywords", [])
                entity_keywords = keywords_data.get("low_level_keywords", [])
                relation_keywords = ", ".join(relation_keywords)
                entity_keywords = ", ".join(entity_keywords)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                return PROMPTS["fail_response"]
    
    # Import the original helper functions from operate.py
    from .operate import (
        _build_entity_query_context,
        _build_relation_query_context
    )
    
    # Perform different actions based on keywords
    if entity_keywords:
        entity_context = await _build_entity_query_context(
            entity_keywords,
            knowledge_hypergraph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )
    
    if relation_keywords:
        relation_context = await _build_relation_query_context(
            relation_keywords,
            knowledge_hypergraph_inst,
            entities_vdb,
            relationships_vdb,
            text_chunks_db,
            query_param,
        )
    
    # Combine contexts and generate response
    from .utils import process_combine_contexts
    
    context_str = process_combine_contexts(
        entity_context, relation_context, query_param
    )
    
    if query_param.only_need_context:
        return context_str
    
    # Generate response
    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        response_type=query_param.response_type,
        context_data=context_str,
    )
    
    ll_context, hl_context = entity_keywords, relation_keywords
    if ll_context and hl_context:
        content = query + PROMPTS["rag_define"].format(
            ll_keywords=ll_context,
            hl_keywords=hl_context,
        )
    else:
        content = query
    
    response = await use_model_func(
        content,
        system_prompt=sys_prompt,
    )
    
    # Clean up the response
    if len(response) > len(sys_prompt):
        response = (
            response[len(sys_prompt):]
            .replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(content, "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )
    
    if query_param.return_type == "json":
        response = {
            "response": response,
            "entity_context": entity_context,
            "relation_context": relation_context,
        }
    
    return response


async def hyper_query_lite_with_structured_output(
    query,
    knowledge_hypergraph_inst: BaseHypergraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    """
    Enhanced hyper_query_lite that uses structured outputs when available.
    Falls back to JSON parsing if structured output is not configured.
    """
    entity_context = None
    use_model_func = global_config["llm_model_func"]
    
    # Check if structured output is available
    use_structured_output = global_config.get("use_structured_output", False)
    structured_model_func = global_config.get("structured_model_func", None)
    
    entity_keywords = None
    
    # Try structured output first if available
    if use_structured_output and structured_model_func:
        try:
            kw_prompt_temp = PROMPTS["keywords_extraction"]
            kw_prompt = kw_prompt_temp.format(query=query)
            
            # Use structured output with the configured model
            result = await structured_model_func(
                model=global_config.get("structured_model_name", "gpt-4o-2024-08-06"),
                prompt=kw_prompt,
                response_model=KeywordExtractionResponse,
                base_url=global_config.get("base_url", None),
                api_key=global_config.get("api_key", None),
            )
            
            entity_keywords = ", ".join(result.low_level_keywords)
            
            logger.info(f"Successfully used structured output for keyword extraction (lite)")
            
        except Exception as e:
            logger.warning(f"Structured output failed, falling back to JSON parsing: {e}")
            use_structured_output = False
    
    # Fall back to traditional JSON parsing if structured output is not available or failed
    if not use_structured_output or entity_keywords is None:
        kw_prompt_temp = PROMPTS["keywords_extraction"]
        kw_prompt = kw_prompt_temp.format(query=query)
        
        result = await use_model_func(kw_prompt)
        
        try:
            keywords_data = json.loads(result)
            entity_keywords = keywords_data.get("low_level_keywords", [])
            entity_keywords = ", ".join(entity_keywords)
        except json.JSONDecodeError:
            try:
                result = (
                    result.replace(kw_prompt[:-1], "")
                    .replace("user", "")
                    .replace("model", "")
                    .strip()
                )
                result = "{" + result.split("{")[1].split("}")[0] + "}"
                keywords_data = json.loads(result)
                entity_keywords = keywords_data.get("low_level_keywords", [])
                entity_keywords = ", ".join(entity_keywords)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                return PROMPTS["fail_response"]
    
    # Import the original helper functions from operate.py
    from .operate import _build_entity_query_context
    
    # Perform actions based on keywords
    if entity_keywords:
        entity_context = await _build_entity_query_context(
            entity_keywords,
            knowledge_hypergraph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )
    
    # Generate response
    if entity_context is None:
        return PROMPTS["fail_response"]
    
    if query_param.only_need_context:
        return entity_context
    
    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        response_type=query_param.response_type,
        context_data=entity_context,
    )
    
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    
    # Clean up the response
    if len(response) > len(sys_prompt):
        response = (
            response[len(sys_prompt):]
            .replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )
    
    if query_param.return_type == "json":
        response = {
            "response": response,
            "context": entity_context,
        }
    
    return response


def configure_structured_output(global_config: dict, use_structured: bool = True, model_name: str = "gpt-4o-2024-08-06"):
    """
    Configure the global_config to use structured outputs.
    
    Args:
        global_config: The global configuration dictionary
        use_structured: Whether to enable structured outputs
        model_name: The OpenAI model name that supports structured outputs
    
    Example:
        from hyperrag.operate_structured import configure_structured_output
        from hyperrag.llm import openai_complete_with_structured_output
        
        # Configure to use structured outputs
        configure_structured_output(
            global_config, 
            use_structured=True,
            model_name="gpt-4o-2024-08-06"
        )
        
        # Set the structured model function
        global_config["structured_model_func"] = openai_complete_with_structured_output
    """
    global_config["use_structured_output"] = use_structured
    global_config["structured_model_name"] = model_name
    
    if use_structured:
        logger.info(f"Configured to use structured outputs with model: {model_name}")
    else:
        logger.info("Structured outputs disabled, using traditional JSON parsing")