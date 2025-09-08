"""
Wrapper functions that automatically choose between structured and traditional output
based on configuration settings.
"""

import json
import os
from typing import Optional, Type, Union, Dict, Any
from pydantic import BaseModel
from .llm import openai_complete_if_cache, openai_complete_with_structured_output
from .structured_outputs import KeywordExtractionResponse, EntityExtractionResponse, convert_to_legacy_format
from .utils import logger
from .logging_utils import log_llm_request, log_llm_response


class StructuredLLMWrapper:
    """
    Wrapper that automatically handles structured vs traditional output based on settings.
    """
    
    def __init__(self, settings: Dict[str, Any]):
        """
        Initialize with settings from settings.json
        
        Args:
            settings: Dictionary containing model configuration
        """
        self.settings = settings
        self.use_structured = settings.get("useStructuredOutput", False)
        self.model_name = settings.get("modelName", "gpt-4o")
        self.model_provider = settings.get("modelProvider", "openai")
        self.base_url = settings.get("baseUrl")
        self.api_key = settings.get("apiKey")
        
        # Handle API key as list or string
        if isinstance(self.api_key, list):
            self.api_key = self.api_key[0] if self.api_key else None
        
        # Disable structured output for non-OpenAI providers
        if self.use_structured and (self.model_provider.lower() != "openai" or 
                                   (self.base_url and "openai" not in self.base_url.lower())):
            logger.warning(f"Structured output disabled | provider={self.model_provider} | base_url={self.base_url}")
            self.use_structured = False
        
        logger.debug(f"LLM wrapper initialized | structured={self.use_structured} | model={self.model_name} | provider={self.model_provider}")
    
    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: list = [],
        response_model: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> Union[str, BaseModel]:
        """
        Complete a prompt with automatic structured/traditional output handling.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            history_messages: Conversation history
            response_model: Optional Pydantic model for structured output
            **kwargs: Additional arguments for the API
        
        Returns:
            Either a string (traditional) or Pydantic model instance (structured)
        """
        # Log request in clean format
        log_llm_request(
            logger,
            model=self.model_name,
            prompt=prompt,
            structured=self.use_structured,
            response_model=response_model.__name__ if response_model else None
        )
        
        # Merge settings kwargs with call kwargs (call kwargs take precedence)
        final_kwargs = {
            "base_url": self.base_url,
            "api_key": self.api_key,
            **kwargs
        }
        
        # Log kwargs (excluding sensitive data)
        safe_kwargs = {k: v for k, v in final_kwargs.items() if k not in ['api_key']}
        logger.debug(f"Request kwargs: {safe_kwargs}")
        
        # If structured output is enabled and a response model is provided
        if self.use_structured and response_model is not None:
            try:
                logger.debug(f"Attempting structured output | model={response_model.__name__} | provider={self.model_provider}")
                # Filter out hashing_kv parameter which is not supported by structured output
                structured_kwargs = {k: v for k, v in final_kwargs.items() if k != 'hashing_kv'}
                logger.debug(f"Filtered kwargs for structured output: {list(structured_kwargs.keys())}")
                
                result = await openai_complete_with_structured_output(
                    model=self.model_name,
                    prompt=prompt,
                    response_model=response_model,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    **structured_kwargs
                )
                logger.debug(f"Structured output successful | type={type(result).__name__}")
                return result
            except Exception as e:
                error_msg = str(e)
                if "404" in error_msg or "Not Found" in error_msg:
                    logger.error(f"Structured output not supported | provider={self.model_provider} | base_url={self.base_url} | disable in settings")
                logger.debug(f"Structured output failed, using traditional | error={str(e)[:100]}")
                # Fall through to traditional method
        
        # Traditional completion
        logger.debug("Using traditional JSON output")
        result = await openai_complete_if_cache(
            model=self.model_name,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **final_kwargs
        )
        
        # Log response in clean format
        log_llm_response(logger, result)
        
        # If a response model was provided, try to parse the JSON
        if response_model is not None:
            try:
                # Try to extract JSON from the response
                json_str = result
                
                # Handle common LLM response patterns
                if "```json" in result:
                    json_str = result.split("```json")[1].split("```")[0]
                    logger.debug("Extracted JSON from markdown code block")
                elif "```" in result:
                    json_str = result.split("```")[1].split("```")[0]
                    logger.debug("Extracted content from code block")
                
                # Parse JSON and create model instance
                data = json.loads(json_str)
                logger.debug(f"JSON parsed | model={response_model.__name__}")
                return response_model(**data)
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"❌ Failed to parse response into {response_model.__name__}: {e}")
                logger.debug(f"Raw response: {result[:500]}")
                # Return raw string if parsing fails
                return result
        
        return result
    
    async def extract_keywords(self, query: str) -> KeywordExtractionResponse:
        """
        Extract keywords from a query with automatic structured/traditional handling.
        
        Args:
            query: The query to extract keywords from
        
        Returns:
            KeywordExtractionResponse object
        """
        from .prompt import PROMPTS
        
        kw_prompt_temp = PROMPTS["keywords_extraction"]
        kw_prompt = kw_prompt_temp.format(query=query)
        
        result = await self.complete(
            prompt=kw_prompt,
            response_model=KeywordExtractionResponse
        )
        
        # If we got a string back (traditional parsing failed), try manual parsing
        if isinstance(result, str):
            try:
                # Try to parse the string as JSON
                if "{" in result and "}" in result:
                    json_str = "{" + result.split("{")[1].split("}")[0] + "}"
                    data = json.loads(json_str)
                    return KeywordExtractionResponse(**data)
            except:
                # Fallback: return empty keywords
                logger.error("Failed to parse keywords, returning empty response")
                return KeywordExtractionResponse(
                    high_level_keywords=[],
                    low_level_keywords=[]
                )
        
        return result


def create_smart_llm_func(settings: Dict[str, Any]):
    """
    Create a drop-in replacement for llm_model_func that handles structured output.
    
    This function creates a callable that can be used as llm_model_func in global_config,
    but with automatic structured output support based on settings.
    
    Args:
        settings: Settings dictionary from settings.json
    
    Returns:
        Async function that can be used as llm_model_func
    """
    wrapper = StructuredLLMWrapper(settings)
    
    async def smart_llm_func(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: list = [],
        **kwargs
    ) -> str:
        """
        Smart LLM function that handles structured output when appropriate.
        
        This function checks if the prompt is requesting JSON output and uses
        structured output if available and configured.
        """
        # Only log at debug level with prompt length
        logger.debug(f"Smart LLM called | prompt_len={len(prompt)}")
        
        # Check if the prompt is asking for JSON (common patterns)
        json_indicators = [
            "json",
            "JSON",
            "high_level_keywords",
            "low_level_keywords",
            "entity_extraction",
            "Format each entity as",
            # Keywords extraction specific patterns
            "Output the keywords in JSON format",
            "identifying both high-level and low-level keywords",
            # Double brace pattern used in prompts
            "{{"
        ]
        
        # Detect if structured output would be beneficial
        needs_json = any(indicator in prompt for indicator in json_indicators)
        
        # Log which indicators were checked
        indicators_found = [ind for ind in json_indicators if ind in prompt]
        if needs_json:
            logger.debug(f"JSON output detected | indicators: {indicators_found}")
        
        if needs_json and wrapper.use_structured:
            # Determine the appropriate response model
            response_model = None
            
            if "keywords_extraction" in prompt or \
               ("high-level and low-level keywords" in prompt) or \
               ("Output the keywords in JSON format" in prompt) or \
               ('"high_level_keywords"' in prompt and '"low_level_keywords"' in prompt):
                response_model = KeywordExtractionResponse
                logger.debug("Detected keyword extraction request")
            elif ("Entity" in prompt and "Low-order Hyperedge" in prompt and "High-order Hyperedge" in prompt) or \
                 ("entity_extraction" in prompt and "entity_name" in prompt):
                response_model = EntityExtractionResponse
                logger.debug("Detected entity extraction request")
            
            if response_model:
                try:
                    logger.debug(f"Using smart structured output | model={response_model.__name__}")
                    result = await wrapper.complete(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        history_messages=history_messages,
                        response_model=response_model,
                        **kwargs
                    )
                    
                    # Convert structured result back to appropriate format for compatibility
                    if isinstance(result, EntityExtractionResponse):
                        # Convert entity extraction to legacy tuple format for backward compatibility
                        legacy_format = convert_to_legacy_format(result)
                        logger.debug("Converted EntityExtractionResponse to legacy format")
                        return legacy_format
                    elif isinstance(result, BaseModel):
                        json_result = result.model_dump_json()
                        logger.debug(f"Converted to JSON | type={type(result).__name__}")
                        return json_result
                    return result
                except Exception as e:
                    logger.warning(f"Smart structured output failed: {e}")
        
        # Fall back to traditional completion
        logger.debug("Using traditional completion")
        return await wrapper.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs
        )
    
    # Store the wrapper for direct access if needed
    smart_llm_func.wrapper = wrapper
    
    return smart_llm_func