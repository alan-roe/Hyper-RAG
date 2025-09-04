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
        self.base_url = settings.get("baseUrl")
        self.api_key = settings.get("apiKey")
        
        # Handle API key as list or string
        if isinstance(self.api_key, list):
            self.api_key = self.api_key[0] if self.api_key else None
        
        logger.info(f"StructuredLLMWrapper initialized: structured={self.use_structured}, model={self.model_name}")
    
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
        # Merge settings kwargs with call kwargs (call kwargs take precedence)
        final_kwargs = {
            "base_url": self.base_url,
            "api_key": self.api_key,
            **kwargs
        }
        
        # If structured output is enabled and a response model is provided
        if self.use_structured and response_model is not None:
            try:
                logger.debug(f"Using structured output with model: {response_model.__name__}")
                result = await openai_complete_with_structured_output(
                    model=self.model_name,
                    prompt=prompt,
                    response_model=response_model,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    **final_kwargs
                )
                return result
            except Exception as e:
                logger.warning(f"Structured output failed, falling back to traditional: {e}")
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
        
        # If a response model was provided, try to parse the JSON
        if response_model is not None:
            try:
                # Try to extract JSON from the response
                json_str = result
                
                # Handle common LLM response patterns
                if "```json" in result:
                    json_str = result.split("```json")[1].split("```")[0]
                elif "```" in result:
                    json_str = result.split("```")[1].split("```")[0]
                
                # Parse JSON and create model instance
                data = json.loads(json_str)
                return response_model(**data)
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse response into {response_model.__name__}: {e}")
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
        # Check if the prompt is asking for JSON (common patterns)
        json_indicators = [
            "json",
            "JSON",
            "high_level_keywords",
            "low_level_keywords",
            "entity_extraction",
            "Format each entity as"
        ]
        
        # Detect if structured output would be beneficial
        needs_json = any(indicator in prompt for indicator in json_indicators)
        
        if needs_json and wrapper.use_structured:
            # Determine the appropriate response model
            response_model = None
            
            if "keywords_extraction" in prompt or ("high_level_keywords" in prompt and "low_level_keywords" in prompt):
                response_model = KeywordExtractionResponse
                logger.debug("Detected keyword extraction request")
            elif ("Entity" in prompt and "Low-order Hyperedge" in prompt and "High-order Hyperedge" in prompt) or \
                 ("entity_extraction" in prompt and "entity_name" in prompt):
                response_model = EntityExtractionResponse
                logger.debug("Detected entity extraction request")
            
            if response_model:
                try:
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
                        return convert_to_legacy_format(result)
                    elif isinstance(result, BaseModel):
                        return result.model_dump_json()
                    return result
                except Exception as e:
                    logger.warning(f"Structured output failed: {e}")
        
        # Fall back to traditional completion
        return await wrapper.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs
        )
    
    # Store the wrapper for direct access if needed
    smart_llm_func.wrapper = wrapper
    
    return smart_llm_func