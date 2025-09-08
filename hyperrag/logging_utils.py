"""
Logging utilities for HyperRAG with clean, structured output.
"""

import logging
from typing import Optional, Any, Dict
import functools
import time
import asyncio


class CleanLogger:
    """Logger that provides clean, non-redundant logging."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._context = {}
    
    def set_context(self, **kwargs):
        """Set context that will be included in all subsequent log messages."""
        self._context.update(kwargs)
    
    def clear_context(self):
        """Clear the logging context."""
        self._context.clear()
    
    def _format_extras(self, extras: Dict[str, Any]) -> str:
        """Format extra data for logging."""
        if not extras:
            return ""
        items = [f"{k}={v}" for k, v in extras.items() if v is not None]
        return " | " + " | ".join(items) if items else ""
    
    def info(self, message: str, **kwargs):
        """Log info message with optional structured data."""
        extras = {**self._context, **kwargs}
        self.logger.info(f"{message}{self._format_extras(extras)}")
    
    def error(self, message: str, **kwargs):
        """Log error message with optional structured data."""
        extras = {**self._context, **kwargs}
        self.logger.error(f"{message}{self._format_extras(extras)}")
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional structured data."""
        extras = {**self._context, **kwargs}
        self.logger.warning(f"{message}{self._format_extras(extras)}")
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional structured data."""
        extras = {**self._context, **kwargs}
        self.logger.debug(f"{message}{self._format_extras(extras)}")


def log_llm_request(logger: logging.Logger, model: str, prompt: str, 
                   structured: bool = False, response_model: Optional[str] = None):
    """Log LLM request in a clean, concise format."""
    prompt_preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
    # Remove newlines for cleaner logs
    prompt_preview = prompt_preview.replace('\n', ' ').replace('\r', ' ')
    
    extras = []
    extras.append(f"model={model}")
    if structured:
        extras.append("structured=true")
    if response_model:
        extras.append(f"response_model={response_model}")
    extras.append(f"prompt_len={len(prompt)}")
    
    logger.info(f"LLM Request | {' | '.join(extras)}")
    logger.debug(f"Prompt preview: {prompt_preview}")


def log_llm_response(logger: logging.Logger, response: str, 
                    duration: Optional[float] = None, tokens: Optional[int] = None):
    """Log LLM response in a clean format."""
    response_preview = response[:200] + "..." if len(response) > 200 else response
    response_preview = response_preview.replace('\n', ' ').replace('\r', ' ')
    
    extras = []
    extras.append(f"response_len={len(response)}")
    if duration:
        extras.append(f"duration={duration:.2f}s")
    if tokens:
        extras.append(f"tokens={tokens}")
    
    logger.info(f"LLM Response | {' | '.join(extras)}")
    logger.debug(f"Response preview: {response_preview}")


def log_entity_extraction(logger: logging.Logger, chunk_id: str, 
                         entities: int, relationships: int):
    """Log entity extraction results concisely."""
    logger.info(f"Entity extraction | chunk={chunk_id} | entities={entities} | relationships={relationships}")


def timed_operation(operation_name: str):
    """Decorator to log operation timing."""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            start_time = time.time()
            logger.info(f"Starting {operation_name}")
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"Completed {operation_name} | duration={duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Failed {operation_name} | duration={duration:.2f}s | error={str(e)}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            start_time = time.time()
            logger.info(f"Starting {operation_name}")
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"Completed {operation_name} | duration={duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Failed {operation_name} | duration={duration:.2f}s | error={str(e)}")
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator