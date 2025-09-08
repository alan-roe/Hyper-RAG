"""
Enhanced logging utilities for HyperRAG backend with structured, contextual logging.
"""

import logging
import functools
import time
import uuid
from contextvars import ContextVar
from typing import Optional, Any, Dict, Callable
from datetime import datetime

# Context variable to track request/operation ID
operation_id: ContextVar[Optional[str]] = ContextVar('operation_id', default=None)

class StructuredLogger:
    """Enhanced logger with structured output and context tracking."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._operation_stack = []
    
    def _format_message(self, message: str, extra: Dict[str, Any] = None) -> str:
        """Format message with operation context."""
        op_id = operation_id.get()
        if op_id:
            prefix = f"[{op_id[:8]}] "
        else:
            prefix = ""
        
        if extra:
            # Add structured data in a clean format
            extra_str = " | ".join([f"{k}={v}" for k, v in extra.items()])
            return f"{prefix}{message} | {extra_str}"
        return f"{prefix}{message}"
    
    def info(self, message: str, **kwargs):
        """Log info with optional structured data."""
        self.logger.info(self._format_message(message, kwargs))
    
    def error(self, message: str, **kwargs):
        """Log error with optional structured data."""
        self.logger.error(self._format_message(message, kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning with optional structured data."""
        self.logger.warning(self._format_message(message, kwargs))
    
    def debug(self, message: str, **kwargs):
        """Log debug with optional structured data."""
        self.logger.debug(self._format_message(message, kwargs))
    
    def operation(self, name: str, **metadata):
        """Context manager for logging operations with timing."""
        return LoggedOperation(self, name, **metadata)


class LoggedOperation:
    """Context manager for tracking operations with automatic timing and logging."""
    
    def __init__(self, logger: StructuredLogger, name: str, **metadata):
        self.logger = logger
        self.name = name
        self.metadata = metadata
        self.start_time = None
        self.op_id = str(uuid.uuid4())
    
    def __enter__(self):
        self.start_time = time.time()
        old_id = operation_id.get()
        self.token = operation_id.set(self.op_id)
        
        self.logger.info(f"Starting {self.name}", **self.metadata)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        operation_id.reset(self.token)
        
        if exc_type:
            self.logger.error(
                f"Failed {self.name}", 
                duration=f"{duration:.2f}s",
                error=str(exc_val),
                **self.metadata
            )
        else:
            self.logger.info(
                f"Completed {self.name}",
                duration=f"{duration:.2f}s",
                **self.metadata
            )


def log_async_operation(operation_name: str, **default_metadata):
    """Decorator for logging async operations with timing."""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            logger = StructuredLogger(func.__module__)
            metadata = {**default_metadata, **kwargs.get('_log_metadata', {})}
            
            with logger.operation(operation_name, **metadata):
                return await func(*args, **kwargs)
        return wrapper
    return decorator


def truncate_content(content: str, max_length: int = 100) -> str:
    """Truncate long content for logging."""
    if len(content) <= max_length:
        return content
    return f"{content[:max_length]}... (truncated, {len(content)} chars total)"


class LLMLogger:
    """Specialized logger for LLM operations with content truncation."""
    
    def __init__(self, base_logger: StructuredLogger):
        self.logger = base_logger
    
    def log_request(self, model: str, prompt: str, **kwargs):
        """Log LLM request with truncated prompt."""
        self.logger.info(
            "LLM Request",
            model=model,
            prompt_preview=truncate_content(prompt, 200),
            prompt_length=len(prompt),
            **kwargs
        )
    
    def log_response(self, response: str, tokens_used: Optional[int] = None, **kwargs):
        """Log LLM response with truncated content."""
        self.logger.info(
            "LLM Response", 
            response_preview=truncate_content(response, 200),
            response_length=len(response),
            tokens_used=tokens_used,
            **kwargs
        )
    
    def log_entity_extraction(self, chunk_id: str, entities_count: int, relationships_count: int):
        """Log entity extraction results."""
        self.logger.info(
            "Entity extraction completed",
            chunk_id=chunk_id,
            entities=entities_count,
            relationships=relationships_count
        )


# Pre-configured loggers for different components
def get_logger(component: str) -> StructuredLogger:
    """Get a structured logger for a specific component."""
    return StructuredLogger(component)


def get_llm_logger(component: str) -> LLMLogger:
    """Get an LLM logger for a specific component."""
    return LLMLogger(get_logger(component))


# Configure logging format for the entire application
def configure_app_logging(level: str = "INFO"):
    """Configure application-wide logging settings."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level),
        format=log_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Reduce verbosity of third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)