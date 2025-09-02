import os
import copy
from functools import lru_cache
import json
import asyncio
import aioboto3
import aiohttp
import httpx
import numpy as np
import time
from collections import deque
import threading

from openai import (
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
    Timeout,
    AsyncAzureOpenAI,
)

import base64
import struct

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from pydantic import BaseModel, Field
from typing import List, Dict, Callable, Any
from .base import BaseKVStorage
from .utils import compute_args_hash, wrap_embedding_func_with_attrs

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class GeminiRateLimiter:
    """Thread-safe rate limiter for Gemini API with configurable limits"""
    
    LIMITS = {
        'text-embedding-004': {'rpm': 50, 'tpm': 30000},
        'gemini-embedding-001': {'rpm': 50, 'tpm': 30000},
        'gemini-2.0-flash-exp': {'rpm': 10, 'tpm': 250000},
        'gemini-2.5-flash': {'rpm': 10, 'tpm': 250000}
    }
    
    def __init__(self, model: str = 'gemini-embedding-001'):
        limits = self.LIMITS.get(model, {'rpm': 100, 'tpm': 30000})
        # Use more conservative limits to avoid hitting rate limits
        self.rpm_limit = int(limits['rpm'] * 0.8)  # Use 80% of limit
        self.tpm_limit = int(limits['tpm'] * 0.8)  # Use 80% of limit
        self.request_times = deque()
        self.token_counts = deque()
        self._lock = asyncio.Lock()
    
    def get_usage_stats(self):
        """Get current usage statistics for debugging"""
        now = time.time()
        # Clean old entries
        valid_times = [t for t in self.request_times if t > now - 60]
        valid_tokens = [self.token_counts[i] for i, t in enumerate(self.request_times) if t > now - 60]
        
        return {
            'requests_last_minute': len(valid_times),
            'tokens_last_minute': sum(valid_tokens),
            'rpm_limit': self.rpm_limit,
            'tpm_limit': self.tpm_limit,
            'rpm_usage_percent': (len(valid_times) / self.rpm_limit * 100) if self.rpm_limit > 0 else 0,
            'tpm_usage_percent': (sum(valid_tokens) / self.tpm_limit * 100) if self.tpm_limit > 0 else 0
        }
    
    async def wait_if_needed(self, estimated_tokens: int):
        """Wait if rate limits would be exceeded - thread-safe"""
        async with self._lock:
            now = time.time()
            
            # Clean old entries (older than 60 seconds)
            while self.request_times and self.request_times[0] < now - 60:
                self.request_times.popleft()
                self.token_counts.popleft()
            
            current_requests = len(self.request_times)
            current_tokens = sum(self.token_counts)
            
            # Debug logging - show current usage
            print(f"[Rate Limiter Debug] Current usage: {current_requests}/{self.rpm_limit} RPM, {current_tokens}/{self.tpm_limit} TPM, New request: {estimated_tokens} tokens")
            
            # Check if we need to wait for RPM limit
            if current_requests >= self.rpm_limit:
                wait_time = 60 - (now - self.request_times[0])
                if wait_time > 0:
                    print(f"[Rate Limiter] RPM limit reached ({current_requests}/{self.rpm_limit}), waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time + 0.1)
                    now = time.time()  # Update now after waiting
            
            # Check if we need to wait for TPM limit
            total_tokens = current_tokens + estimated_tokens
            if total_tokens > self.tpm_limit:
                wait_time = 60 - (now - self.request_times[0])
                if wait_time > 0:
                    print(f"[Rate Limiter] TPM limit would be exceeded ({total_tokens}/{self.tpm_limit}), waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time + 0.1)
                    now = time.time()  # Update now after waiting
            
            # Record this request
            self.request_times.append(now)
            self.token_counts.append(estimated_tokens)


# Module-level rate limiter instances (one per model to avoid conflicts)
_gemini_rate_limiters = {}
_rate_limiter_lock = threading.Lock()

def get_gemini_rate_limiter(model: str):
    """Get or create the singleton rate limiter for the given model - thread-safe"""
    global _gemini_rate_limiters
    
    with _rate_limiter_lock:
        if model not in _gemini_rate_limiters:
            _gemini_rate_limiters[model] = GeminiRateLimiter(model)
        return _gemini_rate_limiters[model]


class RetryableGeminiClient:
    """Wrapper for Gemini client that automatically rotates API keys on rate limit errors"""
    
    def __init__(self, api_keys):
        """
        Initialize with one or more API keys
        
        Args:
            api_keys: Single API key string or list of API key strings
        """
        # Ensure api_keys is a list
        if isinstance(api_keys, str):
            self.api_keys = [api_keys]
        elif isinstance(api_keys, list):
            self.api_keys = api_keys
        else:
            self.api_keys = [str(api_keys)]  # Convert to string if needed
            
        # Filter out None/empty values
        self.api_keys = [k for k in self.api_keys if k]
        
        if not self.api_keys:
            raise ValueError("At least one valid API key is required")
            
        self.current_key_index = 0
        self._create_client()
    
    def _create_client(self):
        """Create a new genai client with the current API key"""
        from google import genai
        self.client = genai.Client(api_key=self.api_keys[self.current_key_index])
        
    def _rotate_key(self):
        """Rotate to the next API key and create a new client"""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self._create_client()
        
    def embed_content(self, **kwargs):
        """
        Call embed_content with automatic API key rotation on rate limit
        
        Returns the result from the first successful API key, or raises
        the exception if all keys are exhausted
        """
        last_exception = None
        keys_tried = 0
        
        while keys_tried < len(self.api_keys):
            try:
                # Try with current API key
                return self.client.models.embed_content(**kwargs)
                
            except Exception as e:
                error_str = str(e)
                
                # Check if this is a rate limit error
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "rate limit" in error_str.lower():
                    last_exception = e
                    keys_tried += 1
                    
                    # If we have more keys to try, rotate
                    if keys_tried < len(self.api_keys):
                        print(f"[RetryableGeminiClient] Rate limit hit on API key {self.current_key_index + 1}/{len(self.api_keys)}, rotating to next key...")
                        self._rotate_key()
                        continue
                    else:
                        # All keys exhausted
                        print(f"[RetryableGeminiClient] All {len(self.api_keys)} API keys hit rate limits")
                        raise last_exception
                else:
                    # Not a rate limit error, raise immediately
                    raise
        
        # This should not be reached, but just in case
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Unexpected state in RetryableGeminiClient")


async def gemini_embedding(
    texts: list[str],
    model: str = "gemini-embedding-001",
    api_key = None,  # Can be string or list of strings
    embedding_dim: int = 1536,
    **kwargs
) -> np.ndarray:
    """
    Generate embeddings using Google Gemini API with proper retry configuration
    
    Args:
        texts: List of texts to embed
        model: Gemini embedding model name
        api_key: Gemini API key (string) or list of API keys for rotation
        embedding_dim: Output embedding dimension
    
    Returns:
        numpy array of embeddings
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError("Please install google-genai: pip install google-genai")
    
    # Handle both single key and multiple keys
    if api_key:
        # api_key provided directly (could be string or list)
        api_keys = api_key
    else:
        # Fall back to environment variable
        env_key = os.environ.get("GOOGLE_API_KEY")
        if env_key:
            api_keys = env_key
        else:
            raise ValueError("No API key provided and GOOGLE_API_KEY environment variable not set")
    
    # Create the retryable client with automatic key rotation
    client = RetryableGeminiClient(api_keys)
    
    # Get rate limiter for this model
    rate_limiter = get_gemini_rate_limiter(model)
    
    # Process in batches with rate limiting
    max_batch_size = 2  # Conservative batch size to avoid rate limits
    all_embeddings = []
    
    for i in range(0, len(texts), max_batch_size):
        batch = texts[i:i + max_batch_size]
        
        # Truncate texts that are too long (gemini-embedding-001 has 2048 token limit)
        truncated_batch = []
        for text in batch:
            # More conservative truncation at ~1500 tokens (assuming ~1.3 tokens per word)
            max_words = int(1500 / 1.3)
            words = text.split()[:max_words]
            truncated_batch.append(' '.join(words))
        
        # Estimate tokens for rate limiting (1.3 tokens per word average)
        estimated_tokens = sum(len(text.split()) * 1.3 for text in truncated_batch)
        
        # Wait if rate limits would be exceeded
        await rate_limiter.wait_if_needed(int(estimated_tokens))
        
        # Generate embeddings for batch with infinite retry for rate limits and overload
        attempt = 0
        while True:
            try:
                response = client.embed_content(
                    model=model,
                    contents=truncated_batch,
                    config=types.EmbedContentConfig(
                        output_dimensionality=embedding_dim,
                        task_type='RETRIEVAL_DOCUMENT'
                    )
                )
                
                # Extract embeddings
                for embedding in response.embeddings:
                    all_embeddings.append(embedding.values)
                break  # Success - exit retry loop
                    
            except Exception as e:
                error_str = str(e)
                
                # Handle rate limit errors (429) - Only reached if ALL API keys are exhausted
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "rate limit" in error_str.lower():
                    # Log current rate limiter state for debugging
                    stats = rate_limiter.get_usage_stats()
                    print(f"\n[429 ERROR] All API keys exhausted! Full request details:")
                    print(f"  - Batch size: {len(truncated_batch)} texts")
                    print(f"  - Estimated tokens for this batch: ~{estimated_tokens:.1f} tokens")
                    
                    # Show sample of actual text content (first text, truncated if needed)
                    if truncated_batch:
                        sample_text = truncated_batch[0]
                        if len(sample_text) > 200:
                            sample_text = sample_text[:200] + "..."
                        print(f"  - First text in batch (truncated): \"{sample_text}\"")
                        
                        # Show length statistics for all texts in batch
                        text_lengths = [len(text) for text in truncated_batch]
                        print(f"  - Text lengths in batch: {text_lengths}")
                        print(f"  - Total characters: {sum(text_lengths)}")
                        print(f"  - Average chars per text: {sum(text_lengths)/len(text_lengths):.1f}")
                    
                    print(f"\n  Current rate limiter state:")
                    print(f"  - Requests in last minute: {stats['requests_last_minute']}/{stats['rpm_limit']} ({stats['rpm_usage_percent']:.1f}%)")
                    print(f"  - Tokens in last minute: {stats['tokens_last_minute']}/{stats['tpm_limit']} ({stats['tpm_usage_percent']:.1f}%)")
                    
                    # Determine which limit was likely hit
                    if stats['rpm_usage_percent'] > stats['tpm_usage_percent']:
                        print(f"  → Likely hit RPM limit (requests per minute)")
                    else:
                        print(f"  → Likely hit TPM limit (tokens per minute)")
                    
                    # Show the actual error message from API
                    print(f"\n  API Error: {error_str[:500]}")  # Truncate very long errors
                    
                    # Wait 60 seconds for rate limit window to reset, with small additional buffer
                    wait_time = 60 + (attempt * 5)  # 60s, 65s, 70s, etc.
                    print(f"\n[429 ERROR] All keys exhausted, waiting {wait_time}s for rate limit reset (attempt {attempt + 1})")
                    await asyncio.sleep(wait_time)
                    attempt += 1
                    continue
                    
                # Handle 503 Service Unavailable / Model Overloaded errors
                elif "503" in error_str or "UNAVAILABLE" in error_str or "overloaded" in error_str.lower():
                    # Exponential backoff for overload: start at 30s (5*6), double each time, cap at 1800s (30 min)
                    wait_time = min(30 * (2 ** attempt), 1800)
                    print(f"Model overloaded (503), waiting {wait_time}s before retry (attempt {attempt + 1})")
                    await asyncio.sleep(wait_time)
                    attempt += 1
                    continue
                    
                else:
                    # For other errors, fail immediately
                    print(f"Error generating embeddings for batch: {e}")
                    raise
    
    return np.array(all_embeddings, dtype=np.float32)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def openai_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    base_url=None,
    api_key=None,
    **kwargs,
) -> str:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    openai_async_client = (
        AsyncOpenAI() if base_url is None else AsyncOpenAI(base_url=base_url)
    )
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
    return response.choices[0].message.content


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def azure_openai_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    base_url=None,
    api_key=None,
    **kwargs,
):
    if api_key:
        os.environ["AZURE_OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["AZURE_OPENAI_ENDPOINT"] = base_url

    openai_async_client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    if prompt is not None:
        messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
    return response.choices[0].message.content


class BedrockError(Exception):
    """Generic error for issues related to Amazon Bedrock"""


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, max=60),
    retry=retry_if_exception_type((BedrockError)),
)
async def bedrock_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    aws_access_key_id=None,
    aws_secret_access_key=None,
    aws_session_token=None,
    **kwargs,
) -> str:
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get(
        "AWS_ACCESS_KEY_ID", aws_access_key_id
    )
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get(
        "AWS_SECRET_ACCESS_KEY", aws_secret_access_key
    )
    os.environ["AWS_SESSION_TOKEN"] = os.environ.get(
        "AWS_SESSION_TOKEN", aws_session_token
    )

    # Fix message history format
    messages = []
    for history_message in history_messages:
        message = copy.copy(history_message)
        message["content"] = [{"text": message["content"]}]
        messages.append(message)

    # Add user prompt
    messages.append({"role": "user", "content": [{"text": prompt}]})

    # Initialize Converse API arguments
    args = {"modelId": model, "messages": messages}

    # Define system prompt
    if system_prompt:
        args["system"] = [{"text": system_prompt}]

    # Map and set up inference parameters
    inference_params_map = {
        "max_tokens": "maxTokens",
        "top_p": "topP",
        "stop_sequences": "stopSequences",
    }
    if inference_params := list(
        set(kwargs) & set(["max_tokens", "temperature", "top_p", "stop_sequences"])
    ):
        args["inferenceConfig"] = {}
        for param in inference_params:
            args["inferenceConfig"][inference_params_map.get(param, param)] = (
                kwargs.pop(param)
            )

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    # Call model via Converse API
    session = aioboto3.Session()
    async with session.client("bedrock-runtime") as bedrock_async_client:
        try:
            response = await bedrock_async_client.converse(**args, **kwargs)
        except Exception as e:
            raise BedrockError(e)

        if hashing_kv is not None:
            await hashing_kv.upsert(
                {
                    args_hash: {
                        "return": response["output"]["message"]["content"][0]["text"],
                        "model": model,
                    }
                }
            )

        return response["output"]["message"]["content"][0]["text"]


async def gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:

    return await openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,

        **kwargs,
    )


async def gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def azure_openai_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await azure_openai_complete_if_cache(
        "conversation-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def bedrock_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await bedrock_complete_if_cache(
        "anthropic.claude-3-haiku-20240307-v1:0",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def openai_embedding(
    texts: list[str],
    model: str = "text-embedding-3-small",
    base_url: str = None,
    api_key: str = None,
) -> np.ndarray:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    openai_async_client = (
        AsyncOpenAI() if base_url is None else AsyncOpenAI(base_url=base_url)
    )
    response = await openai_async_client.embeddings.create(
        model=model, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def azure_openai_embedding(
    texts: list[str],
    model: str = "text-embedding-3-small",
    base_url: str = None,
    api_key: str = None,
) -> np.ndarray:
    if api_key:
        os.environ["AZURE_OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["AZURE_OPENAI_ENDPOINT"] = base_url

    openai_async_client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )

    response = await openai_async_client.embeddings.create(
        model=model, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def siliconcloud_embedding(
    texts: list[str],
    model: str = "netease-youdao/bce-embedding-base_v1",
    base_url: str = "https://api.siliconflow.cn/v1/embeddings",
    max_token_size: int = 512,
    api_key: str = None,
) -> np.ndarray:
    if api_key and not api_key.startswith("Bearer "):
        api_key = "Bearer " + api_key

    headers = {"Authorization": api_key, "Content-Type": "application/json"}

    truncate_texts = [text[0:max_token_size] for text in texts]

    payload = {"model": model, "input": truncate_texts, "encoding_format": "base64"}

    base64_strings = []
    async with aiohttp.ClientSession() as session:
        async with session.post(base_url, headers=headers, json=payload) as response:
            content = await response.json()
            if "code" in content:
                raise ValueError(content)
            base64_strings = [item["embedding"] for item in content["data"]]

    embeddings = []
    for string in base64_strings:
        decode_bytes = base64.b64decode(string)
        n = len(decode_bytes) // 4
        float_array = struct.unpack("<" + "f" * n, decode_bytes)
        embeddings.append(float_array)
    return np.array(embeddings)


# @wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
# @retry(
#     stop=stop_after_attempt(3),
#     wait=wait_exponential(multiplier=1, min=4, max=10),
#     retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),  # TODO: fix exceptions
# )
async def bedrock_embedding(
    texts: list[str],
    model: str = "amazon.titan-embed-text-v2:0",
    aws_access_key_id=None,
    aws_secret_access_key=None,
    aws_session_token=None,
) -> np.ndarray:
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get(
        "AWS_ACCESS_KEY_ID", aws_access_key_id
    )
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get(
        "AWS_SECRET_ACCESS_KEY", aws_secret_access_key
    )
    os.environ["AWS_SESSION_TOKEN"] = os.environ.get(
        "AWS_SESSION_TOKEN", aws_session_token
    )

    session = aioboto3.Session()
    async with session.client("bedrock-runtime") as bedrock_async_client:
        if (model_provider := model.split(".")[0]) == "amazon":
            embed_texts = []
            for text in texts:
                if "v2" in model:
                    body = json.dumps(
                        {
                            "inputText": text,
                            # 'dimensions': embedding_dim,
                            "embeddingTypes": ["float"],
                        }
                    )
                elif "v1" in model:
                    body = json.dumps({"inputText": text})
                else:
                    raise ValueError(f"Model {model} is not supported!")

                response = await bedrock_async_client.invoke_model(
                    modelId=model,
                    body=body,
                    accept="application/json",
                    contentType="application/json",
                )

                response_body = await response.get("body").json()

                embed_texts.append(response_body["embedding"])
        elif model_provider == "cohere":
            body = json.dumps(
                {"texts": texts, "input_type": "search_document", "truncate": "NONE"}
            )

            response = await bedrock_async_client.invoke_model(
                model=model,
                body=body,
                accept="application/json",
                contentType="application/json",
            )

            response_body = json.loads(response.get("body").read())

            embed_texts = response_body["embeddings"]
        else:
            raise ValueError(f"Model provider '{model_provider}' is not supported!")

        return np.array(embed_texts)


class Model(BaseModel):
    """
    This is a Pydantic model class named 'Model' that is used to define a custom language model.

    Attributes:
        gen_func (Callable[[Any], str]): A callable function that generates the response from the language model.
            The function should take any argument and return a string.
        kwargs (Dict[str, Any]): A dictionary that contains the arguments to pass to the callable function.
            This could include parameters such as the model name, API key, etc.

    Example usage:
        Model(gen_func=openai_complete_if_cache, kwargs={"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY_1"]})

    In this example, 'openai_complete_if_cache' is the callable function that generates the response from the OpenAI model.
    The 'kwargs' dictionary contains the model name and API key to be passed to the function.
    """

    gen_func: Callable[[Any], str] = Field(
        ...,
        description="A function that generates the response from the llm. The response must be a string",
    )
    kwargs: Dict[str, Any] = Field(
        ...,
        description="The arguments to pass to the callable function. Eg. the api key, model name, etc",
    )

    class Config:
        arbitrary_types_allowed = True


class MultiModel:
    """
    Distributes the load across multiple language models. Useful for circumventing low rate limits with certain api providers especially if you are on the free tier.
    Could also be used for spliting across diffrent models or providers.

    Attributes:
        models (List[Model]): A list of language models to be used.

    Usage example:
        ```python
        models = [
            Model(gen_func=openai_complete_if_cache, kwargs={"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY_1"]}),
            Model(gen_func=openai_complete_if_cache, kwargs={"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY_2"]}),
            Model(gen_func=openai_complete_if_cache, kwargs={"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY_3"]}),
            Model(gen_func=openai_complete_if_cache, kwargs={"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY_4"]}),
            Model(gen_func=openai_complete_if_cache, kwargs={"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY_5"]}),
        ]
        multi_model = MultiModel(models)
        rag = LightRAG(
            llm_model_func=multi_model.llm_model_func
            / ..other args
            )
        ```
    """

    def __init__(self, models: List[Model]):
        self._models = models
        self._current_model = 0

    def _next_model(self):
        self._current_model = (self._current_model + 1) % len(self._models)
        return self._models[self._current_model]

    async def llm_model_func(
        self, prompt, system_prompt=None, history_messages=[], **kwargs
    ) -> str:
        kwargs.pop("model", None)  # stop from overwriting the custom model name
        next_model = self._next_model()
        args = dict(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
            **next_model.kwargs,
        )

        return await next_model.gen_func(**args)


if __name__ == "__main__":
    import asyncio

    async def main():
        result = await gpt_4o_mini_complete("How are you?")
        print(result)

    asyncio.run(main())
