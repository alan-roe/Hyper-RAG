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
from aiolimiter import AsyncLimiter

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


# Module-level rate limiters for Gemini models using aiolimiter
# Using 80% of actual limits to be conservative
_gemini_limiters = {
    'text-embedding-004': AsyncLimiter(40, 60),  # 40 requests per 60 seconds (80% of 50)
    'gemini-embedding-001': AsyncLimiter(40, 60),  # 40 requests per 60 seconds (80% of 50)
    'gemini-2.0-flash-exp': AsyncLimiter(8, 60),  # 8 requests per 60 seconds (80% of 10)
    'gemini-2.5-flash': AsyncLimiter(8, 60),  # 8 requests per 60 seconds (80% of 10)
}

def get_gemini_limiter(model: str) -> AsyncLimiter:
    """Get the rate limiter for the given model"""
    return _gemini_limiters.get(model, AsyncLimiter(40, 60))  # Default to 40 rpm


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
    limiter = get_gemini_limiter(model)
    
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
        
        # Use aiolimiter to control rate - automatically waits if needed
        async with limiter:
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
                        print(f"\n[429 ERROR] All API keys exhausted!")
                        print(f"  - Batch size: {len(truncated_batch)} texts")
                        
                        # Show sample of actual text content (first text, truncated if needed)
                        if truncated_batch:
                            sample_text = truncated_batch[0]
                            if len(sample_text) > 200:
                                sample_text = sample_text[:200] + "..."
                            print(f"  - First text in batch (truncated): \"{sample_text}\"")
                        
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
                        # Exponential backoff for overload: start at 30s, double each time, cap at 1800s (30 min)
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


@wrap_embedding_func_with_attrs(embedding_dim=768, max_token_size=8192)
async def local_embedding(
    texts: list[str],
    base_url: str = "http://localhost:1234",
    model: str = "nomic-ai/nomic-embed-text-v2-moe-GGUF",
    api_key: str = None,
    batch_size: int = 32,
    **kwargs
) -> np.ndarray:
    """
    Generate embeddings using a local embedding server (e.g., LM Studio, llama.cpp server)
    
    Args:
        texts: List of texts to embed
        base_url: URL of the local embedding server (e.g., http://localhost:1234)
        model: Model name (used for logging, actual model is loaded in the server)
        api_key: Optional API key if the server requires authentication
        batch_size: Number of texts to process in each request
    
    Returns:
        numpy array of embeddings
    """
    import time
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Ensure base_url doesn't have trailing slash
    base_url = base_url.rstrip('/')
    
    # Log embedding progress
    num_texts = len(texts)
    num_batches = (num_texts + batch_size - 1) // batch_size
    
    print(f"[LocalEmbedding] Using server at {base_url}")
    print(f"[LocalEmbedding] Processing {num_texts} texts in {num_batches} batches (batch_size={batch_size})...")
    
    all_embeddings = []
    start_time = time.time()
    
    # Process in batches
    for i in range(0, num_texts, batch_size):
        batch = texts[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        if num_batches > 1:
            print(f"[LocalEmbedding] Processing batch {batch_num}/{num_batches}...")
        
        try:
            # Prepare headers
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            # Try OpenAI-compatible endpoint first
            async with aiohttp.ClientSession() as session:
                # Prepare the request body (OpenAI-compatible format)
                payload = {
                    "input": batch,
                    "model": model,
                    "encoding_format": "float"
                }
                
                # Make the request
                async with session.post(
                    f"{base_url}/v1/embeddings",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        # Extract embeddings from OpenAI-format response
                        for item in result.get("data", []):
                            all_embeddings.append(item["embedding"])
                    else:
                        error_text = await response.text()
                        raise Exception(f"Embedding server error ({response.status}): {error_text}")
                        
        except Exception as e:
            # If OpenAI format fails, try alternative format
            try:
                async with aiohttp.ClientSession() as session:
                    # Alternative format (some local servers use this)
                    payload = {
                        "texts": batch,
                        "model": model
                    }
                    
                    async with session.post(
                        f"{base_url}/embeddings",
                        json=payload,
                        headers=headers
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            embeddings = result.get("embeddings", [])
                            all_embeddings.extend(embeddings)
                        else:
                            raise Exception(f"Both embedding endpoints failed. Original error: {str(e)}")
            except:
                logger.error(f"Failed to get embeddings from {base_url}: {str(e)}")
                raise Exception(f"Could not connect to embedding server at {base_url}. Make sure the server is running and the URL is correct. Error: {str(e)}")
    
    elapsed_time = time.time() - start_time
    texts_per_second = num_texts / elapsed_time if elapsed_time > 0 else 0
    
    print(f"[LocalEmbedding] Completed! Processed {num_texts} texts in {elapsed_time:.2f}s ({texts_per_second:.1f} texts/s)")
    
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    print(f"[LocalEmbedding] Embedding shape: {embeddings_array.shape}")
    
    return embeddings_array


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
