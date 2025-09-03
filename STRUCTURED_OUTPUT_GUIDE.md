# HyperRAG Structured Output Guide

## Overview

HyperRAG now supports OpenAI's Structured Outputs feature, which ensures that LLM responses always conform to a predefined JSON schema. This eliminates JSON parsing errors and provides type-safe, validated responses.

## What is Structured Output?

Structured Outputs is an OpenAI feature that:
- **Guarantees** valid JSON that matches your schema
- **Eliminates** the need for complex parsing and retry logic
- **Provides** type safety through Pydantic models
- **Ensures** consistent field names and types

## Key Components

### 1. Pydantic Models (`hyperrag/structured_outputs.py`)

```python
from hyperrag.structured_outputs import (
    KeywordExtractionResponse,      # For query keyword extraction
    EntityExtractionResponse,       # For entity/relationship extraction
    Entity,                         # Individual entity model
    LowOrderHyperedge,             # Pairwise relationships
    HighOrderHyperedge,            # Multi-entity relationships
)
```

### 2. Structured LLM Function (`hyperrag/llm.py`)

```python
from hyperrag.llm import openai_complete_with_structured_output

# Use structured output with Pydantic model
result = await openai_complete_with_structured_output(
    model="gpt-4o-2024-08-06",  # Must be a model that supports structured outputs
    prompt="Your prompt here",
    response_model=KeywordExtractionResponse,  # Pydantic model class
)
```

### 3. Enhanced Query Functions (`hyperrag/operate_structured.py`)

```python
from hyperrag.operate_structured import (
    hyper_query_with_structured_output,
    hyper_query_lite_with_structured_output,
    configure_structured_output,
)
```

## How to Use

### Basic Setup

```python
import asyncio
from hyperrag.llm import openai_complete_with_structured_output
from hyperrag.structured_outputs import KeywordExtractionResponse

async def extract_keywords(query: str):
    prompt = f"Extract high-level and low-level keywords from: {query}"
    
    result = await openai_complete_with_structured_output(
        model="gpt-4o-2024-08-06",
        prompt=prompt,
        response_model=KeywordExtractionResponse,
    )
    
    # result is a validated Pydantic model instance
    print(f"High-level: {result.high_level_keywords}")
    print(f"Low-level: {result.low_level_keywords}")
    
    return result
```

### Configuring HyperRAG to Use Structured Outputs

```python
from hyperrag.operate_structured import configure_structured_output
from hyperrag.llm import openai_complete_with_structured_output

# Configure global settings
configure_structured_output(
    global_config, 
    use_structured=True,
    model_name="gpt-4o-2024-08-06"
)

# Set the structured model function
global_config["structured_model_func"] = openai_complete_with_structured_output

# Now use the enhanced query functions
from hyperrag.operate_structured import hyper_query_with_structured_output

result = await hyper_query_with_structured_output(
    query="Your question here",
    knowledge_hypergraph_inst=kg,
    entities_vdb=entity_vdb,
    relationships_vdb=relation_vdb,
    text_chunks_db=chunks,
    query_param=param,
    global_config=global_config,
)
```

## Supported Models

Structured Outputs require specific OpenAI models:
- `gpt-4o-mini` (all versions after 2024-07-18)
- `gpt-4o-2024-08-06` and later
- `gpt-4o` (latest versions)

## Benefits Over Traditional JSON Parsing

### Traditional Approach (Error-Prone)
```python
# Old way - multiple failure points
result = await llm_func(prompt)
try:
    data = json.loads(result)  # Can fail
    keywords = data.get("keywords", [])  # May not exist
except json.JSONDecodeError:
    # Complex retry logic needed
    result = result.strip()
    # More parsing attempts...
```

### Structured Output (Guaranteed)
```python
# New way - always works
result = await openai_complete_with_structured_output(
    model="gpt-4o-2024-08-06",
    prompt=prompt,
    response_model=KeywordExtractionResponse,
)
# result.high_level_keywords is guaranteed to exist and be a list
```

## Backwards Compatibility

The structured output system is fully backwards compatible:

1. **Fallback Mechanism**: If structured output fails or isn't configured, the system automatically falls back to traditional JSON parsing
2. **Legacy Format Conversion**: Use `convert_to_legacy_format()` to convert structured responses to the original string format
3. **Opt-in Feature**: Structured outputs are disabled by default - enable them explicitly in your configuration

## Testing

Run the test script to see structured outputs in action:

```bash
python test_structured_output.py
```

## Example: Complete Integration

```python
import asyncio
from hyperrag import HyperRAG
from hyperrag.llm import openai_complete_with_structured_output
from hyperrag.operate_structured import configure_structured_output

async def main():
    # Initialize HyperRAG
    rag = HyperRAG()
    
    # Configure for structured outputs
    configure_structured_output(
        rag.global_config,
        use_structured=True,
        model_name="gpt-4o-2024-08-06"
    )
    
    # Set the structured model function
    rag.global_config["structured_model_func"] = openai_complete_with_structured_output
    
    # Now queries will use structured outputs automatically
    response = await rag.query("What are the benefits of renewable energy?")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

## Troubleshooting

### Common Issues

1. **Model doesn't support structured outputs**
   - Solution: Use a compatible model (gpt-4o-2024-08-06 or later)

2. **Pydantic validation errors**
   - Solution: Check that your prompt clearly instructs the model to provide all required fields

3. **API rate limits**
   - Solution: The retry decorators handle rate limits automatically

### Debug Mode

Enable logging to see when structured outputs are used:

```python
import logging
logging.basicConfig(level=logging.INFO)

# You'll see messages like:
# INFO: Successfully used structured output for keyword extraction
# WARNING: Structured output failed, falling back to JSON parsing
```

## Performance Considerations

- **First Request Latency**: The first request with a new schema has additional latency as OpenAI processes the schema
- **Subsequent Requests**: No additional latency for the same schema
- **Token Usage**: Similar to regular completions, no significant overhead

## Next Steps

1. **Entity Extraction**: Extend the EntityExtractionResponse model for your specific entity types
2. **Custom Models**: Create domain-specific Pydantic models for your use case
3. **Validation**: Add custom validators to your Pydantic models for business logic
4. **Monitoring**: Track structured output success rates in production

## References

- [OpenAI Structured Outputs Documentation](https://platform.openai.com/docs/guides/structured-outputs)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [HyperRAG Documentation](https://github.com/your-repo/hyperrag)