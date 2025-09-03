# Using Gemini with Local Embedding Server

This guide shows how to configure Hyper-RAG to use Google Gemini for the LLM while running embeddings through a local server (e.g., LM Studio, llama.cpp, Ollama).

## Prerequisites

1. **Local Embedding Server**: You need a local server running that provides OpenAI-compatible embedding endpoints. Popular options include:
   - [LM Studio](https://lmstudio.ai/) - Easy GUI for running GGUF models
   - [llama.cpp server](https://github.com/ggerganov/llama.cpp) - Command-line server
   - [Ollama](https://ollama.ai/) - Simple CLI tool

2. **Embedding Model**: Download a GGUF embedding model. We recommend:
   - `nomic-ai/nomic-embed-text-v2-moe-GGUF` - Excellent 768-dimensional embeddings
   - Available from: https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe-GGUF

## Setting Up the Local Server

### Option 1: LM Studio (Recommended for beginners)

1. Download and install LM Studio
2. Download the `nomic-ai/nomic-embed-text-v2-moe-GGUF` model
3. Load the model in LM Studio
4. Start the server (it will run on `http://localhost:1234` by default)

### Option 2: llama.cpp

```bash
# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Download the model
wget https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe-GGUF/resolve/main/nomic-embed-text-v2-moe.Q4_K_M.gguf

# Start the server
./server -m nomic-embed-text-v2-moe.Q4_K_M.gguf --embeddings --port 1234
```

## Configuration

Update your `settings.json` file (located in `web-ui/backend/settings.json`) with the following configuration:

```json
{
  "modelProvider": "gemini",
  "model": "gemini-2.0-flash-exp",
  "apiKey": "YOUR_GEMINI_API_KEY_HERE",
  
  "embeddingProvider": "local",
  "embeddingModel": "nomic-ai/nomic-embed-text-v2-moe-GGUF",
  "embeddingBaseUrl": "http://localhost:1234",
  "embeddingDim": 768,
  "embeddingBatchSize": 32,
  
  "temperature": 0.7,
  "maxTokens": 2048,
  "streamMode": true,
  "language": "en"
}
```

### Configuration Options Explained

**LLM Settings (Gemini):**
- `modelProvider`: Set to `"gemini"` to use Google's Gemini models
- `model`: The Gemini model to use (e.g., `"gemini-2.0-flash-exp"`, `"gemini-2.5-flash"`)
- `apiKey`: Your Gemini API key from Google AI Studio

**Embedding Settings (Local Server):**
- `embeddingProvider`: Set to `"local"` for local embedding server
- `embeddingModel`: Model name (mainly for logging, actual model is loaded in the server)
- `embeddingBaseUrl`: URL of your local embedding server
  - Default: `"http://localhost:1234"`
  - For remote server: `"http://your-server:port"`
- `embeddingDim`: Embedding dimensions (768 for nomic models)
- `embeddingBatchSize`: Number of texts to process per request (adjust based on your hardware)

## Benefits of This Setup

1. **Cost Efficiency**: Gemini Flash is very affordable for LLM operations, embeddings are completely free
2. **Data Privacy**: Your text data never leaves your machine during embedding
3. **No Rate Limits**: Local embeddings have no API rate limits
4. **Low Latency**: Local processing eliminates network latency for embeddings
5. **Offline Capability**: Embeddings work even without internet
6. **Full Control**: You control the model, quantization level, and server settings

## Performance Tips

1. **GPU Acceleration**: If you have a CUDA-capable GPU, the server will use it automatically
2. **Quantization**: Use Q4_K_M or Q5_K_M quantized models for good balance of speed and quality
3. **Batch Size**: Increase `embeddingBatchSize` if you have enough RAM/VRAM
4. **Server Location**: For production, you can run the embedding server on a separate machine

## Monitoring Progress

When processing documents, you'll see detailed progress logs:

```
[LocalEmbedding] Using server at http://localhost:1234
[LocalEmbedding] Processing 150 texts in 5 batches (batch_size=32)...
[LocalEmbedding] Processing batch 1/5...
[LocalEmbedding] Processing batch 2/5...
[LocalEmbedding] Completed! Processed 150 texts in 3.45s (43.5 texts/s)
[LocalEmbedding] Embedding shape: (150, 768)
```

## Troubleshooting

If you encounter issues:

1. **Connection Error**: Make sure your local server is running and accessible
2. **Wrong Dimensions**: Check that `embeddingDim` matches your model's output dimensions
3. **Slow Performance**: Try reducing `embeddingBatchSize` or using a smaller quantized model
4. **Server Crashes**: Check server logs and ensure you have enough RAM for the model

## Alternative Configurations

### Different Embedding Models

For other embedding models, adjust the dimensions accordingly:

```json
{
  "embeddingModel": "BAAI/bge-large-en-v1.5-GGUF",
  "embeddingDim": 1024
}
```

### Using Ollama

If using Ollama, the base URL is typically:

```json
{
  "embeddingBaseUrl": "http://localhost:11434",
  "embeddingModel": "nomic-embed-text"
}
```

### Remote Embedding Server

You can also use a remote server:

```json
{
  "embeddingBaseUrl": "http://192.168.1.100:1234",
  "embeddingApiKey": "optional-auth-token"
}
```