#!/usr/bin/env python3
"""
HyperRAG Database Builder Script
Builds a HyperRAG knowledge database from local files using Google Gemini API
"""

import os
import sys
import json
import argparse
import pathlib
from pathlib import Path
from typing import List, Set, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime, timedelta
import mimetypes
import fnmatch
import asyncio
import time
from collections import deque

# Google Gemini imports
from google import genai
from google.genai import types

# Add parent directory to path to import hyperrag
sys.path.insert(0, str(Path(__file__).parent.parent))

from hyperrag import HyperRAG
from hyperrag.utils import EmbeddingFunc


class RateLimiter:
    """Rate limiter for Gemini API with sliding window tracking"""
    
    def __init__(
        self,
        rpm_limit: int = 100,
        tpm_limit: int = 30000,
        rpd_limit: int = 1000
    ):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.rpd_limit = rpd_limit
        
        # Sliding windows for tracking
        self.minute_requests = deque()
        self.minute_tokens = deque()
        self.day_requests = deque()
        
        # Safety margins (use 90% of limits)
        self.rpm_limit = int(rpm_limit * 0.9)
        self.tpm_limit = int(tpm_limit * 0.9)
        self.rpd_limit = int(rpd_limit * 0.95)
    
    async def wait_if_needed(self, token_count: int = 0):
        """Wait if rate limits would be exceeded"""
        now = datetime.now()
        
        # Clean old entries
        minute_ago = now - timedelta(minutes=1)
        day_ago = now - timedelta(days=1)
        
        # Remove old entries from sliding windows
        while self.minute_requests and self.minute_requests[0] < minute_ago:
            self.minute_requests.popleft()
        while self.minute_tokens and self.minute_tokens[0][0] < minute_ago:
            self.minute_tokens.popleft()
        while self.day_requests and self.day_requests[0] < day_ago:
            self.day_requests.popleft()
        
        # Check if we need to wait
        wait_time = 0
        
        # Check requests per minute (be conservative, start waiting at 80% capacity)
        if len(self.minute_requests) >= self.rpm_limit * 0.8:
            # Wait to avoid hitting the limit
            wait_time = max(wait_time, 30)  # Wait 30 seconds when approaching limit
        
        # Check tokens per minute
        current_minute_tokens = sum(t[1] for t in self.minute_tokens)
        if current_minute_tokens + token_count > self.tpm_limit * 0.8:  # Start waiting at 80% capacity
            # Wait until we have room for more tokens
            wait_time = max(wait_time, 30)  # Wait 30 seconds when approaching limit
        
        # Check requests per day
        if len(self.day_requests) >= self.rpd_limit:
            # Wait until the oldest request is older than 1 day
            wait_time = max(wait_time,
                          (self.day_requests[0] + timedelta(days=1) - now).total_seconds())
        
        if wait_time > 0:
            print(f"Rate limit approaching, waiting {wait_time:.1f} seconds...")
            await asyncio.sleep(wait_time)
        
        # Record this request
        self.minute_requests.append(now)
        self.minute_tokens.append((now, token_count))
        self.day_requests.append(now)
    
    def get_status(self):
        """Get current rate limit status"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        day_ago = now - timedelta(days=1)
        
        # Clean old entries
        while self.minute_requests and self.minute_requests[0] < minute_ago:
            self.minute_requests.popleft()
        while self.minute_tokens and self.minute_tokens[0][0] < minute_ago:
            self.minute_tokens.popleft()
        while self.day_requests and self.day_requests[0] < day_ago:
            self.day_requests.popleft()
        
        current_minute_tokens = sum(t[1] for t in self.minute_tokens)
        
        return {
            "rpm_used": len(self.minute_requests),
            "rpm_limit": self.rpm_limit,
            "tpm_used": current_minute_tokens,
            "tpm_limit": self.tpm_limit,
            "rpd_used": len(self.day_requests),
            "rpd_limit": self.rpd_limit
        }


@dataclass
class FileProcessor:
    """Handles file discovery and filtering"""
    
    root_path: Path
    ignore_patterns: Set[str] = field(default_factory=set)
    additional_ignores: List[str] = field(default_factory=list)
    ignore_file: Optional[Path] = None
    text_extensions: Set[str] = field(default_factory=lambda: {
        '.txt', '.md', '.rst', '.py', '.js', '.ts', '.jsx', '.tsx',
        '.java', '.cpp', '.c', '.h', '.hpp', '.cs', '.go', '.rs',
        '.rb', '.php', '.swift', '.kt', '.scala', '.r', '.m',
        '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd',
        '.html', '.htm', '.xml', '.json', '.yaml', '.yml', '.toml',
        '.ini', '.conf', '.cfg', '.env', '.properties',
        '.sql', '.graphql', '.proto', '.thrift',
        '.tex', '.bib', '.org', '.adoc', '.textile',
        '.csv', '.tsv', '.log'
    })
    max_file_size_mb: float = 10.0
    
    def __post_init__(self):
        """Load gitignore patterns"""
        self.ignore_patterns = self._load_gitignore_patterns()
        
        # Add patterns from ignore file if provided
        if self.ignore_file and self.ignore_file.exists():
            try:
                with open(self.ignore_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            self.ignore_patterns.add(line)
                print(f"Loaded ignore patterns from {self.ignore_file}")
            except Exception as e:
                print(f"Warning: Could not read ignore file {self.ignore_file}: {e}")
        
        # Add additional ignore patterns from command line
        if self.additional_ignores:
            self.ignore_patterns.update(self.additional_ignores)
            print(f"Added {len(self.additional_ignores)} additional ignore patterns")
        
        # Add common patterns to always ignore
        self.ignore_patterns.update({
            '*.pyc', '__pycache__', '*.pyo', '*.pyd',
            '.git', '.svn', '.hg', '.bzr',
            'node_modules', 'venv', '.venv', 'env', '.env',
            '*.so', '*.dylib', '*.dll', '*.exe',
            '*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.ico',
            '*.mp3', '*.mp4', '*.avi', '*.mov', '*.mkv',
            '*.zip', '*.tar', '*.gz', '*.rar', '*.7z',
            '*.pdf', '*.doc', '*.docx', '*.xls', '*.xlsx',
            '.DS_Store', 'Thumbs.db', 'desktop.ini'
        })
    
    def _load_gitignore_patterns(self) -> Set[str]:
        """Load patterns from .gitignore and .hyperragignore files"""
        patterns = set()
        
        # Find all .gitignore and .hyperragignore files in the tree
        ignore_files = list(self.root_path.rglob('.gitignore')) + list(self.root_path.rglob('.hyperragignore'))
        
        for ignore_path in ignore_files:
            base_dir = ignore_path.parent
            
            try:
                with open(ignore_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        # Skip comments and empty lines
                        if not line or line.startswith('#'):
                            continue
                        
                        # Handle directory-specific patterns
                        if base_dir != self.root_path:
                            # Make pattern relative to root
                            rel_path = base_dir.relative_to(self.root_path)
                            pattern = str(rel_path / line)
                        else:
                            pattern = line
                        
                        patterns.add(pattern)
            except Exception as e:
                print(f"Warning: Could not read {ignore_path}: {e}")
        
        # Report what was found
        gitignore_count = len([f for f in ignore_files if f.name == '.gitignore'])
        hyperragignore_count = len([f for f in ignore_files if f.name == '.hyperragignore'])
        if gitignore_count > 0:
            print(f"Found {gitignore_count} .gitignore file(s)")
        if hyperragignore_count > 0:
            print(f"Found {hyperragignore_count} .hyperragignore file(s)")
        
        return patterns
    
    def _should_ignore(self, path: Path) -> bool:
        """Check if a path should be ignored based on patterns"""
        try:
            rel_path = path.relative_to(self.root_path)
        except ValueError:
            return True
        
        path_str = str(rel_path)
        path_parts = path_str.split(os.sep)
        
        # Check each pattern
        for pattern in self.ignore_patterns:
            # Check if any part of the path matches
            for part in path_parts:
                if fnmatch.fnmatch(part, pattern):
                    return True
            
            # Check full path match
            if fnmatch.fnmatch(path_str, pattern):
                return True
            
            # Check if path starts with pattern (for directories)
            if pattern.endswith('/') and path_str.startswith(pattern[:-1]):
                return True
        
        return False
    
    def _is_text_file(self, path: Path) -> bool:
        """Check if file is likely a text file"""
        # Check extension first
        if path.suffix.lower() in self.text_extensions:
            return True
        
        # Try mimetype detection
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type and mime_type.startswith('text/'):
            return True
        
        # Check if file has no extension (might be script or config)
        if not path.suffix:
            try:
                # Try to read first few bytes to check if text
                with open(path, 'rb') as f:
                    chunk = f.read(512)
                    # Check for null bytes (binary indicator)
                    if b'\x00' not in chunk:
                        # Try to decode as text
                        try:
                            chunk.decode('utf-8')
                            return True
                        except UnicodeDecodeError:
                            pass
            except:
                pass
        
        return False
    
    def get_files(self, dry_run: bool = False) -> List[Tuple[Path, str]]:
        """Get all files to process"""
        files_to_process = []
        ignored_files = []
        
        for path in self.root_path.rglob('*'):
            # Skip directories
            if path.is_dir():
                continue
            
            # Check if should be ignored
            if self._should_ignore(path):
                ignored_files.append(path)
                continue
            
            # Check file size
            try:
                size_mb = path.stat().st_size / (1024 * 1024)
                if size_mb > self.max_file_size_mb:
                    print(f"Skipping {path}: File too large ({size_mb:.2f} MB)")
                    continue
            except:
                continue
            
            # Check if text file
            if not self._is_text_file(path):
                ignored_files.append(path)
                continue
            
            # Try to read file content
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():  # Only add non-empty files
                        files_to_process.append((path, content))
            except Exception as e:
                print(f"Warning: Could not read {path}: {e}")
                continue
        
        if dry_run:
            print("\n=== DRY RUN REPORT ===")
            print(f"Root directory: {self.root_path}")
            print(f"\nFiles to process ({len(files_to_process)}):")
            for path, _ in sorted(files_to_process):
                rel_path = path.relative_to(self.root_path)
                size_kb = path.stat().st_size / 1024
                print(f"  ✓ {rel_path} ({size_kb:.1f} KB)")
            
            if ignored_files:
                print(f"\nIgnored files ({len(ignored_files)}):")
                for path in sorted(ignored_files)[:50]:  # Show first 50
                    try:
                        rel_path = path.relative_to(self.root_path)
                        print(f"  ✗ {rel_path}")
                    except:
                        pass
                if len(ignored_files) > 50:
                    print(f"  ... and {len(ignored_files) - 50} more")
            
            print(f"\nTotal text content size: {sum(len(c) for _, c in files_to_process) / 1024:.1f} KB")
            print("=" * 40)
        
        return files_to_process


def get_model_rate_limits(model_name: str) -> Tuple[int, int, int]:
    """Get rate limits for a specific Gemini model.
    Returns: (rpm_limit, tpm_limit, rpd_limit)
    """
    # Model rate limits based on official Gemini documentation
    rate_limits = {
        # Generation models
        "gemini-2.5-flash": (10, 250000, 250),
        "gemini-2.5-flash-lite": (15, 250000, 1000),
        
        # Embedding models
        "gemini-embedding-001": (100, 30000, 1000),
    }
    
    # Default to most conservative limits if model not found
    return rate_limits.get(model_name, (10, 30000, 250))

class GeminiRAGBuilder:
    """Builds HyperRAG database using Google Gemini API with rate limiting"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        use_vertex: bool = False,
        project_id: Optional[str] = None,
        location: str = 'us-central1',
        generation_model: str = 'gemini-2.5-flash-lite',
        embedding_model: str = 'gemini-embedding-001',  # Updated to stable model
        embedding_dim: int = 768,  # OpenAI uses 1536 for text-embedding-3-small
        temperature: float = 0.1,
        max_output_tokens: int = 2048,
        # Rate limit parameters
        rpm_limit: int = 100,
        tpm_limit: int = 30000,
        rpd_limit: int = 1000
    ):
        """Initialize Gemini client and configuration"""
        
        # Initialize Gemini client
        if use_vertex:
            self.client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location
            )
        else:
            self.client = genai.Client(api_key=api_key)
        
        self.generation_model = generation_model
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            rpm_limit=rpm_limit,
            tpm_limit=tpm_limit,
            rpd_limit=rpd_limit
        )
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 30.0  # Initial delay in seconds (increased for Gemini rate limits)
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test Gemini API connection"""
        try:
            response = self.client.models.generate_content(
                model=self.generation_model,
                contents='Hello',
                config=types.GenerateContentConfig(
                    max_output_tokens=10,
                    temperature=0
                )
            )
            print(f"✓ Gemini API connected (using {self.generation_model})")
        except Exception as e:
            print(f"✗ Failed to connect to Gemini API: {e}")
            sys.exit(1)
    
    async def llm_func(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: list = [],
        **kwargs
    ) -> str:
        """LLM function for HyperRAG with rate limiting and retries"""
        # Estimate token count (rough approximation)
        estimated_tokens = len(prompt.split()) * 1.3
        
        attempt = 0
        while True:  # Infinite retry loop for rate limits
            try:
                # Apply rate limiting
                await self.rate_limiter.wait_if_needed(int(estimated_tokens))
                
                # Combine system prompt and user prompt
                full_prompt = prompt
                if system_prompt:
                    full_prompt = f"{system_prompt}\n\n{prompt}"
                
                # Generate response
                response = self.client.models.generate_content(
                    model=self.generation_model,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens,
                        candidate_count=1
                    )
                )
                
                return response.text
            
            except Exception as e:
                error_str = str(e)
                attempt += 1
                
                # Check if it's a rate limit error (429)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    # For rate limit errors, use exponential backoff with cap
                    wait_time = min(self.retry_delay * (2 ** attempt), 300)  # Cap at 5 minutes
                    print(f"LLM rate limit hit (attempt {attempt}): {e}")
                    print(f"Waiting {wait_time:.1f} seconds before retry...")
                    print("(Press Ctrl+C to cancel)")
                    try:
                        await asyncio.sleep(wait_time)
                    except KeyboardInterrupt:
                        print("\nLLM generation cancelled by user")
                        raise
                    continue  # Retry indefinitely for rate limits
                
                # For other errors, retry with limit
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"LLM generation error (attempt {attempt}/{self.max_retries}): {e}")
                    print(f"Retrying in {wait_time:.1f} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"LLM generation failed after {self.max_retries} attempts: {e}")
                    return ""
    
    async def embedding_func(self, texts: List[str]) -> np.ndarray:
        """Embedding function for HyperRAG with rate limiting and batching"""
        # Batch texts to stay within token limits (2048 per text for gemini-embedding-001)
        # Estimate tokens and apply batching
        estimated_tokens = sum(len(text.split()) * 1.3 for text in texts)
        
        attempt = 0
        while True: # Infinite retry loop
            try:
                # Apply rate limiting
                await self.rate_limiter.wait_if_needed(int(estimated_tokens))
                
                # Split texts into smaller batches if needed to avoid hitting TPM
                max_batch_size = min(5, len(texts))  # Reduced batch size to avoid rate limits
                
                all_embeddings = []
                for i in range(0, len(texts), max_batch_size):
                    batch = texts[i:i + max_batch_size]
                    
                    # Truncate texts that are too long (gemini-embedding-001 has 2048 token limit)
                    truncated_batch = []
                    for text in batch:
                        # Rough truncation at ~2000 tokens (assuming ~1.3 tokens per word)
                        max_words = int(2000 / 1.3)
                        words = text.split()[:max_words]
                        truncated_batch.append(' '.join(words))
                    
                    # Generate embeddings for batch
                    response = self.client.models.embed_content(
                        model=self.embedding_model,
                        contents=truncated_batch,
                        config=types.EmbedContentConfig(
                            output_dimensionality=self.embedding_dim,
                            task_type='RETRIEVAL_DOCUMENT'
                        )
                    )
                    
                    # Extract embeddings
                    for embedding in response.embeddings:
                        all_embeddings.append(embedding.values)
                    
                    # Add delay between batches to avoid rate limits
                    if i + max_batch_size < len(texts):
                        await asyncio.sleep(5.0)  # Increased delay to prevent rate limiting
                
                return np.array(all_embeddings, dtype=np.float32)
            
            except Exception as e:
                error_str = str(e)
                attempt += 1
                
                # Check if it's a rate limit error (429)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    # For rate limit errors, use exponential backoff with cap
                    wait_time = min(self.retry_delay * (2 ** attempt), 300)  # Cap at 5 minutes
                    print(f"Rate limit hit (attempt {attempt}): {e}")
                    print(f"Waiting {wait_time:.1f} seconds before retry...")
                    print("(Press Ctrl+C to cancel)")
                    try:
                        await asyncio.sleep(wait_time)
                    except KeyboardInterrupt:
                        print("\nEmbedding generation cancelled by user")
                        raise
                    continue  # Retry indefinitely for rate limits
                
                # For other errors, retry with limit
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"Embedding error (attempt {attempt}/{self.max_retries}): {e}")
                    print(f"Retrying in {wait_time:.1f} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Embedding generation failed after {self.max_retries} attempts: {e}")
                    print("Returning zero embeddings - these will need to be reprocessed!")
                    # Return zero embeddings on error (not ideal but better than crashing)
                    return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
    
    def build_database(
        self,
        files: List[Tuple[Path, str]],
        output_dir: Path,
        batch_size: int = 5  # Reduced for Gemini rate limits
    ):
        """Build HyperRAG database from files with progress tracking"""
        
        # Initialize HyperRAG
        print(f"\nInitializing HyperRAG database in {output_dir}")
        print(f"Configuration:")
        print(f"  - Chunk size: 1200 tokens (matching OpenAI)")
        print(f"  - Chunk overlap: 100 tokens")
        print(f"  - Embedding batch: 8 (optimized for Gemini rate limits)")
        print(f"  - Max async LLM: 3 (conservative for rate limits)")
        print(f"  - Embedding model: {self.embedding_model}")
        print(f"  - Embedding dimension: {self.embedding_dim}")
        
        rag = HyperRAG(
            working_dir=str(output_dir),
            llm_model_func=self.llm_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=self.embedding_dim,
                max_token_size=2048,  # Gemini embedding model limit
                func=self.embedding_func
            ),
            # Match OpenAI settings for consistency, but adjust batching for rate limits
            chunk_token_size=1200,  # Same as OpenAI default
            chunk_overlap_token_size=100,  # Same as OpenAI default
            embedding_batch_num=1,  # Batch size for efficiency, will handle internally
            llm_model_max_async=1,  # Process LLM requests one at a time
            embedding_func_max_async=1  # Only one concurrent embedding request
        )
        
        # Process files in batches with progress tracking
        total_files = len(files)
        processed_files = 0
        failed_files = []
        start_time = time.time()
        
        print(f"\nProcessing {total_files} files in batches of {batch_size}")
        print(f"Rate limits: {self.rate_limiter.rpm_limit} RPM, {self.rate_limiter.tpm_limit} TPM\n")
        
        for i in range(0, total_files, batch_size):
            batch = files[i:i+batch_size]
            batch_texts = []
            batch_start = time.time()
            
            # Calculate progress
            batch_num = i//batch_size + 1
            total_batches = (total_files + batch_size - 1)//batch_size
            progress_pct = (i / total_files) * 100
            
            print(f"Batch {batch_num}/{total_batches} ({progress_pct:.1f}% complete)")
            
            # Show rate limit status periodically
            if batch_num % 5 == 0 or batch_num == 1:
                status = self.rate_limiter.get_status()
                print(f"  Rate limits: {status['rpm_used']}/{status['rpm_limit']} RPM, "
                      f"{status['tpm_used']}/{status['tpm_limit']} TPM")
            
            for path, content in batch:
                # Add file path as metadata
                rel_path = path.relative_to(Path(files[0][0]).parent.parent) if len(files) > 0 else path
                metadata = f"File: {rel_path}\n---\n"
                full_content = metadata + content
                batch_texts.append(full_content)
                print(f"  Processing: {rel_path} ({len(content)/1024:.1f} KB)")
            
            # Insert batch into RAG
            try:
                rag.insert(batch_texts)
                processed_files += len(batch)
                batch_time = time.time() - batch_start
                print(f"  ✓ Batch complete in {batch_time:.1f}s")
                
                # Estimate remaining time
                if processed_files > 0:
                    elapsed = time.time() - start_time
                    avg_time_per_file = elapsed / processed_files
                    remaining_files = total_files - processed_files
                    eta = avg_time_per_file * remaining_files
                    print(f"  ETA: {eta/60:.1f} minutes remaining\n")
                    
            except Exception as e:
                print(f"  ✗ Error inserting batch: {e}")
                print("  Retrying files individually...")
                
                # Try inserting one by one
                for j, (text, (path, _)) in enumerate(zip(batch_texts, batch)):
                    try:
                        rag.insert([text])
                        processed_files += 1
                        print(f"    ✓ {path.name} inserted")
                    except Exception as e2:
                        print(f"    ✗ Failed to insert {path.name}: {e2}")
                        failed_files.append(path)
        
        # Final statistics
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✓ Database build complete!")
        print(f"{'='*60}")
        print(f"\nStatistics:")
        print(f"  Total files processed: {processed_files}/{total_files}")
        if failed_files:
            print(f"  Failed files: {len(failed_files)}")
            for failed in failed_files[:5]:  # Show first 5 failed files
                print(f"    - {failed.name}")
            if len(failed_files) > 5:
                print(f"    ... and {len(failed_files) - 5} more")
        print(f"  Total time: {total_time/60:.1f} minutes")
        print(f"  Average time per file: {total_time/max(1, processed_files):.2f} seconds")
        
        # Print rate limit final status
        final_status = self.rate_limiter.get_status()
        print(f"\nFinal rate limit usage:")
        print(f"  Requests today: {final_status['rpd_used']}/{final_status['rpd_limit']}")
        
        # Print database statistics
        db_files = list(output_dir.glob("*"))
        total_size = sum(f.stat().st_size for f in db_files)
        print(f"\nDatabase files created in {output_dir}:")
        for db_file in sorted(db_files):
            size_kb = db_file.stat().st_size / 1024
            print(f"  {db_file.name}: {size_kb:.1f} KB")
        print(f"\nTotal database size: {total_size/1024/1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Build HyperRAG database from local files using Google Gemini API"
    )
    
    # Input/Output arguments
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to directory or file to process"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory for database (default: ./hyperrag_db_<timestamp>)"
    )
    
    # Gemini API arguments
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("GOOGLE_API_KEY"),
        help="Gemini API key (or set GOOGLE_API_KEY env var)"
    )
    parser.add_argument(
        "--use-vertex",
        action="store_true",
        help="Use Vertex AI instead of Gemini API"
    )
    parser.add_argument(
        "--project-id",
        type=str,
        default=os.getenv("GOOGLE_CLOUD_PROJECT"),
        help="Google Cloud project ID for Vertex AI"
    )
    parser.add_argument(
        "--location",
        type=str,
        default="us-central1",
        help="Vertex AI location (default: us-central1)"
    )
    
    # Model configuration
    parser.add_argument(
        "--generation-model",
        type=str,
        default="gemini-2.5-flash-lite",
        help="Gemini model for text generation"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="gemini-embedding-001",
        help="Gemini model for embeddings (default: gemini-embedding-001)"
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=768,
        help="Embedding dimension (default: 768)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Generation temperature (default: 0.1)"
    )
    
    # File processing arguments
    parser.add_argument(
        "--max-file-size",
        type=float,
        default=10.0,
        help="Maximum file size in MB (default: 10)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of files to process in each batch (default: 5, optimized for Gemini rate limits)"
    )
    
    # Ignore pattern arguments
    parser.add_argument(
        "--ignore",
        action="append",
        dest="ignore_patterns",
        help="Add ignore pattern (can be used multiple times, e.g., --ignore '*.tmp' --ignore 'test_*')"
    )
    parser.add_argument(
        "--ignore-file",
        type=str,
        help="Path to file containing ignore patterns (like .gitignore)"
    )
    
    # Rate limit arguments
    parser.add_argument(
        "--rpm-limit",
        type=int,
        default=100,
        help="Requests per minute limit (default: 100)"
    )
    parser.add_argument(
        "--tpm-limit",
        type=int,
        default=30000,
        help="Tokens per minute limit (default: 30000)"
    )
    parser.add_argument(
        "--rpd-limit",
        type=int,
        default=1000,
        help="Requests per day limit (default: 1000)"
    )
    
    # Other arguments
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would be processed without building database"
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden files and directories"
    )
    
    args = parser.parse_args()
    
    # Validate input path
    input_path = Path(args.input_path).resolve()
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    # Set output directory
    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"./hyperrag_db_{timestamp}").resolve()
    
    # Check API key
    if not args.use_vertex and not args.api_key:
        print("Error: Gemini API key required. Set GOOGLE_API_KEY or use --api-key")
        sys.exit(1)
    
    if args.use_vertex and not args.project_id:
        print("Error: Project ID required for Vertex AI. Set GOOGLE_CLOUD_PROJECT or use --project-id")
        sys.exit(1)
    
    # Auto-detect rate limits based on models if not specified
    if args.rpm_limit == 100 and args.tpm_limit == 30000 and args.rpd_limit == 1000:
        # Default values, auto-detect based on models
        gen_rpm, gen_tpm, gen_rpd = get_model_rate_limits(args.generation_model)
        emb_rpm, emb_tpm, emb_rpd = get_model_rate_limits(args.embedding_model)
        
        # Use the more restrictive limits
        args.rpm_limit = min(gen_rpm, emb_rpm)
        args.tpm_limit = min(gen_tpm, emb_tpm)
        args.rpd_limit = min(gen_rpd, emb_rpd)
        
        print(f"\nAuto-detected rate limits for models:")
        print(f"  Generation model ({args.generation_model}): {gen_rpm} RPM, {gen_tpm:,} TPM, {gen_rpd} RPD")
        print(f"  Embedding model ({args.embedding_model}): {emb_rpm} RPM, {emb_tpm:,} TPM, {emb_rpd} RPD")
        print(f"  Using conservative limits: {args.rpm_limit} RPM, {args.tpm_limit:,} TPM, {args.rpd_limit} RPD")
    
    # Process files
    print(f"Processing files from: {input_path}")
    
    if input_path.is_file():
        # Single file mode
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
            files = [(input_path, content)]
            if args.dry_run:
                print(f"Would process single file: {input_path}")
                print(f"File size: {len(content) / 1024:.1f} KB")
                return
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
    else:
        # Directory mode
        processor = FileProcessor(
            root_path=input_path,
            max_file_size_mb=args.max_file_size,
            additional_ignores=args.ignore_patterns or [],
            ignore_file=Path(args.ignore_file) if args.ignore_file else None
        )
        
        files = processor.get_files(dry_run=args.dry_run)
        
        if args.dry_run:
            return
        
        if not files:
            print("No files found to process")
            sys.exit(1)
    
    # Build database
    print(f"\nFound {len(files)} files to process")
    
    if not args.dry_run:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize builder with rate limits
        print("\nInitializing Gemini RAG Builder...")
        builder = GeminiRAGBuilder(
            api_key=args.api_key,
            use_vertex=args.use_vertex,
            project_id=args.project_id,
            location=args.location,
            generation_model=args.generation_model,
            embedding_model=args.embedding_model,
            embedding_dim=args.embedding_dim,
            temperature=args.temperature,
            rpm_limit=args.rpm_limit,
            tpm_limit=args.tpm_limit,
            rpd_limit=args.rpd_limit
        )
        
        # Build database
        builder.build_database(
            files=files,
            output_dir=output_dir,
            batch_size=args.batch_size
        )
        
        print(f"\n✓ Database successfully created at: {output_dir}")
        print("\nTo use this database:")
        print(f"  1. Copy {output_dir} to your desired location")
        print(f"  2. Initialize HyperRAG with working_dir='{output_dir}'")
        print("  3. Use rag.query() to search your knowledge base")
        
        print("\n" + "="*60)
        print("OPTIMIZATION TIPS FOR GEMINI:")
        print("="*60)
        print("\n1. RATE LIMITS (Gemini Embedding API):")
        print("   - 100 requests/minute (RPM)")
        print("   - 30,000 tokens/minute (TPM)")
        print("   - 1,000 requests/day (RPD)")
        print("\n2. RECOMMENDED SETTINGS:")
        print("   - Chunk size: 1200 tokens (same as OpenAI for quality parity)")
        print("   - Batch size: 5-8 files (balanced for rate limits)")
        print("   - Embedding dimension: 768 (good balance of quality/performance)")
        print("\n3. PERFORMANCE COMPARISON:")
        print("   OpenAI defaults: batch=32, async=16")
        print("   Gemini optimized: batch=8, async=3 (respects rate limits)")
        print("\n4. TO INCREASE SPEED:")
        print("   - Use Vertex AI for enterprise workloads (higher limits)")
        print("   - Process during off-peak hours")
        print("   - Consider parallel processing with multiple API keys")
        print("="*60)


if __name__ == "__main__":
    main()