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
from datetime import datetime
import mimetypes
import fnmatch

# Google Gemini imports
from google import genai
from google.genai import types

# Add parent directory to path to import hyperrag
sys.path.insert(0, str(Path(__file__).parent.parent))

from hyperrag import HyperRAG
from hyperrag.utils import EmbeddingFunc


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


class GeminiRAGBuilder:
    """Builds HyperRAG database using Google Gemini API"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        use_vertex: bool = False,
        project_id: Optional[str] = None,
        location: str = 'us-central1',
        generation_model: str = 'gemini-2.0-flash-exp',
        embedding_model: str = 'text-embedding-004',
        embedding_dim: int = 768,
        temperature: float = 0.1,
        max_output_tokens: int = 2048
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
        """LLM function for HyperRAG"""
        try:
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
            print(f"LLM generation error: {e}")
            return ""
    
    async def embedding_func(self, texts: List[str]) -> np.ndarray:
        """Embedding function for HyperRAG"""
        try:
            # Generate embeddings
            response = self.client.models.embed_content(
                model=self.embedding_model,
                contents=texts,
                config=types.EmbedContentConfig(
                    output_dimensionality=self.embedding_dim,
                    task_type='RETRIEVAL_DOCUMENT',
                    auto_truncate=True
                )
            )
            
            # Convert to numpy array
            embeddings = []
            for embedding in response.embeddings:
                embeddings.append(embedding.values)
            
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            print(f"Embedding generation error: {e}")
            # Return zero embeddings on error
            return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
    
    def build_database(
        self,
        files: List[Tuple[Path, str]],
        output_dir: Path,
        batch_size: int = 10
    ):
        """Build HyperRAG database from files"""
        
        # Initialize HyperRAG
        print(f"\nInitializing HyperRAG database in {output_dir}")
        rag = HyperRAG(
            working_dir=str(output_dir),
            llm_model_func=self.llm_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=self.embedding_dim,
                max_token_size=8192,
                func=self.embedding_func
            ),
            # Tune these for Gemini
            chunk_token_size=1000,
            chunk_overlap_token_size=100,
            embedding_batch_num=20,
            llm_model_max_async=5  # Gemini rate limits
        )
        
        # Process files in batches
        total_files = len(files)
        for i in range(0, total_files, batch_size):
            batch = files[i:i+batch_size]
            batch_texts = []
            
            print(f"\nProcessing batch {i//batch_size + 1}/{(total_files + batch_size - 1)//batch_size}")
            
            for path, content in batch:
                # Add file path as metadata
                rel_path = path.relative_to(Path(files[0][0]).parent.parent) if len(files) > 0 else path
                metadata = f"File: {rel_path}\n---\n"
                full_content = metadata + content
                batch_texts.append(full_content)
                print(f"  Adding: {rel_path}")
            
            # Insert batch into RAG
            try:
                rag.insert(batch_texts)
                print(f"  ✓ Batch inserted successfully")
            except Exception as e:
                print(f"  ✗ Error inserting batch: {e}")
                print("  Retrying with smaller chunks...")
                # Try inserting one by one
                for text in batch_texts:
                    try:
                        rag.insert(text)
                    except Exception as e2:
                        print(f"    Failed to insert file: {e2}")
        
        print(f"\n✓ Database built successfully in {output_dir}")
        
        # Print database statistics
        db_files = list(output_dir.glob("*"))
        print("\nDatabase files created:")
        for db_file in db_files:
            size_kb = db_file.stat().st_size / 1024
            print(f"  {db_file.name}: {size_kb:.1f} KB")


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
        default="gemini-2.0-flash-exp",
        help="Gemini model for text generation"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-004",
        help="Gemini model for embeddings"
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
        default=10,
        help="Number of files to process in each batch"
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
        
        # Initialize builder
        builder = GeminiRAGBuilder(
            api_key=args.api_key,
            use_vertex=args.use_vertex,
            project_id=args.project_id,
            location=args.location,
            generation_model=args.generation_model,
            embedding_model=args.embedding_model,
            embedding_dim=args.embedding_dim,
            temperature=args.temperature
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


if __name__ == "__main__":
    main()