"""
Clean logging utilities for file processing operations.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime


class FileProcessingLogger:
    """Manages clean, structured logging for file processing operations."""
    
    def __init__(self, logger_name: str = "file_processing"):
        self.logger = logging.getLogger(logger_name)
    
    def log_batch_start(self, total_files: int, config: Dict[str, Any]):
        """Log the start of a batch file processing operation."""
        self.logger.info(
            f"Starting batch processing | files={total_files} | "
            f"chunk_size={config.get('chunk_size')} | "
            f"chunk_overlap={config.get('chunk_overlap')}"
        )
    
    def log_file_start(self, file_num: int, total: int, file_id: str, filename: str):
        """Log the start of processing a single file."""
        self.logger.info(
            f"Processing file {file_num}/{total} | "
            f"file={filename} | id={file_id[:8]}..."
        )
    
    def log_file_info(self, filename: str, size: int, database: str):
        """Log file information."""
        size_mb = size / (1024 * 1024)
        self.logger.debug(
            f"File info | name={filename} | "
            f"size={size_mb:.2f}MB | database={database}"
        )
    
    def log_processing_stage(self, stage: str, file: str, details: Optional[str] = None):
        """Log a processing stage for a file."""
        msg = f"Stage: {stage} | file={file}"
        if details:
            msg += f" | {details}"
        self.logger.info(msg)
    
    def log_chunk_processing(self, file: str, chunks: int):
        """Log chunk processing information."""
        self.logger.info(f"Chunking complete | file={file} | chunks={chunks}")
    
    def log_entity_extraction(self, file: str, entities: int, relationships: int, duration: float):
        """Log entity extraction results."""
        self.logger.info(
            f"Entity extraction | file={file} | "
            f"entities={entities} | relationships={relationships} | "
            f"duration={duration:.1f}s"
        )
    
    def log_embedding_progress(self, file: str, current: int, total: int):
        """Log embedding progress."""
        percentage = (current / total * 100) if total > 0 else 0
        self.logger.debug(
            f"Embedding progress | file={file} | "
            f"progress={current}/{total} ({percentage:.0f}%)"
        )
    
    def log_file_complete(self, file: str, duration: float, status: str = "success"):
        """Log file processing completion."""
        self.logger.info(
            f"File complete | file={file} | "
            f"status={status} | duration={duration:.1f}s"
        )
    
    def log_batch_complete(self, successful: int, failed: int, duration: float):
        """Log batch processing completion."""
        self.logger.info(
            f"Batch complete | successful={successful} | "
            f"failed={failed} | duration={duration:.1f}s"
        )
    
    def log_error(self, file: str, error: str, stage: Optional[str] = None):
        """Log an error during file processing."""
        msg = f"Processing error | file={file}"
        if stage:
            msg += f" | stage={stage}"
        msg += f" | error={error}"
        self.logger.error(msg)


def create_file_logger() -> FileProcessingLogger:
    """Create and return a file processing logger instance."""
    return FileProcessingLogger()