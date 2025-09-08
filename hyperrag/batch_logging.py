"""
Batch logging utilities to reduce log verbosity for operations that process many items.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime


class BatchProgressLogger:
    """
    Logs progress for batch operations in a condensed format.
    Instead of logging each item, logs summary at intervals.
    """
    
    def __init__(self, logger: logging.Logger, operation_name: str, 
                 total_items: int, log_interval: int = 10):
        """
        Initialize batch logger.
        
        Args:
            logger: The logger instance to use
            operation_name: Name of the operation (e.g., "Entity extraction")
            total_items: Total number of items to process
            log_interval: Log progress every N items (default 10)
        """
        self.logger = logger
        self.operation_name = operation_name
        self.total_items = total_items
        self.log_interval = log_interval
        self.processed = 0
        self.start_time = datetime.now()
        self.stats = {
            'successful': 0,
            'failed': 0,
            'entities': 0,
            'relationships': 0
        }
    
    def update(self, success: bool = True, entities: int = 0, relationships: int = 0):
        """Update progress and log if at interval."""
        self.processed += 1
        if success:
            self.stats['successful'] += 1
            self.stats['entities'] += entities
            self.stats['relationships'] += relationships
        else:
            self.stats['failed'] += 1
        
        # Log at intervals or when complete
        should_log = (
            self.processed % self.log_interval == 0 or 
            self.processed == self.total_items
        )
        
        if should_log:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = self.processed / elapsed if elapsed > 0 else 0
            
            self.logger.info(
                f"{self.operation_name} progress | "
                f"processed={self.processed}/{self.total_items} "
                f"({self.processed/self.total_items*100:.0f}%) | "
                f"rate={rate:.1f}/s | "
                f"entities={self.stats['entities']} | "
                f"relationships={self.stats['relationships']}"
            )
    
    def complete(self):
        """Log final summary."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        self.logger.info(
            f"{self.operation_name} complete | "
            f"total={self.total_items} | "
            f"successful={self.stats['successful']} | "
            f"failed={self.stats['failed']} | "
            f"duration={elapsed:.1f}s | "
            f"entities={self.stats['entities']} | "
            f"relationships={self.stats['relationships']}"
        )