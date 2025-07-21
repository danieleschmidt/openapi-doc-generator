"""Utility functions for openapi-doc-generator."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import time
import uuid
from functools import lru_cache
from typing import Any, Dict, Optional


def echo(value: object | None = None) -> object | None:
    """Return the provided value unchanged."""
    return value


# LRU cache for parsed AST trees to avoid repeated parsing
@lru_cache(maxsize=128)
def _parse_ast_cached(source_hash: str, source: str, filename: str) -> ast.AST:
    """Internal cached AST parsing function.
    
    Uses source hash as cache key to ensure content changes invalidate cache.
    """
    return ast.parse(source, filename=filename)


def get_cached_ast(source: str, filename: str) -> ast.AST:
    """Get parsed AST with caching for performance optimization.
    
    Args:
        source: Python source code to parse
        filename: Filename for error reporting
        
    Returns:
        Parsed AST tree
        
    Raises:
        SyntaxError: If source code has syntax errors
    """
    # Create hash of source content for cache key
    source_hash = hashlib.sha256(source.encode('utf-8')).hexdigest()[:16]
    
    return _parse_ast_cached(source_hash, source, filename)


def _clear_ast_cache() -> None:
    """Clear the AST cache. Useful for testing."""
    _parse_ast_cached.cache_clear()


# Global correlation ID for the current execution
_correlation_id: Optional[str] = None
_execution_start_time: Optional[float] = None


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        global _correlation_id, _execution_start_time
        
        if _correlation_id is None:
            _correlation_id = str(uuid.uuid4())[:8]
        
        if _execution_start_time is None:
            _execution_start_time = time.time()
        
        log_entry: Dict[str, Any] = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S.%fZ', time.gmtime()),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'correlation_id': _correlation_id,
        }
        
        # Add timing information if available
        current_time = time.time()
        if _execution_start_time is not None:
            log_entry['execution_time_ms'] = round((current_time - _execution_start_time) * 1000, 2)
        
        # Add extra fields from record if present
        if hasattr(record, 'duration_ms'):
            log_entry['duration_ms'] = record.duration_ms
        
        if hasattr(record, 'timing'):
            log_entry['timing'] = record.timing
            
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, separators=(',', ':'))


def setup_json_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure JSON logging and return logger instance."""
    global _correlation_id, _execution_start_time
    
    # Reset for new execution
    _correlation_id = str(uuid.uuid4())[:8]
    _execution_start_time = time.time()
    
    # Remove any existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create new handler with JSON formatter
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    
    logging.basicConfig(
        level=level,
        handlers=[handler],
        force=True
    )
    
    return logging.getLogger(__name__)


def get_correlation_id() -> str:
    """Get the current correlation ID."""
    global _correlation_id
    if _correlation_id is None:
        _correlation_id = str(uuid.uuid4())[:8]
    return _correlation_id


def reset_correlation_id() -> None:
    """Reset the correlation ID. Useful for testing."""
    global _correlation_id, _execution_start_time
    _correlation_id = None
    _execution_start_time = None
