"""Utility functions for openapi-doc-generator."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import time
import uuid
import tracemalloc
from functools import lru_cache, wraps
from typing import Any, Dict, Optional, Callable, TypeVar


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
    logger = logging.getLogger(__name__)

    # Create hash of source content for cache key
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]

    # Check if we already have this in cache
    cache_info = _parse_ast_cached.cache_info()
    initial_hits = cache_info.hits

    result = _parse_ast_cached(source_hash, source, filename)

    # Log cache performance
    new_cache_info = _parse_ast_cached.cache_info()
    if new_cache_info.hits > initial_hits:
        logger.debug(
            f"AST cache hit for {filename}",
            extra={
                "operation": "ast_cache",
                "cache_hit": True,
                "source_file": filename,
                "cache_size": new_cache_info.currsize,
                "hit_rate": new_cache_info.hits
                / (new_cache_info.hits + new_cache_info.misses),
            },
        )
    else:
        logger.debug(
            f"AST cache miss for {filename}",
            extra={
                "operation": "ast_cache",
                "cache_hit": False,
                "source_file": filename,
                "cache_size": new_cache_info.currsize,
                "hit_rate": new_cache_info.hits
                / (new_cache_info.hits + new_cache_info.misses)
                if (new_cache_info.hits + new_cache_info.misses) > 0
                else 0,
            },
        )

    return result


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
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime()),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": _correlation_id,
        }

        # Add timing information if available
        current_time = time.time()
        if _execution_start_time is not None:
            log_entry["execution_time_ms"] = round(
                (current_time - _execution_start_time) * 1000, 2
            )

        # Add extra fields from record if present
        if hasattr(record, "duration_ms"):
            log_entry["duration_ms"] = record.duration_ms

        if hasattr(record, "timing"):
            log_entry["timing"] = record.timing

        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, separators=(",", ":"))


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

    logging.basicConfig(level=level, handlers=[handler], force=True)

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


# Performance tracking globals
_performance_tracking_enabled = True
_performance_stats: Dict[str, Dict[str, Any]] = {}

F = TypeVar("F", bound=Callable[..., Any])


def set_performance_tracking(enabled: bool) -> None:
    """Enable or disable performance tracking."""
    global _performance_tracking_enabled
    _performance_tracking_enabled = enabled


def get_performance_summary() -> Dict[str, Dict[str, Any]]:
    """Get aggregated performance statistics."""
    return _performance_stats.copy()


def clear_performance_stats() -> None:
    """Clear all performance statistics. Useful for testing."""
    global _performance_stats
    _performance_stats = {}


def measure_performance(operation_name: str) -> Callable[[F], F]:
    """Decorator to measure and log performance metrics for a function.

    Args:
        operation_name: Name of the operation for logging and aggregation

    Returns:
        Decorated function that logs performance metrics
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not _performance_tracking_enabled:
                return func(*args, **kwargs)

            logger = logging.getLogger(func.__module__)

            # Start memory tracking if available
            memory_start = None
            memory_peak = None
            if hasattr(tracemalloc, "start"):
                if not tracemalloc.is_tracing():
                    tracemalloc.start()
                memory_start = tracemalloc.get_traced_memory()[0]

            # Start timing
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Calculate duration
                end_time = time.perf_counter()
                duration_ms = round((end_time - start_time) * 1000, 2)

                # Calculate memory usage if tracking is available
                if memory_start is not None and hasattr(
                    tracemalloc, "get_traced_memory"
                ):
                    try:
                        current_memory, peak_memory = tracemalloc.get_traced_memory()
                        memory_peak = round(
                            (peak_memory - memory_start) / (1024 * 1024), 2
                        )
                    except (ValueError, AttributeError, OSError) as e:
                        # Handle specific tracemalloc errors
                        logger.debug("Memory tracking failed: %s", e)
                        memory_peak = None

                # Update aggregated statistics
                if operation_name not in _performance_stats:
                    _performance_stats[operation_name] = {
                        "count": 0,
                        "total_duration_ms": 0,
                        "avg_duration_ms": 0,
                        "min_duration_ms": float("inf"),
                        "max_duration_ms": 0,
                    }

                stats = _performance_stats[operation_name]
                stats["count"] += 1
                stats["total_duration_ms"] += duration_ms
                stats["avg_duration_ms"] = stats["total_duration_ms"] / stats["count"]
                stats["min_duration_ms"] = min(stats["min_duration_ms"], duration_ms)
                stats["max_duration_ms"] = max(stats["max_duration_ms"], duration_ms)

                # Log performance metrics
                extra_data = {
                    "operation": operation_name,
                    "duration_ms": duration_ms,
                    "correlation_id": get_correlation_id(),
                }

                if memory_peak is not None:
                    extra_data["memory_peak_mb"] = memory_peak

                logger.info(
                    f"Performance: {operation_name} completed in {duration_ms}ms",
                    extra=extra_data,
                )

        return wrapper

    return decorator
