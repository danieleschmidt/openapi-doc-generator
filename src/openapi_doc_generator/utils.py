"""Utility functions for openapi-doc-generator."""

from __future__ import annotations

import ast
import concurrent.futures
import hashlib
import json
import logging
import threading
import time
import tracemalloc
import uuid
from collections import deque
from dataclasses import dataclass
from functools import lru_cache, wraps
from typing import Any, Callable, TypeVar

import psutil

from .config import config


def echo(value: object | None = None) -> object | None:
    """Return the provided value unchanged."""
    return value


@dataclass
class PerformanceMetrics:
    """Advanced performance metrics for scaling optimization."""
    operation_name: str
    duration_ms: float
    memory_usage_mb: float
    cpu_percent: float
    timestamp: float
    thread_count: int = 1
    cache_hit_rate: float = 0.0
    processing_rate: float = 0.0  # items per second


class AdvancedPerformanceTracker:
    """Advanced performance tracker with scaling optimization."""

    def __init__(self, max_history: int = 1000):
        self.metrics_history: deque = deque(maxlen=max_history)
        self.operation_stats: dict[str, list[PerformanceMetrics]] = {}
        self.lock = threading.RLock()
        self.start_time = time.time()

    def record_metric(self, metric: PerformanceMetrics):
        """Record a performance metric with thread safety."""
        with self.lock:
            self.metrics_history.append(metric)
            if metric.operation_name not in self.operation_stats:
                self.operation_stats[metric.operation_name] = []
            self.operation_stats[metric.operation_name].append(metric)

    def get_operation_stats(self, operation_name: str) -> dict[str, float]:
        """Get aggregated statistics for a specific operation."""
        with self.lock:
            if operation_name not in self.operation_stats:
                return {}

            metrics = self.operation_stats[operation_name]
            if not metrics:
                return {}

            durations = [m.duration_ms for m in metrics]
            memory_usage = [m.memory_usage_mb for m in metrics]

            return {
                'count': len(metrics),
                'avg_duration_ms': sum(durations) / len(durations),
                'max_duration_ms': max(durations),
                'min_duration_ms': min(durations),
                'avg_memory_mb': sum(memory_usage) / len(memory_usage) if memory_usage else 0,
                'total_duration_ms': sum(durations),
                'operations_per_second': len(metrics) / (time.time() - self.start_time)
            }

    def get_system_health(self) -> dict[str, Any]:
        """Get current system health metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
                'healthy': cpu_percent < 80 and memory.percent < 85 and disk.percent < 90
            }
        except Exception:
            return {'healthy': True, 'error': 'Unable to get system metrics'}


# Global performance tracker instance
_performance_tracker = AdvancedPerformanceTracker()


class ConcurrentProcessingPool:
    """Thread pool for concurrent processing with intelligent scaling."""

    def __init__(self, max_workers: int | None = None):
        self.max_workers = max_workers or min(32, (psutil.cpu_count() or 1) + 4)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self.active_tasks = 0
        self.lock = threading.Lock()

    def submit_task(self, func: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit a task for concurrent execution."""
        with self.lock:
            self.active_tasks += 1

        future = self.executor.submit(func, *args, **kwargs)
        future.add_done_callback(lambda f: self._task_completed())
        return future

    def _task_completed(self):
        """Callback when task is completed."""
        with self.lock:
            self.active_tasks = max(0, self.active_tasks - 1)

    def get_load_factor(self) -> float:
        """Get current load factor (0.0 to 1.0)."""
        with self.lock:
            return self.active_tasks / self.max_workers

    def shutdown(self, wait: bool = True):
        """Shutdown the thread pool."""
        self.executor.shutdown(wait=wait)


# Global concurrent processing pool
_processing_pool = ConcurrentProcessingPool()


def get_processing_pool() -> ConcurrentProcessingPool:
    """Get the global processing pool."""
    return _processing_pool


def get_performance_tracker() -> AdvancedPerformanceTracker:
    """Get the global performance tracker."""
    return _performance_tracker


# LRU cache for parsed AST trees to avoid repeated parsing
@lru_cache(maxsize=config.AST_CACHE_SIZE)
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
_correlation_id: str | None = None
_execution_start_time: float | None = None


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        global _correlation_id, _execution_start_time

        if _correlation_id is None:
            _correlation_id = str(uuid.uuid4())[:8]

        if _execution_start_time is None:
            _execution_start_time = time.time()

        log_entry: dict[str, Any] = {
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
_performance_stats: dict[str, dict[str, Any]] = {}

F = TypeVar("F", bound=Callable[..., Any])


def set_performance_tracking(enabled: bool) -> None:
    """Enable or disable performance tracking."""
    global _performance_tracking_enabled
    _performance_tracking_enabled = enabled


def get_performance_summary() -> dict[str, dict[str, Any]]:
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
                            (peak_memory - memory_start) / config.MEMORY_CONVERSION_FACTOR, 2
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
