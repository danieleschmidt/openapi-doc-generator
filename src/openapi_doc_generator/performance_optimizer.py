"""
Advanced Performance Optimization System

This module provides comprehensive performance optimization capabilities including
intelligent caching, concurrent processing, memory optimization, and adaptive
scaling for the OpenAPI documentation generator.
"""

import asyncio
import gc
import hashlib
import multiprocessing as mp
import pickle
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import psutil

from .enhanced_monitoring import get_monitor


class CacheStrategy(Enum):
    """Cache strategies for different types of operations."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


class ProcessingMode(Enum):
    """Processing modes for different workload types."""
    SEQUENTIAL = "sequential"
    THREADED = "threaded"
    PROCESS_POOL = "process_pool"
    ASYNC_CONCURRENT = "async_concurrent"
    HYBRID = "hybrid"


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    operation: str
    duration: float
    memory_usage: float
    cpu_usage: float
    cache_hits: int = 0
    cache_misses: int = 0
    parallel_tasks: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    enable_caching: bool = True
    cache_strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    max_cache_size: int = 1000
    cache_ttl: float = 3600.0  # 1 hour
    
    enable_parallel_processing: bool = True
    processing_mode: ProcessingMode = ProcessingMode.HYBRID
    max_workers: Optional[int] = None
    
    enable_memory_optimization: bool = True
    memory_threshold: float = 0.8  # 80% of available memory
    
    enable_adaptive_scaling: bool = True
    scaling_threshold: float = 0.7  # 70% resource utilization


class AdvancedCache:
    """Advanced caching system with multiple strategies."""
    
    def __init__(self, strategy: CacheStrategy = CacheStrategy.ADAPTIVE, 
                 max_size: int = 1000, ttl: float = 3600.0):
        self.strategy = strategy
        self.max_size = max_size
        self.ttl = ttl
        
        # Storage
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.creation_times: Dict[str, float] = {}
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Thread safety
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            # Check TTL
            if self.strategy in [CacheStrategy.TTL, CacheStrategy.ADAPTIVE]:
                if time.time() - self.creation_times[key] > self.ttl:
                    self._remove(key)
                    self.misses += 1
                    return None
            
            # Update access statistics
            self.access_times[key] = time.time()
            self.access_counts[key] += 1
            self.hits += 1
            
            return self.cache[key]
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        with self.lock:
            current_time = time.time()
            
            # If cache is full, evict based on strategy
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict()
            
            # Store value
            self.cache[key] = value
            self.access_times[key] = current_time
            self.creation_times[key] = current_time
            self.access_counts[key] = 1
    
    def _evict(self) -> None:
        """Evict items based on cache strategy."""
        if not self.cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            oldest_key = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
        elif self.strategy == CacheStrategy.TTL:
            # Remove oldest by creation time
            oldest_key = min(self.creation_times.keys(), key=lambda k: self.creation_times[k])
        else:  # ADAPTIVE
            # Combine multiple factors
            current_time = time.time()
            scores = {}
            for key in self.cache.keys():
                age_score = (current_time - self.creation_times[key]) / self.ttl
                freq_score = 1.0 / (self.access_counts[key] + 1)
                recency_score = (current_time - self.access_times[key]) / 3600.0
                scores[key] = age_score + freq_score + recency_score
            oldest_key = max(scores.keys(), key=lambda k: scores[k])
        
        self._remove(oldest_key)
        self.evictions += 1
    
    def _remove(self, key: str) -> None:
        """Remove item from cache."""
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
            del self.access_counts[key]
            del self.creation_times[key]
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.creation_times.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": hit_rate,
                "strategy": self.strategy.value
            }


class ParallelProcessor:
    """Advanced parallel processing system."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.monitor = get_monitor()
        
        # Determine optimal worker count
        if config.max_workers:
            self.max_workers = config.max_workers
        else:
            # Use CPU count but consider memory constraints
            cpu_count = mp.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            # Limit workers based on available memory (assume 1GB per worker minimum)
            memory_workers = max(1, int(memory_gb / 2))
            self.max_workers = min(cpu_count, memory_workers)
        
        # Thread pools for I/O bound tasks
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Process pool for CPU bound tasks (created lazily)
        self._process_executor: Optional[ProcessPoolExecutor] = None
    
    @property
    def process_executor(self) -> ProcessPoolExecutor:
        """Get process executor (created lazily)."""
        if self._process_executor is None:
            self._process_executor = ProcessPoolExecutor(max_workers=self.max_workers)
        return self._process_executor
    
    async def process_concurrent(self, tasks: List[Callable], 
                                task_type: str = "mixed") -> List[Any]:
        """Process tasks concurrently with optimal strategy."""
        if not tasks:
            return []
        
        start_time = time.time()
        
        try:
            if self.config.processing_mode == ProcessingMode.SEQUENTIAL:
                results = [await self._run_task(task) for task in tasks]
            
            elif self.config.processing_mode == ProcessingMode.THREADED:
                results = await self._process_threaded(tasks)
            
            elif self.config.processing_mode == ProcessingMode.PROCESS_POOL:
                results = await self._process_with_pool(tasks)
            
            elif self.config.processing_mode == ProcessingMode.ASYNC_CONCURRENT:
                results = await self._process_async(tasks)
            
            else:  # HYBRID
                results = await self._process_hybrid(tasks, task_type)
            
            duration = time.time() - start_time
            self.monitor.record_metric(f"parallel_processing_{task_type}_duration", 
                                     duration, "timer")
            self.monitor.record_metric(f"parallel_processing_{task_type}_tasks", 
                                     len(tasks), "counter")
            
            return results
            
        except Exception as e:
            duration = time.time() - start_time
            self.monitor.record_metric(f"parallel_processing_{task_type}_error_duration", 
                                     duration, "timer")
            raise
    
    async def _run_task(self, task: Callable) -> Any:
        """Run a single task."""
        if asyncio.iscoroutinefunction(task):
            return await task()
        else:
            return task()
    
    async def _process_threaded(self, tasks: List[Callable]) -> List[Any]:
        """Process tasks using thread pool."""
        loop = asyncio.get_event_loop()
        futures = []
        
        for task in tasks:
            if asyncio.iscoroutinefunction(task):
                # Run coroutine in thread
                future = loop.run_in_executor(self.thread_executor, 
                                            lambda: asyncio.run(task()))
            else:
                future = loop.run_in_executor(self.thread_executor, task)
            futures.append(future)
        
        return await asyncio.gather(*futures)
    
    async def _process_with_pool(self, tasks: List[Callable]) -> List[Any]:
        """Process tasks using process pool."""
        loop = asyncio.get_event_loop()
        
        # Only use process pool for serializable tasks
        serializable_tasks = []
        for task in tasks:
            try:
                pickle.dumps(task)
                serializable_tasks.append(task)
            except (pickle.PicklingError, TypeError):
                # Fall back to thread pool for non-serializable tasks
                return await self._process_threaded(tasks)
        
        futures = [
            loop.run_in_executor(self.process_executor, task)
            for task in serializable_tasks
        ]
        
        return await asyncio.gather(*futures)
    
    async def _process_async(self, tasks: List[Callable]) -> List[Any]:
        """Process tasks using async concurrency."""
        async_tasks = []
        
        for task in tasks:
            if asyncio.iscoroutinefunction(task):
                async_tasks.append(task())
            else:
                # Wrap sync function in async
                async_tasks.append(asyncio.to_thread(task))
        
        return await asyncio.gather(*async_tasks)
    
    async def _process_hybrid(self, tasks: List[Callable], task_type: str) -> List[Any]:
        """Process tasks using hybrid strategy based on task characteristics."""
        # Categorize tasks
        cpu_bound = []
        io_bound = []
        async_tasks = []
        
        for task in tasks:
            if asyncio.iscoroutinefunction(task):
                async_tasks.append(task)
            elif task_type in ["parsing", "analysis", "computation"]:
                cpu_bound.append(task)
            else:
                io_bound.append(task)
        
        # Process each category optimally
        results = []
        
        if async_tasks:
            async_results = await self._process_async(async_tasks)
            results.extend(async_results)
        
        if cpu_bound:
            # Use process pool for CPU-bound tasks
            cpu_results = await self._process_with_pool(cpu_bound)
            results.extend(cpu_results)
        
        if io_bound:
            # Use thread pool for I/O-bound tasks
            io_results = await self._process_threaded(io_bound)
            results.extend(io_results)
        
        return results
    
    def shutdown(self):
        """Shutdown executors."""
        self.thread_executor.shutdown(wait=True)
        if self._process_executor:
            self._process_executor.shutdown(wait=True)


class MemoryOptimizer:
    """Memory optimization and management system."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.monitor = get_monitor()
        self._memory_threshold = config.memory_threshold
        
    def optimize_memory(self) -> Dict[str, Any]:
        """Perform memory optimization."""
        before_stats = self._get_memory_stats()
        
        if self.config.enable_memory_optimization:
            # Force garbage collection
            collected = gc.collect()
            
            # Check memory usage
            memory_percent = psutil.virtual_memory().percent / 100.0
            
            if memory_percent > self._memory_threshold:
                # Aggressive optimization
                self._aggressive_cleanup()
        
        after_stats = self._get_memory_stats()
        
        optimization_stats = {
            "before": before_stats,
            "after": after_stats,
            "freed_mb": before_stats["used_mb"] - after_stats["used_mb"],
            "gc_collected": gc.get_count()
        }
        
        self.monitor.record_metric("memory_optimization_freed_mb", 
                                 optimization_stats["freed_mb"], "gauge")
        
        return optimization_stats
    
    def _get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        
        return {
            "total_mb": memory.total / (1024**2),
            "available_mb": memory.available / (1024**2),
            "used_mb": memory.used / (1024**2),
            "percent": memory.percent,
            "process_mb": process.memory_info().rss / (1024**2)
        }
    
    def _aggressive_cleanup(self):
        """Perform aggressive memory cleanup."""
        # Force multiple GC cycles
        for _ in range(3):
            gc.collect()
        
        # Clear caches if available
        try:
            # Clear LRU caches
            for obj in gc.get_objects():
                if hasattr(obj, 'cache_clear') and callable(obj.cache_clear):
                    obj.cache_clear()
        except Exception:
            pass  # Ignore errors in cleanup


class PerformanceOptimizer:
    """Main performance optimization system."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.monitor = get_monitor()
        
        # Initialize subsystems
        self.cache = AdvancedCache(
            strategy=self.config.cache_strategy,
            max_size=self.config.max_cache_size,
            ttl=self.config.cache_ttl
        ) if self.config.enable_caching else None
        
        self.parallel_processor = ParallelProcessor(self.config)
        self.memory_optimizer = MemoryOptimizer(self.config)
        
        # Performance tracking
        self.metrics_history: deque = deque(maxlen=1000)
        self.operation_profiles: Dict[str, List[float]] = defaultdict(list)
    
    def cached(self, key_func: Optional[Callable] = None, ttl: Optional[float] = None):
        """Decorator for caching function results."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.cache:
                    return func(*args, **kwargs)
                
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    # Default key generation
                    key_parts = [func.__name__, str(args), str(sorted(kwargs.items()))]
                    cache_key = hashlib.sha256(
                        pickle.dumps(key_parts)
                    ).hexdigest()[:16]
                
                # Try to get from cache
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.cache.put(cache_key, result)
                return result
            
            return wrapper
        return decorator
    
    async def optimize_batch_operation(self, operations: List[Callable], 
                                     operation_type: str = "mixed") -> List[Any]:
        """Optimize batch operations with intelligent processing."""
        start_time = time.time()
        
        # Check memory before processing
        if self.config.enable_memory_optimization:
            memory_stats = self.memory_optimizer.optimize_memory()
            if memory_stats["after"]["percent"] > 90:
                # Memory is very high, process in smaller batches
                return await self._process_in_batches(operations, operation_type)
        
        # Process operations
        results = await self.parallel_processor.process_concurrent(
            operations, operation_type
        )
        
        # Record performance metrics
        duration = time.time() - start_time
        self._record_operation_performance(operation_type, duration, len(operations))
        
        return results
    
    async def _process_in_batches(self, operations: List[Callable], 
                                operation_type: str, batch_size: int = 10) -> List[Any]:
        """Process operations in smaller batches to manage memory."""
        results = []
        
        for i in range(0, len(operations), batch_size):
            batch = operations[i:i + batch_size]
            batch_results = await self.parallel_processor.process_concurrent(
                batch, f"{operation_type}_batch"
            )
            results.extend(batch_results)
            
            # Optimize memory between batches
            if self.config.enable_memory_optimization:
                self.memory_optimizer.optimize_memory()
        
        return results
    
    def _record_operation_performance(self, operation: str, duration: float, 
                                    task_count: int):
        """Record performance metrics for an operation."""
        memory_info = psutil.virtual_memory()
        process = psutil.Process()
        
        metrics = PerformanceMetrics(
            operation=operation,
            duration=duration,
            memory_usage=process.memory_info().rss / (1024**2),
            cpu_usage=process.cpu_percent(),
            parallel_tasks=task_count
        )
        
        if self.cache:
            cache_stats = self.cache.stats()
            metrics.cache_hits = cache_stats["hits"]
            metrics.cache_misses = cache_stats["misses"]
        
        self.metrics_history.append(metrics)
        self.operation_profiles[operation].append(duration)
        
        # Record in monitor
        self.monitor.record_metric(f"operation_{operation}_duration", duration, "timer")
        self.monitor.record_metric(f"operation_{operation}_tasks", task_count, "counter")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            "config": {
                "caching_enabled": self.config.enable_caching,
                "parallel_processing_enabled": self.config.enable_parallel_processing,
                "memory_optimization_enabled": self.config.enable_memory_optimization,
                "processing_mode": self.config.processing_mode.value,
                "max_workers": self.parallel_processor.max_workers
            },
            "cache_stats": self.cache.stats() if self.cache else None,
            "operation_profiles": {
                op: {
                    "count": len(times),
                    "avg_duration": sum(times) / len(times) if times else 0,
                    "min_duration": min(times) if times else 0,
                    "max_duration": max(times) if times else 0
                }
                for op, times in self.operation_profiles.items()
            },
            "recent_metrics": [
                {
                    "operation": m.operation,
                    "duration": m.duration,
                    "memory_usage": m.memory_usage,
                    "cache_hit_rate": m.cache_hits / (m.cache_hits + m.cache_misses) if (m.cache_hits + m.cache_misses) > 0 else 0
                }
                for m in list(self.metrics_history)[-10:]  # Last 10 operations
            ]
        }
        
        return summary
    
    def shutdown(self):
        """Shutdown optimization system."""
        self.parallel_processor.shutdown()
        if self.cache:
            self.cache.clear()


# Global performance optimizer
_global_optimizer: Optional[PerformanceOptimizer] = None


def get_optimizer(config: Optional[OptimizationConfig] = None) -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer(config)
    return _global_optimizer


def optimized(operation_type: str = "general", cache_key_func: Optional[Callable] = None):
    """Decorator for performance optimization."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            optimizer = get_optimizer()
            
            # Use caching if enabled
            if optimizer.cache and cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
                cached_result = optimizer.cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Execute function
            start_time = time.time()
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Cache result if enabled
            if optimizer.cache and cache_key_func:
                optimizer.cache.put(cache_key, result)
            
            # Record performance
            optimizer._record_operation_performance(operation_type, duration, 1)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            optimizer = get_optimizer()
            
            # Use caching if enabled
            if optimizer.cache and cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
                cached_result = optimizer.cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Execute function
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Cache result if enabled
            if optimizer.cache and cache_key_func:
                optimizer.cache.put(cache_key, result)
            
            # Record performance
            optimizer._record_operation_performance(operation_type, duration, 1)
            
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator