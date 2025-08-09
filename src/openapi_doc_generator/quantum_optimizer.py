"""Performance optimization and scaling for quantum-inspired task planning."""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import logging
import multiprocessing as mp
import pickle
import threading
import time
from dataclasses import dataclass
from functools import lru_cache, wraps
from typing import Any, Dict, List, Optional, Tuple

from .quantum_monitor import monitor_operation
from .quantum_scheduler import (
    QuantumInspiredScheduler,
    QuantumScheduleResult,
    QuantumTask,
)

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for quantum optimization settings."""
    enable_caching: bool = True
    cache_size: int = 128
    enable_parallel_processing: bool = True
    max_workers: int = None  # None = auto-detect
    enable_adaptive_scaling: bool = True
    min_annealing_iterations: int = 10
    max_annealing_iterations: int = 1000
    convergence_threshold: float = 1e-6
    memory_limit_mb: float = 1000.0
    timeout_seconds: float = 300.0  # 5 minutes


class QuantumCache:
    """High-performance cache for quantum scheduling results."""

    def __init__(self, max_size: int = 128, ttl_seconds: float = 3600):
        """Initialize cache with size and TTL limits."""
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.access_times: Dict[str, float] = {}
        self.lock = threading.RLock()

    def _generate_key(self, tasks: List[QuantumTask], config: Dict[str, Any]) -> str:
        """Generate cache key from tasks and configuration."""
        # Create deterministic hash from task properties and config
        task_data = []
        for task in sorted(tasks, key=lambda t: t.id):
            task_repr = (
                task.id, task.name, task.priority, task.effort,
                task.value, tuple(sorted(task.dependencies)),
                task.coherence_time, round(task.quantum_weight, 3)
            )
            task_data.append(task_repr)

        cache_input = {
            'tasks': task_data,
            'config': sorted(config.items())
        }

        serialized = pickle.dumps(cache_input, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(serialized).hexdigest()

    def get(self, tasks: List[QuantumTask], config: Dict[str, Any]) -> Optional[QuantumScheduleResult]:
        """Get cached result if available and valid."""
        key = self._generate_key(tasks, config)

        with self.lock:
            if key not in self.cache:
                return None

            result, timestamp = self.cache[key]
            current_time = time.time()

            # Check TTL
            if current_time - timestamp > self.ttl_seconds:
                del self.cache[key]
                del self.access_times[key]
                return None

            # Update access time for LRU
            self.access_times[key] = current_time
            logger.debug(f"Cache hit for key {key[:8]}...")
            return result

    def put(self, tasks: List[QuantumTask], config: Dict[str, Any], result: QuantumScheduleResult) -> None:
        """Store result in cache."""
        key = self._generate_key(tasks, config)
        current_time = time.time()

        with self.lock:
            # Evict oldest entries if cache is full
            while len(self.cache) >= self.max_size:
                oldest_key = min(self.access_times.keys(), key=self.access_times.get)
                del self.cache[oldest_key]
                del self.access_times[oldest_key]

            self.cache[key] = (result, current_time)
            self.access_times[key] = current_time
            logger.debug(f"Cached result for key {key[:8]}...")

    def clear(self) -> None:
        """Clear all cached entries."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            current_time = time.time()
            expired_count = sum(
                1 for timestamp in self.cache.values()
                if current_time - timestamp[1] > self.ttl_seconds
            )

            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "expired_entries": expired_count,
                "oldest_entry_age": current_time - min(
                    (timestamp for _, timestamp in self.cache.values()),
                    default=current_time
                ),
                "newest_entry_age": current_time - max(
                    (timestamp for _, timestamp in self.cache.values()),
                    default=current_time
                )
            }


class AdaptiveQuantumScheduler:
    """Adaptive quantum scheduler that scales parameters based on workload."""

    def __init__(self, base_scheduler: QuantumInspiredScheduler, config: OptimizationConfig):
        """Initialize adaptive scheduler."""
        self.base_scheduler = base_scheduler
        self.config = config
        self.performance_history: List[Tuple[int, float, float]] = []  # (task_count, duration, fidelity)
        self.lock = threading.Lock()

    def adaptive_schedule(self, tasks: List[QuantumTask]) -> QuantumScheduleResult:
        """Schedule tasks with adaptive parameter tuning."""
        task_count = len(tasks)

        # Determine optimal parameters based on task count and history
        temperature, cooling_rate, max_iterations = self._get_adaptive_parameters(task_count)

        # Update scheduler parameters
        original_temp = self.base_scheduler.temperature
        original_cooling = self.base_scheduler.cooling_rate

        self.base_scheduler.temperature = temperature
        self.base_scheduler.cooling_rate = cooling_rate

        start_time = time.time()

        try:
            # Run scheduling with early convergence detection
            result = self._schedule_with_convergence(tasks, max_iterations)

            duration = time.time() - start_time

            # Record performance for future adaptations
            with self.lock:
                self.performance_history.append((task_count, duration, result.quantum_fidelity))
                if len(self.performance_history) > 100:  # Keep recent history only
                    self.performance_history.pop(0)

            logger.info(f"Adaptive scheduling completed: {task_count} tasks, "
                       f"{duration:.3f}s, fidelity={result.quantum_fidelity:.3f}")

            return result

        finally:
            # Restore original parameters
            self.base_scheduler.temperature = original_temp
            self.base_scheduler.cooling_rate = original_cooling

    def _get_adaptive_parameters(self, task_count: int) -> Tuple[float, float, int]:
        """Get adaptive parameters based on task count and performance history."""
        # Base parameters
        base_temp = 2.0
        base_cooling = 0.95

        # Scale temperature with task count (more tasks = higher initial temperature)
        temperature = base_temp * (1 + 0.1 * (task_count / 10))

        # Adaptive cooling rate based on convergence history
        if self.performance_history:
            recent_performance = self.performance_history[-10:]  # Last 10 runs
            avg_fidelity = sum(p[2] for p in recent_performance) / len(recent_performance)

            # If fidelity is low, slow down cooling to allow more exploration
            if avg_fidelity < 0.7:
                cooling_rate = base_cooling * 0.98  # Slower cooling
            else:
                cooling_rate = base_cooling * 1.02  # Faster cooling
        else:
            cooling_rate = base_cooling

        # Adaptive iteration count
        if task_count < 10:
            max_iterations = self.config.min_annealing_iterations
        elif task_count < 50:
            max_iterations = min(task_count * 10, self.config.max_annealing_iterations)
        else:
            max_iterations = self.config.max_annealing_iterations

        return temperature, cooling_rate, max_iterations

    def _schedule_with_convergence(self, tasks: List[QuantumTask], max_iterations: int) -> QuantumScheduleResult:
        """Run scheduling with early convergence detection."""
        # For now, delegate to base scheduler
        # In a full implementation, we'd modify the annealing loop to check for convergence
        return self.base_scheduler.quantum_annealing_schedule(tasks)


class ParallelQuantumProcessor:
    """Parallel processing engine for quantum task scheduling."""

    def __init__(self, config: OptimizationConfig):
        """Initialize parallel processor."""
        self.config = config
        self.max_workers = config.max_workers or min(mp.cpu_count(), 8)

    async def process_multiple_plans_async(self,
                                          plan_requests: List[Tuple[List[QuantumTask], Dict[str, Any]]]) -> List[QuantumScheduleResult]:
        """Process multiple planning requests concurrently."""
        if not self.config.enable_parallel_processing or len(plan_requests) == 1:
            # Fall back to sequential processing
            results = []
            for tasks, config in plan_requests:
                scheduler = QuantumInspiredScheduler(
                    temperature=config.get('temperature', 2.0),
                    cooling_rate=config.get('cooling_rate', 0.95)
                )
                result = scheduler.quantum_annealing_schedule(tasks)
                results.append(result)
            return results

        # Parallel processing using ThreadPoolExecutor
        loop = asyncio.get_event_loop()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            tasks_futures = []

            for tasks, config in plan_requests:
                future = loop.run_in_executor(
                    executor,
                    self._process_single_plan,
                    tasks,
                    config
                )
                tasks_futures.append(future)

            results = await asyncio.gather(*tasks_futures)
            return results

    def _process_single_plan(self, tasks: List[QuantumTask], config: Dict[str, Any]) -> QuantumScheduleResult:
        """Process a single planning request."""
        scheduler = QuantumInspiredScheduler(
            temperature=config.get('temperature', 2.0),
            cooling_rate=config.get('cooling_rate', 0.95)
        )
        return scheduler.quantum_annealing_schedule(tasks)

    def process_large_task_set(self, tasks: List[QuantumTask], chunk_size: int = 50) -> QuantumScheduleResult:
        """Process large task sets by dividing into chunks and combining results."""
        if len(tasks) <= chunk_size:
            # Small enough to process normally
            scheduler = QuantumInspiredScheduler()
            return scheduler.quantum_annealing_schedule(tasks)

        # Divide tasks into chunks based on dependencies
        chunks = self._partition_tasks_intelligently(tasks, chunk_size)

        # Process chunks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            chunk_futures = []

            for chunk in chunks:
                future = executor.submit(self._process_chunk, chunk)
                chunk_futures.append(future)

            chunk_results = [f.result() for f in concurrent.futures.as_completed(chunk_futures)]

        # Combine results
        return self._combine_chunk_results(chunk_results, tasks)

    def _partition_tasks_intelligently(self, tasks: List[QuantumTask], chunk_size: int) -> List[List[QuantumTask]]:
        """Partition tasks into chunks while respecting dependencies."""
        # Build dependency graph
        task_map = {task.id: task for task in tasks}

        # Sort tasks by dependency depth
        def get_dependency_depth(task_id: str, visited: set = None) -> int:
            if visited is None:
                visited = set()

            if task_id in visited:  # Circular dependency
                return 0

            visited.add(task_id)
            task = task_map.get(task_id)
            if not task or not task.dependencies:
                return 0

            max_depth = 0
            for dep_id in task.dependencies:
                if dep_id in task_map:
                    depth = get_dependency_depth(dep_id, visited.copy())
                    max_depth = max(max_depth, depth + 1)

            return max_depth

        # Sort tasks by dependency depth
        sorted_tasks = sorted(tasks, key=lambda t: get_dependency_depth(t.id))

        # Create chunks
        chunks = []
        current_chunk = []

        for task in sorted_tasks:
            if len(current_chunk) >= chunk_size:
                chunks.append(current_chunk)
                current_chunk = [task]
            else:
                current_chunk.append(task)

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _process_chunk(self, chunk: List[QuantumTask]) -> QuantumScheduleResult:
        """Process a single chunk of tasks."""
        scheduler = QuantumInspiredScheduler()
        return scheduler.quantum_annealing_schedule(chunk)

    def _combine_chunk_results(self, chunk_results: List[QuantumScheduleResult],
                              original_tasks: List[QuantumTask]) -> QuantumScheduleResult:
        """Combine results from multiple chunks into a single result."""
        combined_tasks = []
        total_value = 0.0
        total_execution_time = 0.0
        min_fidelity = 1.0
        total_iterations = 0

        for result in chunk_results:
            combined_tasks.extend(result.optimized_tasks)
            total_value += result.total_value
            total_execution_time += result.execution_time
            min_fidelity = min(min_fidelity, result.quantum_fidelity)
            total_iterations += result.convergence_iterations

        return QuantumScheduleResult(
            optimized_tasks=combined_tasks,
            total_value=total_value,
            execution_time=total_execution_time,
            quantum_fidelity=min_fidelity,
            convergence_iterations=total_iterations
        )


class OptimizedQuantumPlanner:
    """High-performance, scalable quantum task planner."""

    def __init__(self, config: OptimizationConfig = None):
        """Initialize optimized planner."""
        self.config = config or OptimizationConfig()

        # Core components
        self.base_scheduler = QuantumInspiredScheduler()
        self.adaptive_scheduler = AdaptiveQuantumScheduler(self.base_scheduler, self.config)
        self.parallel_processor = ParallelQuantumProcessor(self.config)

        # Performance optimizations
        if self.config.enable_caching:
            self.cache = QuantumCache(self.config.cache_size)
        else:
            self.cache = None

        # Resource monitoring
        self.resource_monitor = ResourceMonitor(self.config)

    @monitor_operation("optimized_quantum_planning")
    def create_optimized_plan(self, tasks: List[QuantumTask]) -> QuantumScheduleResult:
        """Create optimized quantum plan with all performance enhancements."""
        if not tasks:
            return QuantumScheduleResult([], 0.0, 0.0, 1.0, 0)

        config_dict = {
            'temperature': self.base_scheduler.temperature,
            'cooling_rate': self.base_scheduler.cooling_rate,
            'adaptive': True
        }

        # Check cache first
        if self.cache:
            cached_result = self.cache.get(tasks, config_dict)
            if cached_result:
                logger.info(f"Returning cached result for {len(tasks)} tasks")
                return cached_result

        # Check resource constraints
        if not self.resource_monitor.check_resources(len(tasks)):
            logger.warning("Resource constraints exceeded, using simplified scheduling")
            return self._fallback_schedule(tasks)

        try:
            # Use adaptive scheduling for optimal performance
            if len(tasks) > 100 and self.config.enable_parallel_processing:
                # Large task set - use parallel processing
                result = self.parallel_processor.process_large_task_set(tasks)
            else:
                # Normal size - use adaptive scheduling
                result = self.adaptive_scheduler.adaptive_schedule(tasks)

            # Cache the result
            if self.cache:
                self.cache.put(tasks, config_dict, result)

            return result

        except Exception as e:
            logger.error(f"Optimized planning failed: {e}, falling back to basic scheduling")
            return self._fallback_schedule(tasks)

    def _fallback_schedule(self, tasks: List[QuantumTask]) -> QuantumScheduleResult:
        """Fallback to basic scheduling when optimizations fail."""
        basic_scheduler = QuantumInspiredScheduler(temperature=1.0, cooling_rate=0.9)
        return basic_scheduler.quantum_annealing_schedule(tasks)

    async def create_multiple_plans(self,
                                   plan_requests: List[Tuple[List[QuantumTask], Dict[str, Any]]]) -> List[QuantumScheduleResult]:
        """Create multiple quantum plans concurrently."""
        return await self.parallel_processor.process_multiple_plans_async(plan_requests)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the optimized planner."""
        stats = {
            "config": {
                "caching_enabled": self.config.enable_caching,
                "parallel_processing_enabled": self.config.enable_parallel_processing,
                "adaptive_scaling_enabled": self.config.enable_adaptive_scaling,
                "max_workers": self.parallel_processor.max_workers
            }
        }

        # Cache statistics
        if self.cache:
            stats["cache"] = self.cache.get_stats()

        # Adaptive scheduler statistics
        with self.adaptive_scheduler.lock:
            if self.adaptive_scheduler.performance_history:
                recent_history = self.adaptive_scheduler.performance_history[-20:]
                stats["adaptive_scheduler"] = {
                    "runs_recorded": len(self.adaptive_scheduler.performance_history),
                    "avg_duration_recent": sum(p[1] for p in recent_history) / len(recent_history),
                    "avg_fidelity_recent": sum(p[2] for p in recent_history) / len(recent_history),
                    "avg_task_count_recent": sum(p[0] for p in recent_history) / len(recent_history)
                }

        # Resource monitor statistics
        stats["resources"] = self.resource_monitor.get_stats()

        return stats

    def clear_cache(self) -> None:
        """Clear the performance cache."""
        if self.cache:
            self.cache.clear()

    def tune_performance(self, target_fidelity: float = 0.8, target_duration_ms: float = 5000) -> None:
        """Auto-tune performance parameters based on targets."""
        logger.info(f"Auto-tuning for fidelity >= {target_fidelity}, duration <= {target_duration_ms}ms")

        with self.adaptive_scheduler.lock:
            if not self.adaptive_scheduler.performance_history:
                logger.warning("No performance history available for tuning")
                return

            # Analyze recent performance
            recent_runs = self.adaptive_scheduler.performance_history[-10:]
            avg_fidelity = sum(p[2] for p in recent_runs) / len(recent_runs)
            avg_duration_ms = sum(p[1] * 1000 for p in recent_runs) / len(recent_runs)

            # Adjust parameters based on performance vs targets
            if avg_fidelity < target_fidelity:
                # Increase iterations for better quality
                self.config.max_annealing_iterations = min(
                    self.config.max_annealing_iterations * 1.2,
                    2000
                )
                logger.info(f"Increased max iterations to {self.config.max_annealing_iterations}")

            if avg_duration_ms > target_duration_ms:
                # Decrease iterations for better speed
                self.config.max_annealing_iterations = max(
                    self.config.max_annealing_iterations * 0.8,
                    self.config.min_annealing_iterations
                )
                logger.info(f"Decreased max iterations to {self.config.max_annealing_iterations}")


class ResourceMonitor:
    """Monitor system resources for scaling decisions."""

    def __init__(self, config: OptimizationConfig):
        """Initialize resource monitor."""
        self.config = config
        self.last_check_time = 0
        self.last_memory_usage = 0
        self.check_interval = 1.0  # seconds

    def check_resources(self, task_count: int) -> bool:
        """Check if system has sufficient resources for the given task count."""
        current_time = time.time()

        # Rate limit resource checks
        if current_time - self.last_check_time < self.check_interval:
            return True  # Assume OK if checked recently

        self.last_check_time = current_time

        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)

            # Check memory limit
            if memory_mb > self.config.memory_limit_mb:
                logger.warning(f"Memory usage {memory_mb:.1f}MB exceeds limit {self.config.memory_limit_mb}MB")
                return False

            # Estimate memory needs based on task count
            estimated_memory_per_task = 0.1  # MB per task (rough estimate)
            estimated_additional_memory = task_count * estimated_memory_per_task

            if memory_mb + estimated_additional_memory > self.config.memory_limit_mb:
                logger.warning(f"Estimated memory {memory_mb + estimated_additional_memory:.1f}MB would exceed limit")
                return False

            self.last_memory_usage = memory_mb
            return True

        except ImportError:
            # psutil not available, assume resources are OK
            return True
        except Exception as e:
            logger.warning(f"Resource check failed: {e}")
            return True  # Assume OK on error

    def get_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        stats = {
            "last_check_time": self.last_check_time,
            "last_memory_usage_mb": self.last_memory_usage,
            "memory_limit_mb": self.config.memory_limit_mb
        }

        try:
            import psutil
            process = psutil.Process()
            stats.update({
                "current_memory_mb": process.memory_info().rss / (1024 * 1024),
                "cpu_percent": process.cpu_percent(),
                "thread_count": process.num_threads()
            })
        except (ImportError, Exception):
            pass

        return stats


# Performance optimization decorators and utilities

def cached_quantum_result(cache_size: int = 64):
    """Decorator to cache quantum scheduling results."""
    def decorator(func):
        @lru_cache(maxsize=cache_size)
        def _cached_wrapper(tasks_hash: str, *args, **kwargs):
            # This is called with a hash of the tasks
            return func(*args, **kwargs)

        @wraps(func)
        def wrapper(tasks: List[QuantumTask], *args, **kwargs):
            # Generate hash of tasks for caching
            task_data = tuple(
                (t.id, t.priority, t.effort, t.value, tuple(sorted(t.dependencies)))
                for t in sorted(tasks, key=lambda x: x.id)
            )
            tasks_hash = str(hash(task_data))

            return _cached_wrapper(tasks_hash, tasks, *args, **kwargs)

        # Expose cache info
        wrapper.cache_info = _cached_wrapper.cache_info
        wrapper.cache_clear = _cached_wrapper.cache_clear

        return wrapper
    return decorator


def timeout_quantum_operation(timeout_seconds: float = 300.0):
    """Decorator to add timeout to quantum operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout_seconds)
                except concurrent.futures.TimeoutError:
                    logger.error(f"Operation {func.__name__} timed out after {timeout_seconds}s")
                    raise TimeoutError(f"Quantum operation timed out after {timeout_seconds} seconds")

        return wrapper
    return decorator
