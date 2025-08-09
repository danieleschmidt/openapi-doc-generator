"""Advanced scaling and performance optimization for quantum task planning."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import multiprocessing as mp
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .quantum_recovery import RecoveryStrategy, get_recovery_manager
from .quantum_scheduler import QuantumScheduleResult, QuantumTask

logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Scaling strategies for quantum operations."""
    THREAD_BASED = "thread_based"
    PROCESS_BASED = "process_based"
    ASYNC_BASED = "async_based"
    HYBRID = "hybrid"
    AUTO_ADAPTIVE = "auto_adaptive"


@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions."""
    cpu_usage: float
    memory_usage: float
    task_queue_length: int
    average_processing_time: float
    concurrent_operations: int
    error_rate: float


@dataclass
class ScalingConfig:
    """Configuration for auto-scaling."""
    min_workers: int = 2
    max_workers: int = 16
    cpu_threshold_scale_up: float = 80.0
    cpu_threshold_scale_down: float = 30.0
    memory_threshold: float = 85.0
    queue_length_threshold: int = 100
    scale_up_cooldown: float = 60.0
    scale_down_cooldown: float = 300.0


class QuantumTaskScaler:
    """Advanced scaling manager for quantum task processing."""

    def __init__(self, config: Optional[ScalingConfig] = None):
        """Initialize quantum task scaler."""
        self.config = config or ScalingConfig()
        self.strategy = ScalingStrategy.AUTO_ADAPTIVE
        self.current_workers = self.config.min_workers

        # Worker pools
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.process_pool: Optional[ProcessPoolExecutor] = None
        self.task_queue = queue.Queue()

        # Metrics and monitoring
        self.metrics_history: List[ScalingMetrics] = []
        self.last_scale_up = 0.0
        self.last_scale_down = 0.0
        self.scaling_decisions: List[Dict[str, Any]] = []

        # Performance caches
        self.result_cache: Dict[str, Any] = {}
        self.computation_cache: Dict[str, Any] = {}
        self.cache_stats = {"hits": 0, "misses": 0, "evictions": 0}

        # Recovery manager integration
        self.recovery_manager = get_recovery_manager()

        # Initialize worker pools
        self._initialize_pools()

        # Monitoring thread
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

    def _initialize_pools(self):
        """Initialize worker pools based on current strategy."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)

        # Always maintain thread pool for I/O operations
        self.thread_pool = ThreadPoolExecutor(max_workers=self.current_workers)

        # Process pool for CPU-intensive operations
        if self.strategy in [ScalingStrategy.PROCESS_BASED, ScalingStrategy.HYBRID, ScalingStrategy.AUTO_ADAPTIVE]:
            cpu_workers = min(self.current_workers, mp.cpu_count())
            self.process_pool = ProcessPoolExecutor(max_workers=cpu_workers)

        logger.info(f"Initialized pools: {self.current_workers} thread workers, "
                   f"{cpu_workers if self.process_pool else 0} process workers")

    async def process_tasks_concurrent(self,
                                     tasks: List[QuantumTask],
                                     operation: Callable,
                                     batch_size: Optional[int] = None) -> List[Any]:
        """Process tasks concurrently using optimal scaling strategy."""
        if not tasks:
            return []

        batch_size = batch_size or min(len(tasks), self.current_workers * 2)
        batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]

        results = []

        if self.strategy == ScalingStrategy.ASYNC_BASED:
            results = await self._process_async_batches(batches, operation)
        elif self.strategy == ScalingStrategy.PROCESS_BASED:
            results = await self._process_with_processes(batches, operation)
        elif self.strategy == ScalingStrategy.THREAD_BASED:
            results = await self._process_with_threads(batches, operation)
        elif self.strategy == ScalingStrategy.HYBRID:
            results = await self._process_hybrid(batches, operation)
        else:  # AUTO_ADAPTIVE
            results = await self._process_adaptive(batches, operation)

        # Flatten results
        flat_results = []
        for batch_result in results:
            if isinstance(batch_result, list):
                flat_results.extend(batch_result)
            else:
                flat_results.append(batch_result)

        return flat_results

    async def _process_async_batches(self, batches: List[List[QuantumTask]], operation: Callable) -> List[Any]:
        """Process batches using asyncio."""
        semaphore = asyncio.Semaphore(self.current_workers)

        async def process_batch_async(batch: List[QuantumTask]) -> Any:
            async with semaphore:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(self.thread_pool, operation, batch)

        tasks = [process_batch_async(batch) for batch in batches]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_with_processes(self, batches: List[List[QuantumTask]], operation: Callable) -> List[Any]:
        """Process batches using process pool."""
        loop = asyncio.get_event_loop()

        futures = []
        for batch in batches:
            future = loop.run_in_executor(self.process_pool, operation, batch)
            futures.append(future)

        return await asyncio.gather(*futures, return_exceptions=True)

    async def _process_with_threads(self, batches: List[List[QuantumTask]], operation: Callable) -> List[Any]:
        """Process batches using thread pool."""
        loop = asyncio.get_event_loop()

        futures = []
        for batch in batches:
            future = loop.run_in_executor(self.thread_pool, operation, batch)
            futures.append(future)

        return await asyncio.gather(*futures, return_exceptions=True)

    async def _process_hybrid(self, batches: List[List[QuantumTask]], operation: Callable) -> List[Any]:
        """Process using hybrid thread/process strategy."""
        # Use processes for CPU-intensive batches, threads for I/O
        cpu_intensive_threshold = 10  # tasks per batch

        cpu_batches = [b for b in batches if len(b) >= cpu_intensive_threshold]
        io_batches = [b for b in batches if len(b) < cpu_intensive_threshold]

        results = []

        # Process CPU-intensive batches with processes
        if cpu_batches and self.process_pool:
            cpu_results = await self._process_with_processes(cpu_batches, operation)
            results.extend(cpu_results)

        # Process I/O batches with threads
        if io_batches:
            io_results = await self._process_with_threads(io_batches, operation)
            results.extend(io_results)

        return results

    async def _process_adaptive(self, batches: List[List[QuantumTask]], operation: Callable) -> List[Any]:
        """Adaptively choose processing strategy based on current metrics."""
        current_metrics = self._get_current_metrics()

        # Decision logic based on current system state
        if current_metrics.cpu_usage > 80 and self.process_pool:
            # High CPU usage, prefer process-based parallelism
            return await self._process_with_processes(batches, operation)
        elif current_metrics.memory_usage > 70:
            # High memory usage, prefer thread-based approach
            return await self._process_with_threads(batches, operation)
        elif len(batches) > self.current_workers * 2:
            # Many small batches, use async approach
            return await self._process_async_batches(batches, operation)
        else:
            # Default to hybrid approach
            return await self._process_hybrid(batches, operation)

    def optimize_quantum_annealing(self,
                                 planner_function: Callable,
                                 tasks: List[QuantumTask],
                                 num_iterations: int = 10) -> QuantumScheduleResult:
        """Optimize quantum annealing with parallel exploration."""
        cache_key = self._generate_cache_key("annealing", tasks, num_iterations)

        # Check cache first
        if cache_key in self.result_cache:
            self.cache_stats["hits"] += 1
            logger.debug("Cache hit for quantum annealing optimization")
            return self.result_cache[cache_key]

        self.cache_stats["misses"] += 1

        # Run multiple annealing iterations in parallel
        def run_annealing_iteration(iteration_seed: int) -> QuantumScheduleResult:
            # Modify planner with different random seed for diversity
            import random
            random.seed(iteration_seed)
            return planner_function(tasks)

        # Execute parallel iterations
        with self.recovery_manager.resilient_execution(
            "quantum_annealing_optimization",
            RecoveryStrategy.RETRY
        ) as context:

            futures = []
            with ThreadPoolExecutor(max_workers=min(num_iterations, self.current_workers)) as executor:
                for i in range(num_iterations):
                    future = executor.submit(run_annealing_iteration, i * 42)
                    futures.append(future)

            # Collect results
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=30.0)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Annealing iteration failed: {str(e)}")

            if not results:
                raise RuntimeError("All annealing iterations failed")

            # Select best result based on quantum fidelity and total value
            best_result = max(results, key=lambda r: r.quantum_fidelity * r.total_value)

            # Cache the result
            self._cache_result(cache_key, best_result)

            return best_result

    def parallel_resource_allocation(self,
                                   allocator_function: Callable,
                                   task_groups: List[List[QuantumTask]]) -> Dict[str, int]:
        """Optimize resource allocation using parallel processing."""
        cache_key = self._generate_cache_key("resource_allocation", task_groups)

        if cache_key in self.computation_cache:
            self.cache_stats["hits"] += 1
            return self.computation_cache[cache_key]

        self.cache_stats["misses"] += 1

        with self.recovery_manager.resilient_execution(
            "resource_allocation",
            RecoveryStrategy.GRACEFUL_DEGRADATION
        ) as context:

            # Process each group in parallel
            allocation_results = {}

            if self.process_pool and len(task_groups) > 1:
                # Use process pool for parallel allocation
                futures = {}
                with ProcessPoolExecutor(max_workers=min(len(task_groups), self.current_workers)) as executor:
                    for i, group in enumerate(task_groups):
                        future = executor.submit(allocator_function, group)
                        futures[future] = i

                for future in concurrent.futures.as_completed(futures):
                    try:
                        group_index = futures[future]
                        allocation = future.result(timeout=15.0)
                        allocation_results.update(allocation)
                    except Exception as e:
                        logger.warning(f"Resource allocation failed for group {futures[future]}: {str(e)}")
            else:
                # Fallback to sequential processing
                for group in task_groups:
                    try:
                        allocation = allocator_function(group)
                        allocation_results.update(allocation)
                    except Exception as e:
                        logger.warning(f"Resource allocation failed: {str(e)}")

            self._cache_result(cache_key, allocation_results, cache_type="computation")
            return allocation_results

    def auto_scale_workers(self):
        """Automatically scale worker pools based on current metrics."""
        current_time = time.time()
        metrics = self._get_current_metrics()

        should_scale_up = (
            metrics.cpu_usage > self.config.cpu_threshold_scale_up or
            metrics.task_queue_length > self.config.queue_length_threshold
        ) and (current_time - self.last_scale_up) > self.config.scale_up_cooldown

        should_scale_down = (
            metrics.cpu_usage < self.config.cpu_threshold_scale_down and
            metrics.task_queue_length < 10
        ) and (current_time - self.last_scale_down) > self.config.scale_down_cooldown

        if should_scale_up and self.current_workers < self.config.max_workers:
            new_workers = min(self.current_workers + 2, self.config.max_workers)
            self._scale_to_workers(new_workers)
            self.last_scale_up = current_time

            self.scaling_decisions.append({
                "timestamp": current_time,
                "action": "scale_up",
                "old_workers": self.current_workers,
                "new_workers": new_workers,
                "reason": f"CPU: {metrics.cpu_usage}%, Queue: {metrics.task_queue_length}"
            })

        elif should_scale_down and self.current_workers > self.config.min_workers:
            new_workers = max(self.current_workers - 1, self.config.min_workers)
            self._scale_to_workers(new_workers)
            self.last_scale_down = current_time

            self.scaling_decisions.append({
                "timestamp": current_time,
                "action": "scale_down",
                "old_workers": self.current_workers,
                "new_workers": new_workers,
                "reason": f"CPU: {metrics.cpu_usage}%, Low utilization"
            })

    def _scale_to_workers(self, target_workers: int):
        """Scale worker pools to target size."""
        if target_workers == self.current_workers:
            return

        logger.info(f"Scaling workers from {self.current_workers} to {target_workers}")
        self.current_workers = target_workers
        self._initialize_pools()

    def _get_current_metrics(self) -> ScalingMetrics:
        """Get current system metrics for scaling decisions."""
        import psutil

        return ScalingMetrics(
            cpu_usage=psutil.cpu_percent(interval=0.1),
            memory_usage=psutil.virtual_memory().percent,
            task_queue_length=self.task_queue.qsize(),
            average_processing_time=self._calculate_average_processing_time(),
            concurrent_operations=self._count_concurrent_operations(),
            error_rate=self._calculate_error_rate()
        )

    def _calculate_average_processing_time(self) -> float:
        """Calculate average processing time from recent metrics."""
        if not self.metrics_history:
            return 1.0

        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
        if not recent_metrics:
            return 1.0

        return sum(m.average_processing_time for m in recent_metrics) / len(recent_metrics)

    def _count_concurrent_operations(self) -> int:
        """Count currently running concurrent operations."""
        active_threads = threading.active_count() - 1  # Exclude monitoring thread
        return active_threads

    def _calculate_error_rate(self) -> float:
        """Calculate recent error rate."""
        recovery_stats = self.recovery_manager.get_recovery_statistics()

        total_operations = sum(
            stats.get("total_attempts", 0)
            for stats in recovery_stats.get("recovery_stats", {}).values()
        )

        failed_operations = sum(
            stats.get("failed_recoveries", 0)
            for stats in recovery_stats.get("recovery_stats", {}).values()
        )

        if total_operations == 0:
            return 0.0

        return (failed_operations / total_operations) * 100.0

    def _generate_cache_key(self, operation: str, *args) -> str:
        """Generate cache key for operation and arguments."""
        import hashlib

        # Create a deterministic key from operation and args
        key_data = f"{operation}:{str(args)}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _cache_result(self, key: str, result: Any, cache_type: str = "result"):
        """Cache computation result with LRU eviction."""
        cache = self.result_cache if cache_type == "result" else self.computation_cache
        max_cache_size = 1000 if cache_type == "result" else 500

        # LRU eviction if cache is full
        if len(cache) >= max_cache_size:
            # Remove oldest entry (simple FIFO for now)
            oldest_key = next(iter(cache))
            del cache[oldest_key]
            self.cache_stats["evictions"] += 1

        cache[key] = result

    def _monitoring_loop(self):
        """Background monitoring loop for metrics collection."""
        while self.monitoring_active:
            try:
                metrics = self._get_current_metrics()
                self.metrics_history.append(metrics)

                # Keep only recent history (last hour at 10s intervals)
                if len(self.metrics_history) > 360:
                    self.metrics_history.pop(0)

                # Trigger auto-scaling check
                self.auto_scale_workers()

                time.sleep(10)  # Monitor every 10 seconds

            except Exception as e:
                logger.error(f"Monitoring loop error: {str(e)}")
                time.sleep(30)  # Back off on errors

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        current_metrics = self._get_current_metrics()

        return {
            "scaling": {
                "strategy": self.strategy.value,
                "current_workers": self.current_workers,
                "min_workers": self.config.min_workers,
                "max_workers": self.config.max_workers,
                "scaling_decisions": len(self.scaling_decisions)
            },
            "performance": {
                "cpu_usage": current_metrics.cpu_usage,
                "memory_usage": current_metrics.memory_usage,
                "queue_length": current_metrics.task_queue_length,
                "error_rate": current_metrics.error_rate,
                "concurrent_ops": current_metrics.concurrent_operations
            },
            "caching": {
                "cache_hit_rate": self.cache_stats["hits"] / max(1, self.cache_stats["hits"] + self.cache_stats["misses"]) * 100,
                "total_hits": self.cache_stats["hits"],
                "total_misses": self.cache_stats["misses"],
                "total_evictions": self.cache_stats["evictions"],
                "result_cache_size": len(self.result_cache),
                "computation_cache_size": len(self.computation_cache)
            },
            "history": {
                "metrics_collected": len(self.metrics_history),
                "recent_scaling_decisions": self.scaling_decisions[-5:] if self.scaling_decisions else []
            }
        }

    def shutdown(self):
        """Gracefully shutdown the scaler."""
        self.monitoring_active = False

        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)

        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)

        if self.process_pool:
            self.process_pool.shutdown(wait=True)

        logger.info("Quantum task scaler shutdown complete")


# Global scaler instance
_quantum_scaler = None


def get_quantum_scaler(config: Optional[ScalingConfig] = None) -> QuantumTaskScaler:
    """Get global quantum task scaler instance."""
    global _quantum_scaler
    if _quantum_scaler is None:
        _quantum_scaler = QuantumTaskScaler(config)
    return _quantum_scaler
