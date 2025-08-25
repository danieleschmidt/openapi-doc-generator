"""
Parallel Processing Engine for High-Performance Documentation Generation

Provides intelligent parallel processing of documentation tasks with dynamic
resource allocation and load balancing.
"""

import multiprocessing
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional, TypeVar

from .advanced_caching import get_cache
from .circuit_breaker import CircuitBreakerConfig, get_circuit_breaker

T = TypeVar('T')


@dataclass
class ProcessingTask:
    """Task for parallel processing."""
    id: str
    func: Callable
    args: tuple
    kwargs: dict
    priority: int = 0
    timeout: float = 30.0
    retry_count: int = 0
    max_retries: int = 3

    def __post_init__(self):
        self.created_at = time.time()


@dataclass
class ProcessingResult:
    """Result of parallel processing task."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    worker_id: str = ""

    def __post_init__(self):
        self.completed_at = time.time()


@dataclass
class WorkerStats:
    """Statistics for individual worker."""
    worker_id: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    avg_execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0

    def update_stats(self, execution_time: float, success: bool):
        """Update worker statistics."""
        if success:
            self.tasks_completed += 1
        else:
            self.tasks_failed += 1

        self.total_execution_time += execution_time
        total_tasks = self.tasks_completed + self.tasks_failed
        self.avg_execution_time = self.total_execution_time / max(total_tasks, 1)


class AdaptiveWorkerPool:
    """Self-optimizing worker pool with dynamic scaling."""

    def __init__(self,
                 min_workers: int = 2,
                 max_workers: Optional[int] = None,
                 queue_size: int = 1000):
        self.min_workers = min_workers
        self.max_workers = max_workers or min(32, (multiprocessing.cpu_count() or 1) + 4)
        self.queue_size = queue_size

        self.task_queue: Queue[ProcessingTask] = Queue(maxsize=queue_size)
        self.result_queue: Queue[ProcessingResult] = Queue()

        self.workers: List[threading.Thread] = []
        self.worker_stats: Dict[str, WorkerStats] = {}

        self._shutdown = threading.Event()
        self._lock = threading.RLock()

        # Performance monitoring
        self.total_tasks_processed = 0
        self.total_tasks_failed = 0
        self.avg_queue_size = 0.0

        # Start with minimum workers
        self._start_workers(self.min_workers)

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_performance, daemon=True)
        self.monitor_thread.start()

    def _start_workers(self, count: int) -> None:
        """Start additional worker threads."""
        with self._lock:
            for i in range(count):
                worker_id = f"worker_{len(self.workers) + 1}"
                worker = threading.Thread(
                    target=self._worker_loop,
                    args=(worker_id,),
                    daemon=True,
                    name=worker_id
                )
                worker.start()
                self.workers.append(worker)
                self.worker_stats[worker_id] = WorkerStats(worker_id=worker_id)

    def _worker_loop(self, worker_id: str) -> None:
        """Main worker loop."""
        cache = get_cache()
        circuit = get_circuit_breaker(f"worker_{worker_id}",
                                     CircuitBreakerConfig(failure_threshold=5, timeout=60.0))

        while not self._shutdown.is_set():
            try:
                # Get task with timeout
                try:
                    task = self.task_queue.get(timeout=1.0)
                except Empty:
                    continue

                start_time = time.time()
                result = ProcessingResult(task_id=task.id, success=False, worker_id=worker_id)

                try:
                    # Execute task with circuit breaker protection
                    with circuit.protect(f"task_{task.id}"):
                        task_result = task.func(*task.args, **task.kwargs)

                    result.success = True
                    result.result = task_result

                except Exception as e:
                    result.error = e

                    # Retry logic
                    if task.retry_count < task.max_retries:
                        task.retry_count += 1
                        self.task_queue.put(task)
                        continue

                result.execution_time = time.time() - start_time

                # Update worker stats
                with self._lock:
                    if worker_id in self.worker_stats:
                        self.worker_stats[worker_id].update_stats(
                            result.execution_time, result.success
                        )

                # Submit result
                self.result_queue.put(result)
                self.task_queue.task_done()

            except Exception as e:
                # Log worker error
                import logging
                logger = logging.getLogger("parallel_processor")
                logger.error(f"Worker {worker_id} error: {e}")

    def _monitor_performance(self) -> None:
        """Monitor and adjust worker pool size based on performance."""
        while not self._shutdown.is_set():
            try:
                time.sleep(10)  # Check every 10 seconds

                with self._lock:
                    queue_size = self.task_queue.qsize()
                    active_workers = len([w for w in self.workers if w.is_alive()])

                    # Scale up if queue is growing
                    if (queue_size > active_workers * 2 and
                        active_workers < self.max_workers):
                        self._start_workers(min(2, self.max_workers - active_workers))

                    # Scale down if queue is consistently small
                    elif (queue_size < active_workers // 2 and
                          active_workers > self.min_workers):
                        # This is simplified - in production you'd gracefully stop workers
                        pass

                    # Update performance metrics
                    self.avg_queue_size = (self.avg_queue_size * 0.9) + (queue_size * 0.1)

            except Exception as e:
                import logging
                logger = logging.getLogger("parallel_processor")
                logger.error(f"Performance monitor error: {e}")

    def submit_task(self,
                   task_id: str,
                   func: Callable,
                   *args,
                   priority: int = 0,
                   timeout: float = 30.0,
                   max_retries: int = 3,
                   **kwargs) -> bool:
        """Submit task for parallel processing."""
        task = ProcessingTask(
            id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
            max_retries=max_retries
        )

        try:
            self.task_queue.put(task, timeout=1.0)
            return True
        except Exception:
            return False

    def get_result(self, timeout: float = None) -> Optional[ProcessingResult]:
        """Get processing result."""
        try:
            return self.result_queue.get(timeout=timeout)
        except Empty:
            return None

    def wait_for_completion(self, timeout: float = None) -> bool:
        """Wait for all tasks to complete."""
        try:
            # This is a simplified implementation
            # In production, you'd want more sophisticated completion tracking
            deadline = time.time() + (timeout or float('inf'))
            while time.time() < deadline:
                if self.task_queue.empty():
                    return True
                time.sleep(0.1)
            return False
        except Exception:
            return False

    def shutdown(self, timeout: float = 30.0) -> None:
        """Shutdown worker pool."""
        self._shutdown.set()

        # Wait for workers to finish
        deadline = time.time() + timeout
        for worker in self.workers:
            remaining_time = max(0, deadline - time.time())
            worker.join(timeout=remaining_time)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive worker pool statistics."""
        with self._lock:
            return {
                "active_workers": len([w for w in self.workers if w.is_alive()]),
                "total_workers": len(self.workers),
                "queue_size": self.task_queue.qsize(),
                "avg_queue_size": self.avg_queue_size,
                "worker_stats": {wid: {
                    "tasks_completed": stats.tasks_completed,
                    "tasks_failed": stats.tasks_failed,
                    "avg_execution_time": stats.avg_execution_time,
                    "success_rate": stats.tasks_completed / max(1, stats.tasks_completed + stats.tasks_failed)
                } for wid, stats in self.worker_stats.items()}
            }


class ParallelDocumentationProcessor:
    """High-level parallel processor for documentation tasks."""

    def __init__(self, max_workers: int = None):
        self.worker_pool = AdaptiveWorkerPool(
            min_workers=2,
            max_workers=max_workers,
            queue_size=1000
        )
        self.cache = get_cache()

    def process_files_parallel(self,
                             file_paths: List[Path],
                             processor_func: Callable,
                             **kwargs) -> List[ProcessingResult]:
        """Process multiple files in parallel."""
        results = []

        # Submit all tasks
        for i, file_path in enumerate(file_paths):
            task_id = f"file_process_{i}_{file_path.name}"
            self.worker_pool.submit_task(
                task_id=task_id,
                func=processor_func,
                args=(file_path,),
                **kwargs
            )

        # Collect results
        for _ in file_paths:
            result = self.worker_pool.get_result(timeout=60.0)
            if result:
                results.append(result)

        return results

    def process_routes_parallel(self,
                              routes: List[Dict[str, Any]],
                              processor_func: Callable,
                              **kwargs) -> List[ProcessingResult]:
        """Process routes in parallel."""
        results = []

        # Submit route processing tasks
        for i, route in enumerate(routes):
            task_id = f"route_process_{i}_{route.get('path', 'unknown')}"
            self.worker_pool.submit_task(
                task_id=task_id,
                func=processor_func,
                args=(route,),
                **kwargs
            )

        # Collect results
        for _ in routes:
            result = self.worker_pool.get_result(timeout=30.0)
            if result:
                results.append(result)

        return results

    def shutdown(self) -> None:
        """Shutdown the parallel processor."""
        self.worker_pool.shutdown()

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "worker_pool": self.worker_pool.get_stats(),
            "cache_stats": self.cache.get_overall_stats()
        }


# Global processor instance
_global_processor: Optional[ParallelDocumentationProcessor] = None
_processor_lock = threading.RLock()


def get_parallel_processor(max_workers: int = None) -> ParallelDocumentationProcessor:
    """Get global parallel processor instance."""
    global _global_processor
    with _processor_lock:
        if _global_processor is None:
            _global_processor = ParallelDocumentationProcessor(max_workers)
        return _global_processor


def parallel_operation(max_workers: int = None):
    """Decorator for parallelizing operations."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            processor = get_parallel_processor(max_workers)

            # For single operations, just execute normally
            # This decorator is mainly for marking functions that can be parallelized
            return func(*args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator
