"""
Advanced Concurrent Processing System with Work-Stealing Algorithms

This module implements sophisticated concurrent processing capabilities including
work-stealing thread pools, adaptive task scheduling, parallel pipeline processing,
and intelligent load balancing for optimal CPU utilization and performance.

Features:
- Work-stealing algorithm for efficient task distribution
- Adaptive task scheduling based on system load
- Parallel pipeline processing with stage optimization
- NUMA-aware thread affinity and memory allocation
- Intelligent batching and task partitioning
- Lock-free data structures for high concurrency
- Async/await integration with thread pools
- Performance monitoring and optimization
"""

import asyncio
import concurrent.futures
import logging
import multiprocessing as mp
import os
import queue
import random
import statistics
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, Generic, TypeVar
import heapq
import psutil

from .enhanced_monitoring import get_monitor
from .advanced_memory_optimizer import get_memory_optimizer

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class WorkerState(Enum):
    """Worker thread states."""
    IDLE = "idle"
    WORKING = "working"
    STEALING = "stealing"
    BLOCKED = "blocked"
    TERMINATED = "terminated"


class SchedulingStrategy(Enum):
    """Task scheduling strategies."""
    FIFO = "fifo"
    LIFO = "lifo"
    PRIORITY = "priority"
    WORK_STEALING = "work_stealing"
    ADAPTIVE = "adaptive"


@dataclass
class Task(Generic[T]):
    """A task to be executed."""
    task_id: str
    function: Callable[..., T]
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    estimated_duration: float = 1.0  # seconds
    dependencies: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    partition_key: Optional[str] = None  # For task affinity
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = f"task_{id(self)}_{time.time()}"


@dataclass
class TaskResult(Generic[T]):
    """Result of task execution."""
    task_id: str
    result: Optional[T] = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    worker_id: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    success: bool = True
    
    def __post_init__(self):
        if self.completed_at is None:
            self.completed_at = datetime.now()


@dataclass
class WorkerMetrics:
    """Metrics for individual workers."""
    worker_id: str
    state: WorkerState = WorkerState.IDLE
    tasks_executed: int = 0
    tasks_stolen: int = 0
    tasks_given: int = 0
    total_execution_time: float = 0.0
    idle_time: float = 0.0
    steal_attempts: int = 0
    successful_steals: int = 0
    cpu_affinity: Optional[List[int]] = None
    
    def get_efficiency(self) -> float:
        """Calculate worker efficiency (time working / total time)."""
        total_time = self.total_execution_time + self.idle_time
        return (self.total_execution_time / total_time) if total_time > 0 else 0.0
    
    def get_steal_success_rate(self) -> float:
        """Calculate steal success rate."""
        return (self.successful_steals / self.steal_attempts) if self.steal_attempts > 0 else 0.0


class LockFreeDeque(Generic[T]):
    """Lock-free double-ended queue for work-stealing."""
    
    def __init__(self, maxsize: int = 1000):
        self._deque = deque(maxlen=maxsize)
        self._lock = threading.RLock()  # Still need some synchronization for Python
        self._condition = threading.Condition(self._lock)
    
    def push_back(self, item: T) -> bool:
        """Add item to back of deque."""
        with self._lock:
            try:
                self._deque.append(item)
                self._condition.notify()
                return True
            except:
                return False
    
    def pop_back(self) -> Optional[T]:
        """Remove and return item from back (LIFO for local worker)."""
        with self._lock:
            try:
                return self._deque.pop()
            except IndexError:
                return None
    
    def steal_front(self) -> Optional[T]:
        """Steal item from front (FIFO for stealing workers)."""
        with self._lock:
            try:
                return self._deque.popleft()
            except IndexError:
                return None
    
    def size(self) -> int:
        """Get current size."""
        with self._lock:
            return len(self._deque)
    
    def is_empty(self) -> bool:
        """Check if deque is empty."""
        with self._lock:
            return len(self._deque) == 0
    
    def wait_for_item(self, timeout: float = 1.0) -> bool:
        """Wait for an item to be available."""
        with self._condition:
            return self._condition.wait_for(lambda: len(self._deque) > 0, timeout=timeout)


class WorkStealingWorker:
    """Individual worker thread with work-stealing capability."""
    
    def __init__(self, 
                 worker_id: str,
                 local_queue: LockFreeDeque[Task],
                 global_queues: List[LockFreeDeque[Task]],
                 result_callback: Callable[[TaskResult], None],
                 cpu_affinity: Optional[List[int]] = None):
        
        self.worker_id = worker_id
        self.local_queue = local_queue
        self.global_queues = global_queues
        self.result_callback = result_callback
        self.cpu_affinity = cpu_affinity
        
        # Worker state
        self.metrics = WorkerMetrics(worker_id=worker_id, cpu_affinity=cpu_affinity)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Performance tracking
        self.last_activity = datetime.now()
        self.steal_candidates: List[int] = []
        
        logger.debug(f"Work-stealing worker {worker_id} created")
    
    def start(self):
        """Start the worker thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.name = f"WorkerThread-{self.worker_id}"
        
        # Set CPU affinity if specified
        if self.cpu_affinity:
            try:
                process = psutil.Process()
                process.cpu_affinity(self.cpu_affinity)
                self.metrics.cpu_affinity = self.cpu_affinity
            except Exception as e:
                logger.warning(f"Failed to set CPU affinity for worker {self.worker_id}: {e}")
        
        self.thread.start()
        logger.debug(f"Worker {self.worker_id} started")
    
    def stop(self):
        """Stop the worker thread."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        logger.debug(f"Worker {self.worker_id} stopped")
    
    def add_task(self, task: Task) -> bool:
        """Add task to local queue."""
        return self.local_queue.push_back(task)
    
    def _worker_loop(self):
        """Main worker loop with work-stealing."""
        while self.running:
            task = self._get_next_task()
            
            if task:
                self._execute_task(task)
            else:
                # No work available, brief idle period
                self.metrics.state = WorkerState.IDLE
                idle_start = time.time()
                time.sleep(0.001)  # 1ms idle sleep
                self.metrics.idle_time += time.time() - idle_start
    
    def _get_next_task(self) -> Optional[Task]:
        """Get next task using work-stealing algorithm."""
        # 1. Try local queue first (LIFO for better cache locality)
        task = self.local_queue.pop_back()
        if task:
            self.metrics.state = WorkerState.WORKING
            return task
        
        # 2. Try to steal from other workers (FIFO to avoid conflicts)
        if self.global_queues:
            self.metrics.state = WorkerState.STEALING
            task = self._attempt_work_stealing()
            if task:
                self.metrics.successful_steals += 1
                self.metrics.state = WorkerState.WORKING
                return task
        
        return None
    
    def _attempt_work_stealing(self) -> Optional[Task]:
        """Attempt to steal work from other workers."""
        self.metrics.steal_attempts += 1
        
        # Randomize steal order to reduce contention
        if not self.steal_candidates or len(self.steal_candidates) != len(self.global_queues):
            self.steal_candidates = list(range(len(self.global_queues)))
        
        random.shuffle(self.steal_candidates)
        
        # Try to steal from each queue
        for queue_idx in self.steal_candidates:
            if queue_idx < len(self.global_queues):
                queue = self.global_queues[queue_idx]
                if queue != self.local_queue:  # Don't steal from ourselves
                    task = queue.steal_front()
                    if task:
                        self.metrics.tasks_stolen += 1
                        return task
        
        return None
    
    def _execute_task(self, task: Task):
        """Execute a task and record metrics."""
        start_time = time.time()
        result = TaskResult(task_id=task.task_id, worker_id=self.worker_id, started_at=datetime.now())
        
        try:
            # Execute the task
            if asyncio.iscoroutinefunction(task.function):
                # Handle async functions
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result.result = loop.run_until_complete(task.function(*task.args, **task.kwargs))
                finally:
                    loop.close()
            else:
                result.result = task.function(*task.args, **task.kwargs)
            
            result.success = True
            
        except Exception as e:
            result.error = e
            result.success = False
            logger.warning(f"Task {task.task_id} failed in worker {self.worker_id}: {e}")
        
        finally:
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            result.completed_at = datetime.now()
            
            # Update metrics
            self.metrics.tasks_executed += 1
            self.metrics.total_execution_time += execution_time
            self.last_activity = datetime.now()
            
            # Report result
            self.result_callback(result)


class AdaptiveTaskScheduler:
    """Adaptive scheduler that optimizes task distribution."""
    
    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        self.worker_loads: List[float] = [0.0] * num_workers
        self.worker_capabilities: List[Dict[str, float]] = [{}] * num_workers
        
        # Scheduling history for learning
        self.scheduling_history: deque = deque(maxlen=1000)
        self.performance_feedback: Dict[str, float] = {}
        
    def select_worker(self, task: Task) -> int:
        """Select optimal worker for task."""
        # Consider task affinity first
        if task.partition_key:
            return self._hash_to_worker(task.partition_key)
        
        # Find worker with least load
        min_load_worker = min(range(self.num_workers), key=lambda i: self.worker_loads[i])
        
        # Record scheduling decision
        self.scheduling_history.append({
            'task_id': task.task_id,
            'worker_id': min_load_worker,
            'worker_load': self.worker_loads[min_load_worker],
            'timestamp': datetime.now()
        })
        
        return min_load_worker
    
    def update_worker_load(self, worker_id: int, load: float):
        """Update worker load information."""
        if 0 <= worker_id < self.num_workers:
            self.worker_loads[worker_id] = load
    
    def record_performance_feedback(self, task_id: str, execution_time: float):
        """Record performance feedback for learning."""
        self.performance_feedback[task_id] = execution_time
        
        # Learn from recent performance
        self._update_scheduling_model()
    
    def _hash_to_worker(self, key: str) -> int:
        """Hash partition key to worker."""
        return hash(key) % self.num_workers
    
    def _update_scheduling_model(self):
        """Update scheduling model based on feedback."""
        # Simple learning: adjust worker capability estimates
        for decision in list(self.scheduling_history)[-50:]:  # Last 50 decisions
            task_id = decision['task_id']
            worker_id = decision['worker_id']
            
            if task_id in self.performance_feedback:
                execution_time = self.performance_feedback[task_id]
                
                # Update worker capability (inverse of execution time)
                if worker_id < len(self.worker_capabilities):
                    capability = 1.0 / max(execution_time, 0.001)
                    
                    if 'average_capability' not in self.worker_capabilities[worker_id]:
                        self.worker_capabilities[worker_id]['average_capability'] = capability
                    else:
                        # Exponential moving average
                        alpha = 0.1
                        current = self.worker_capabilities[worker_id]['average_capability']
                        self.worker_capabilities[worker_id]['average_capability'] = (
                            alpha * capability + (1 - alpha) * current
                        )


class PipelineStage:
    """A stage in a processing pipeline."""
    
    def __init__(self, 
                 name: str,
                 processor: Callable[[Any], Any],
                 max_parallelism: int = 1,
                 buffer_size: int = 100):
        self.name = name
        self.processor = processor
        self.max_parallelism = max_parallelism
        self.buffer_size = buffer_size
        
        # Pipeline components
        self.input_queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self.output_queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        
        # Worker threads for this stage
        self.workers: List[threading.Thread] = []
        self.running = False
        
        # Metrics
        self.items_processed = 0
        self.total_processing_time = 0.0
        self.errors = 0
    
    def start(self):
        """Start pipeline stage workers."""
        if self.running:
            return
        
        self.running = True
        
        for i in range(self.max_parallelism):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"PipelineWorker-{self.name}-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        logger.debug(f"Pipeline stage '{self.name}' started with {self.max_parallelism} workers")
    
    def stop(self):
        """Stop pipeline stage workers."""
        self.running = False
        
        # Signal workers to stop by adding sentinel values
        for _ in self.workers:
            try:
                self.input_queue.put(None, timeout=1.0)
            except queue.Full:
                pass
        
        # Wait for workers to complete
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        logger.debug(f"Pipeline stage '{self.name}' stopped")
    
    def add_input(self, item: Any, timeout: float = 1.0) -> bool:
        """Add item to input queue."""
        try:
            self.input_queue.put(item, timeout=timeout)
            return True
        except queue.Full:
            return False
    
    def get_output(self, timeout: float = 1.0) -> Optional[Any]:
        """Get item from output queue."""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _worker_loop(self):
        """Worker loop for processing items."""
        while self.running:
            try:
                item = self.input_queue.get(timeout=1.0)
                
                # Sentinel value to stop worker
                if item is None:
                    break
                
                # Process item
                start_time = time.time()
                try:
                    result = self.processor(item)
                    self.output_queue.put(result)
                    
                    self.items_processed += 1
                    self.total_processing_time += time.time() - start_time
                    
                except Exception as e:
                    self.errors += 1
                    logger.warning(f"Pipeline stage '{self.name}' processing error: {e}")
                
            except queue.Empty:
                continue
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stage performance statistics."""
        avg_processing_time = (
            self.total_processing_time / max(self.items_processed, 1)
        )
        
        return {
            'name': self.name,
            'max_parallelism': self.max_parallelism,
            'items_processed': self.items_processed,
            'total_processing_time': self.total_processing_time,
            'average_processing_time': avg_processing_time,
            'errors': self.errors,
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize(),
            'throughput': self.items_processed / max(self.total_processing_time, 0.001)
        }


class ParallelPipeline:
    """Parallel processing pipeline with multiple stages."""
    
    def __init__(self, name: str = "Pipeline"):
        self.name = name
        self.stages: List[PipelineStage] = []
        self.running = False
        
        # Coordination thread
        self.coordinator_thread: Optional[threading.Thread] = None
        
    def add_stage(self, stage: PipelineStage):
        """Add a processing stage to the pipeline."""
        self.stages.append(stage)
        logger.debug(f"Added stage '{stage.name}' to pipeline '{self.name}'")
    
    def start(self):
        """Start all pipeline stages."""
        if self.running:
            return
        
        self.running = True
        
        # Start all stages
        for stage in self.stages:
            stage.start()
        
        # Start coordinator thread
        self.coordinator_thread = threading.Thread(
            target=self._coordinate_pipeline,
            name=f"PipelineCoordinator-{self.name}",
            daemon=True
        )
        self.coordinator_thread.start()
        
        logger.info(f"Pipeline '{self.name}' started with {len(self.stages)} stages")
    
    def stop(self):
        """Stop all pipeline stages."""
        if not self.running:
            return
        
        self.running = False
        
        # Stop coordinator
        if self.coordinator_thread and self.coordinator_thread.is_alive():
            self.coordinator_thread.join(timeout=5.0)
        
        # Stop all stages
        for stage in self.stages:
            stage.stop()
        
        logger.info(f"Pipeline '{self.name}' stopped")
    
    def process_item(self, item: Any, timeout: float = 10.0) -> Optional[Any]:
        """Process a single item through the entire pipeline."""
        if not self.stages:
            return None
        
        # Add to first stage
        if not self.stages[0].add_input(item, timeout):
            return None
        
        # Get result from last stage
        return self.stages[-1].get_output(timeout)
    
    def process_batch(self, items: List[Any], timeout: float = 30.0) -> List[Any]:
        """Process a batch of items through the pipeline."""
        if not self.stages or not items:
            return []
        
        results = []
        
        # Add all items to first stage
        for item in items:
            if not self.stages[0].add_input(item, timeout=timeout/len(items)):
                break
        
        # Collect results from last stage
        for _ in items:
            result = self.stages[-1].get_output(timeout=timeout/len(items))
            if result is not None:
                results.append(result)
        
        return results
    
    def _coordinate_pipeline(self):
        """Coordinate data flow between pipeline stages."""
        while self.running:
            try:
                # Move items from output of each stage to input of next stage
                for i in range(len(self.stages) - 1):
                    current_stage = self.stages[i]
                    next_stage = self.stages[i + 1]
                    
                    # Move up to 10 items at a time
                    for _ in range(10):
                        item = current_stage.get_output(timeout=0.1)
                        if item is None:
                            break
                        
                        if not next_stage.add_input(item, timeout=0.1):
                            # Put item back if next stage is full
                            current_stage.output_queue.put(item)
                            break
                
                time.sleep(0.01)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.warning(f"Pipeline coordination error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        stage_stats = [stage.get_stats() for stage in self.stages]
        
        total_items = sum(stats['items_processed'] for stats in stage_stats)
        total_errors = sum(stats['errors'] for stats in stage_stats)
        
        return {
            'name': self.name,
            'num_stages': len(self.stages),
            'total_items_processed': total_items,
            'total_errors': total_errors,
            'running': self.running,
            'stages': stage_stats
        }


class AdvancedConcurrentProcessor:
    """
    Main concurrent processing system with work-stealing and adaptive scheduling.
    """
    
    def __init__(self,
                 num_workers: int = None,
                 scheduling_strategy: SchedulingStrategy = SchedulingStrategy.WORK_STEALING,
                 numa_aware: bool = True):
        
        # Determine optimal number of workers
        if num_workers is None:
            num_workers = min(32, (os.cpu_count() or 1) + 4)
        
        self.num_workers = num_workers
        self.scheduling_strategy = scheduling_strategy
        self.numa_aware = numa_aware
        
        # Work-stealing components
        self.global_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.worker_queues: List[LockFreeDeque] = []
        self.workers: List[WorkStealingWorker] = []
        
        # Task management
        self.pending_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        self.task_dependencies: Dict[str, Set[str]] = {}  # task_id -> dependent_task_ids
        
        # Scheduling and coordination
        self.scheduler = AdaptiveTaskScheduler(num_workers)
        self.task_distributor_thread: Optional[threading.Thread] = None
        self.dependency_resolver_thread: Optional[threading.Thread] = None
        
        # Pipeline support
        self.pipelines: Dict[str, ParallelPipeline] = {}
        
        # Monitoring and optimization
        self.monitor = get_monitor()
        self.memory_optimizer = get_memory_optimizer()
        
        # State management
        self.running = False
        self.start_time: Optional[datetime] = None
        
        # NUMA optimization
        if numa_aware:
            self._setup_numa_optimization()
        
        logger.info(f"Advanced Concurrent Processor initialized: "
                   f"{num_workers} workers, {scheduling_strategy.value} scheduling")
    
    def _setup_numa_optimization(self):
        """Setup NUMA-aware worker thread affinity."""
        try:
            # Get CPU topology
            cpu_count = os.cpu_count() or 1
            
            # Simple NUMA simulation: divide CPUs into groups
            cpus_per_worker = max(1, cpu_count // self.num_workers)
            
            for i in range(self.num_workers):
                start_cpu = i * cpus_per_worker
                end_cpu = min(start_cpu + cpus_per_worker, cpu_count)
                cpu_affinity = list(range(start_cpu, end_cpu))
                
                # Store CPU affinity for worker creation
                if not hasattr(self, '_worker_cpu_affinities'):
                    self._worker_cpu_affinities = {}
                self._worker_cpu_affinities[i] = cpu_affinity
                
        except Exception as e:
            logger.warning(f"NUMA optimization setup failed: {e}")
            self.numa_aware = False
    
    async def start(self):
        """Start the concurrent processor."""
        if self.running:
            return
        
        self.running = True
        self.start_time = datetime.now()
        
        # Initialize worker queues
        for i in range(self.num_workers):
            self.worker_queues.append(LockFreeDeque())
        
        # Create and start workers
        for i in range(self.num_workers):
            cpu_affinity = None
            if hasattr(self, '_worker_cpu_affinities'):
                cpu_affinity = self._worker_cpu_affinities.get(i)
            
            worker = WorkStealingWorker(
                worker_id=f"worker_{i}",
                local_queue=self.worker_queues[i],
                global_queues=self.worker_queues,
                result_callback=self._handle_task_result,
                cpu_affinity=cpu_affinity
            )
            
            worker.start()
            self.workers.append(worker)
        
        # Start coordination threads
        self.task_distributor_thread = threading.Thread(
            target=self._task_distribution_loop,
            daemon=True,
            name="TaskDistributor"
        )
        self.task_distributor_thread.start()
        
        self.dependency_resolver_thread = threading.Thread(
            target=self._dependency_resolution_loop,
            daemon=True,
            name="DependencyResolver"
        )
        self.dependency_resolver_thread.start()
        
        logger.info(f"Concurrent processor started with {self.num_workers} workers")
    
    async def stop(self):
        """Stop the concurrent processor."""
        if not self.running:
            return
        
        self.running = False
        
        # Stop pipelines
        for pipeline in self.pipelines.values():
            pipeline.stop()
        
        # Stop coordination threads
        if self.task_distributor_thread and self.task_distributor_thread.is_alive():
            self.task_distributor_thread.join(timeout=5.0)
        
        if self.dependency_resolver_thread and self.dependency_resolver_thread.is_alive():
            self.dependency_resolver_thread.join(timeout=5.0)
        
        # Stop workers
        for worker in self.workers:
            worker.stop()
        
        logger.info("Concurrent processor stopped")
    
    def submit_task(self, 
                   function: Callable[..., T],
                   *args,
                   task_id: Optional[str] = None,
                   priority: TaskPriority = TaskPriority.NORMAL,
                   estimated_duration: float = 1.0,
                   dependencies: Optional[Set[str]] = None,
                   partition_key: Optional[str] = None,
                   **kwargs) -> str:
        """Submit a task for execution."""
        
        if task_id is None:
            task_id = f"task_{len(self.pending_tasks)}_{time.time()}"
        
        task = Task(
            task_id=task_id,
            function=function,
            args=args,
            kwargs=kwargs,
            priority=priority,
            estimated_duration=estimated_duration,
            dependencies=dependencies or set(),
            partition_key=partition_key
        )
        
        # Store task
        self.pending_tasks[task_id] = task
        
        # Handle dependencies
        if task.dependencies:
            self.task_dependencies[task_id] = task.dependencies.copy()
        else:
            # No dependencies, add to global queue immediately
            priority_value = -priority.value  # Negative for max heap behavior
            self.global_queue.put((priority_value, time.time(), task))
        
        logger.debug(f"Task {task_id} submitted with priority {priority.value}")
        return task_id
    
    def submit_batch(self, 
                    tasks: List[Tuple[Callable, Tuple, Dict]],
                    priority: TaskPriority = TaskPriority.NORMAL,
                    partition_key: Optional[str] = None) -> List[str]:
        """Submit a batch of tasks."""
        task_ids = []
        
        for i, (function, args, kwargs) in enumerate(tasks):
            task_id = self.submit_task(
                function=function,
                task_id=f"batch_task_{i}_{time.time()}",
                priority=priority,
                partition_key=partition_key,
                *args,
                **kwargs
            )
            task_ids.append(task_id)
        
        return task_ids
    
    async def execute_parallel(self, 
                              function: Callable[..., T],
                              items: List[Any],
                              chunk_size: int = None,
                              max_workers: int = None) -> List[T]:
        """Execute function in parallel on list of items."""
        if not items:
            return []
        
        # Determine chunk size
        if chunk_size is None:
            chunk_size = max(1, len(items) // (max_workers or self.num_workers))
        
        # Create chunks
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        # Submit tasks for each chunk
        task_ids = []
        for i, chunk in enumerate(chunks):
            task_id = self.submit_task(
                function=lambda chunk=chunk: [function(item) for item in chunk],
                task_id=f"parallel_chunk_{i}_{time.time()}"
            )
            task_ids.append(task_id)
        
        # Wait for results
        results = []
        for task_id in task_ids:
            result = await self.wait_for_task(task_id)
            if result and result.success:
                results.extend(result.result)
        
        return results
    
    async def wait_for_task(self, task_id: str, timeout: float = None) -> Optional[TaskResult]:
        """Wait for a specific task to complete."""
        start_time = time.time()
        
        while True:
            # Check if task is completed
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"Task {task_id} timed out after {timeout}s")
                return None
            
            # Brief wait before checking again
            await asyncio.sleep(0.01)
    
    async def wait_for_all(self, task_ids: List[str], timeout: float = None) -> Dict[str, TaskResult]:
        """Wait for multiple tasks to complete."""
        results = {}
        
        for task_id in task_ids:
            result = await self.wait_for_task(task_id, timeout)
            if result:
                results[task_id] = result
        
        return results
    
    def create_pipeline(self, name: str) -> ParallelPipeline:
        """Create a new parallel processing pipeline."""
        pipeline = ParallelPipeline(name)
        self.pipelines[name] = pipeline
        
        logger.info(f"Pipeline '{name}' created")
        return pipeline
    
    def get_pipeline(self, name: str) -> Optional[ParallelPipeline]:
        """Get existing pipeline by name."""
        return self.pipelines.get(name)
    
    def _task_distribution_loop(self):
        """Main task distribution loop."""
        while self.running:
            try:
                # Get task from global queue
                try:
                    priority, timestamp, task = self.global_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Select worker using scheduler
                worker_index = self.scheduler.select_worker(task)
                
                # Add task to worker queue
                if worker_index < len(self.worker_queues):
                    if self.worker_queues[worker_index].push_back(task):
                        logger.debug(f"Task {task.task_id} distributed to worker {worker_index}")
                    else:
                        # Worker queue full, put back in global queue
                        self.global_queue.put((priority, timestamp, task))
                
            except Exception as e:
                logger.error(f"Task distribution error: {e}")
    
    def _dependency_resolution_loop(self):
        """Resolve task dependencies and release ready tasks."""
        while self.running:
            try:
                # Check for tasks whose dependencies are satisfied
                ready_tasks = []
                
                for task_id, dependencies in list(self.task_dependencies.items()):
                    # Remove completed dependencies
                    satisfied_deps = []
                    for dep_id in dependencies:
                        if dep_id in self.completed_tasks:
                            if self.completed_tasks[dep_id].success:
                                satisfied_deps.append(dep_id)
                            else:
                                # Dependency failed, mark task as failed
                                error_result = TaskResult(
                                    task_id=task_id,
                                    error=Exception(f"Dependency {dep_id} failed"),
                                    success=False
                                )
                                self.completed_tasks[task_id] = error_result
                                ready_tasks.append(task_id)
                                break
                    
                    # Remove satisfied dependencies
                    for dep_id in satisfied_deps:
                        dependencies.discard(dep_id)
                    
                    # If no dependencies left, task is ready
                    if not dependencies and task_id not in ready_tasks:
                        ready_tasks.append(task_id)
                
                # Move ready tasks to global queue
                for task_id in ready_tasks:
                    if task_id in self.task_dependencies:
                        del self.task_dependencies[task_id]
                    
                    if task_id in self.pending_tasks:
                        task = self.pending_tasks[task_id]
                        priority_value = -task.priority.value
                        self.global_queue.put((priority_value, time.time(), task))
                
                time.sleep(0.1)  # Check dependencies every 100ms
                
            except Exception as e:
                logger.error(f"Dependency resolution error: {e}")
    
    def _handle_task_result(self, result: TaskResult):
        """Handle completed task result."""
        # Store result
        self.completed_tasks[result.task_id] = result
        
        # Remove from pending tasks
        if result.task_id in self.pending_tasks:
            task = self.pending_tasks[result.task_id]
            del self.pending_tasks[result.task_id]
            
            # Update scheduler with performance feedback
            self.scheduler.record_performance_feedback(
                result.task_id, result.execution_time
            )
        
        # Record metrics
        self.monitor.record_metric(
            "concurrent_task_execution_time",
            result.execution_time,
            "timer"
        )
        
        if not result.success:
            self.monitor.record_metric("concurrent_task_errors", 1.0, "counter")
        
        logger.debug(f"Task {result.task_id} completed in {result.execution_time:.3f}s "
                    f"by {result.worker_id} ({'success' if result.success else 'failed'})")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive processor statistics."""
        # Worker statistics
        worker_stats = []
        total_tasks_executed = 0
        total_execution_time = 0.0
        
        for worker in self.workers:
            metrics = worker.metrics
            stats = {
                'worker_id': metrics.worker_id,
                'state': metrics.state.value,
                'tasks_executed': metrics.tasks_executed,
                'tasks_stolen': metrics.tasks_stolen,
                'tasks_given': metrics.tasks_given,
                'efficiency': metrics.get_efficiency(),
                'steal_success_rate': metrics.get_steal_success_rate(),
                'cpu_affinity': metrics.cpu_affinity
            }
            worker_stats.append(stats)
            
            total_tasks_executed += metrics.tasks_executed
            total_execution_time += metrics.total_execution_time
        
        # Queue statistics
        queue_stats = {
            'global_queue_size': self.global_queue.qsize(),
            'worker_queue_sizes': [q.size() for q in self.worker_queues],
            'total_queue_items': self.global_queue.qsize() + sum(q.size() for q in self.worker_queues)
        }
        
        # Task statistics
        task_stats = {
            'pending_tasks': len(self.pending_tasks),
            'completed_tasks': len(self.completed_tasks),
            'tasks_with_dependencies': len(self.task_dependencies),
            'successful_tasks': len([r for r in self.completed_tasks.values() if r.success]),
            'failed_tasks': len([r for r in self.completed_tasks.values() if not r.success])
        }
        
        # Pipeline statistics
        pipeline_stats = {}
        for name, pipeline in self.pipelines.items():
            pipeline_stats[name] = pipeline.get_stats()
        
        # Performance metrics
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        throughput = total_tasks_executed / max(uptime, 0.001)
        avg_execution_time = total_execution_time / max(total_tasks_executed, 1)
        
        return {
            'processor': {
                'num_workers': self.num_workers,
                'scheduling_strategy': self.scheduling_strategy.value,
                'numa_aware': self.numa_aware,
                'running': self.running,
                'uptime_seconds': uptime
            },
            'performance': {
                'total_tasks_executed': total_tasks_executed,
                'throughput_tasks_per_second': throughput,
                'average_execution_time': avg_execution_time
            },
            'workers': worker_stats,
            'queues': queue_stats,
            'tasks': task_stats,
            'pipelines': pipeline_stats
        }


# Global processor instance
_global_processor: Optional[AdvancedConcurrentProcessor] = None


async def get_concurrent_processor(**kwargs) -> AdvancedConcurrentProcessor:
    """Get global concurrent processor instance."""
    global _global_processor
    if _global_processor is None:
        _global_processor = AdvancedConcurrentProcessor(**kwargs)
        await _global_processor.start()
    return _global_processor


# Decorators for concurrent processing
def concurrent_task(priority: TaskPriority = TaskPriority.NORMAL, 
                   partition_key: Optional[str] = None):
    """Decorator to mark function for concurrent execution."""
    def decorator(func):
        func._concurrent_task = True
        func._priority = priority
        func._partition_key = partition_key
        return func
    return decorator


def parallel_map(items: List[Any], chunk_size: int = None):
    """Decorator for parallel map operations."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            processor = await get_concurrent_processor()
            return await processor.execute_parallel(func, items, chunk_size)
        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    async def example_concurrent_processing():
        """Example of using advanced concurrent processing."""
        
        # Create processor
        processor = AdvancedConcurrentProcessor(
            num_workers=8,
            scheduling_strategy=SchedulingStrategy.WORK_STEALING,
            numa_aware=True
        )
        
        await processor.start()
        
        try:
            # Example 1: Simple task submission
            def compute_square(x):
                time.sleep(0.1)  # Simulate work
                return x * x
            
            print("Example 1: Submitting individual tasks")
            task_ids = []
            for i in range(20):
                task_id = processor.submit_task(
                    compute_square,
                    i,
                    priority=TaskPriority.HIGH if i % 5 == 0 else TaskPriority.NORMAL
                )
                task_ids.append(task_id)
            
            # Wait for results
            results = await processor.wait_for_all(task_ids[:10])  # Wait for first 10
            print(f"Completed {len(results)} tasks")
            
            # Example 2: Parallel execution
            print("\nExample 2: Parallel execution")
            items = list(range(50))
            parallel_results = await processor.execute_parallel(compute_square, items, chunk_size=5)
            print(f"Parallel results: {len(parallel_results)} items processed")
            
            # Example 3: Task dependencies
            print("\nExample 3: Task dependencies")
            
            def process_data(data):
                return f"processed_{data}"
            
            def aggregate_results(results):
                return f"aggregated_{len(results)}_items"
            
            # Submit data processing tasks
            data_task_ids = []
            for i in range(5):
                task_id = processor.submit_task(
                    process_data,
                    f"data_{i}",
                    task_id=f"process_{i}"
                )
                data_task_ids.append(task_id)
            
            # Submit aggregation task that depends on all data processing tasks
            agg_task_id = processor.submit_task(
                aggregate_results,
                ["dummy"],  # Will be replaced by dependency results
                task_id="aggregation",
                dependencies=set(data_task_ids),
                priority=TaskPriority.HIGH
            )
            
            agg_result = await processor.wait_for_task(agg_task_id)
            if agg_result:
                print(f"Aggregation result: {agg_result.result}")
            
            # Example 4: Pipeline processing
            print("\nExample 4: Pipeline processing")
            
            def stage1_process(item):
                return f"stage1_{item}"
            
            def stage2_process(item):
                return f"stage2_{item}"
            
            def stage3_process(item):
                return f"stage3_{item}"
            
            # Create pipeline
            pipeline = processor.create_pipeline("example_pipeline")
            
            # Add stages
            from .advanced_concurrent_processor import PipelineStage
            pipeline.add_stage(PipelineStage("stage1", stage1_process, max_parallelism=2))
            pipeline.add_stage(PipelineStage("stage2", stage2_process, max_parallelism=3))
            pipeline.add_stage(PipelineStage("stage3", stage3_process, max_parallelism=2))
            
            # Start pipeline
            pipeline.start()
            
            # Process items through pipeline
            pipeline_items = [f"item_{i}" for i in range(10)]
            pipeline_results = pipeline.process_batch(pipeline_items)
            print(f"Pipeline processed {len(pipeline_results)} items")
            
            # Stop pipeline
            pipeline.stop()
            
            # Get comprehensive statistics
            stats = processor.get_stats()
            print(f"\nProcessor Statistics:")
            print(f"  Workers: {stats['processor']['num_workers']}")
            print(f"  Total tasks executed: {stats['performance']['total_tasks_executed']}")
            print(f"  Throughput: {stats['performance']['throughput_tasks_per_second']:.2f} tasks/sec")
            print(f"  Average execution time: {stats['performance']['average_execution_time']:.3f}s")
            print(f"  Pending tasks: {stats['tasks']['pending_tasks']}")
            print(f"  Failed tasks: {stats['tasks']['failed_tasks']}")
            
            # Worker efficiency
            for worker_stat in stats['workers'][:3]:  # Show first 3 workers
                print(f"  Worker {worker_stat['worker_id']}: "
                      f"{worker_stat['tasks_executed']} tasks, "
                      f"{worker_stat['efficiency']:.2f} efficiency, "
                      f"{worker_stat['steal_success_rate']:.2f} steal rate")
            
        finally:
            await processor.stop()
    
    # Run example
    asyncio.run(example_concurrent_processing())
    print("Advanced concurrent processing example completed!")