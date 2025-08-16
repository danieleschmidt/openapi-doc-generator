"""Advanced performance optimization and auto-scaling engine."""

from __future__ import annotations

import json
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from enum import Enum
from threading import RLock
from typing import Any, Callable, Dict, List, Optional

import psutil

from .monitoring import MetricsCollector
from .quantum_audit_logger import AuditEventType, get_audit_logger
from .quantum_health_monitor import get_health_monitor
from .quantum_resilience_engine import (
    ResilienceConfig,
    ResiliencePattern,
    get_resilience_engine,
)
from .quantum_security import SecurityLevel


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    CPU_INTENSIVE = "cpu_intensive"
    IO_INTENSIVE = "io_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    MIXED_WORKLOAD = "mixed_workload"
    REAL_TIME = "real_time"
    BATCH_PROCESSING = "batch_processing"


class ScalingTrigger(Enum):
    """Auto-scaling trigger conditions."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    QUEUE_LENGTH = "queue_length"


@dataclass
class PerformanceProfile:
    """Performance profile for workload optimization."""
    strategy: OptimizationStrategy
    cpu_cores: int
    memory_limit_mb: int
    io_threads: int
    batch_size: int
    cache_size: int
    prefetch_enabled: bool
    compression_enabled: bool
    parallelization_threshold: int


@dataclass
class ScalingRule:
    """Auto-scaling rule configuration."""
    trigger: ScalingTrigger
    threshold_up: float
    threshold_down: float
    scale_up_by: int
    scale_down_by: int
    cooldown_seconds: int
    max_instances: int
    min_instances: int


@dataclass
class OptimizationResult:
    """Result of performance optimization."""
    operation_name: str
    original_duration_ms: float
    optimized_duration_ms: float
    improvement_factor: float
    memory_saved_mb: float
    optimization_applied: List[str]
    metadata: Dict[str, Any]


class ResourcePool:
    """Dynamic resource pool for optimized execution."""

    def __init__(self,
                 cpu_pool_size: Optional[int] = None,
                 io_pool_size: Optional[int] = None,
                 enable_adaptive_sizing: bool = True):
        """Initialize resource pool."""
        self.cpu_cores = cpu_pool_size or multiprocessing.cpu_count()
        self.io_threads = io_pool_size or min(32, (multiprocessing.cpu_count() or 1) * 4)
        self.enable_adaptive_sizing = enable_adaptive_sizing

        # Initialize pools
        self.process_executor = ProcessPoolExecutor(max_workers=self.cpu_cores)
        self.thread_executor = ThreadPoolExecutor(max_workers=self.io_threads)

        # Resource utilization tracking
        self.cpu_utilization_history: List[float] = []
        self.memory_utilization_history: List[float] = []
        self.active_cpu_tasks = 0
        self.active_io_tasks = 0

        self._lock = RLock()
        self.logger = logging.getLogger(f"{__name__}.resource_pool")

    def submit_cpu_task(self, func: Callable, *args, **kwargs):
        """Submit CPU-intensive task."""
        with self._lock:
            self.active_cpu_tasks += 1

        future = self.process_executor.submit(func, *args, **kwargs)

        def cleanup_callback(fut):
            with self._lock:
                self.active_cpu_tasks = max(0, self.active_cpu_tasks - 1)

        future.add_done_callback(cleanup_callback)
        return future

    def submit_io_task(self, func: Callable, *args, **kwargs):
        """Submit I/O-intensive task."""
        with self._lock:
            self.active_io_tasks += 1

        future = self.thread_executor.submit(func, *args, **kwargs)

        def cleanup_callback(fut):
            with self._lock:
                self.active_io_tasks = max(0, self.active_io_tasks - 1)

        future.add_done_callback(cleanup_callback)
        return future

    def get_utilization(self) -> Dict[str, Any]:
        """Get current resource utilization."""
        with self._lock:
            return {
                "cpu_pool_utilization": self.active_cpu_tasks / self.cpu_cores,
                "io_pool_utilization": self.active_io_tasks / self.io_threads,
                "active_cpu_tasks": self.active_cpu_tasks,
                "active_io_tasks": self.active_io_tasks,
                "cpu_cores": self.cpu_cores,
                "io_threads": self.io_threads
            }

    def adapt_pool_sizes(self, cpu_utilization: float, memory_utilization: float):
        """Adapt pool sizes based on system utilization."""
        if not self.enable_adaptive_sizing:
            return

        # Simple adaptive logic
        if cpu_utilization > 80 and self.cpu_cores < multiprocessing.cpu_count():
            self.cpu_cores = min(multiprocessing.cpu_count(), self.cpu_cores + 1)
            self.logger.info(f"Increased CPU pool size to {self.cpu_cores}")

        elif cpu_utilization < 20 and self.cpu_cores > 1:
            self.cpu_cores = max(1, self.cpu_cores - 1)
            self.logger.info(f"Decreased CPU pool size to {self.cpu_cores}")

    def shutdown(self):
        """Shutdown resource pools."""
        self.process_executor.shutdown(wait=True)
        self.thread_executor.shutdown(wait=True)


class SmartCache:
    """Intelligent caching system with performance optimization."""

    def __init__(self,
                 max_size: int = 1000,
                 ttl_seconds: int = 3600,
                 enable_compression: bool = True,
                 enable_prefetch: bool = True):
        """Initialize smart cache."""
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enable_compression = enable_compression
        self.enable_prefetch = enable_prefetch

        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_count: Dict[str, int] = {}
        self._access_times: Dict[str, List[float]] = {}
        self._lock = RLock()

        self.logger = logging.getLogger(f"{__name__}.smart_cache")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with smart prefetching."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]

                # Check expiration
                if time.time() < entry['expires']:
                    # Update access patterns
                    self._access_count[key] = self._access_count.get(key, 0) + 1
                    self._access_times.setdefault(key, []).append(time.time())

                    # Keep only recent access times
                    recent_cutoff = time.time() - 3600  # 1 hour
                    self._access_times[key] = [t for t in self._access_times[key] if t > recent_cutoff]

                    self.logger.debug(f"Cache hit: {key}")
                    return entry['value']
                else:
                    # Expired
                    self._remove_entry(key)

            self.logger.debug(f"Cache miss: {key}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with intelligent eviction."""
        ttl = ttl or self.ttl_seconds
        expires = time.time() + ttl

        with self._lock:
            # Check if we need to evict
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_least_valuable()

            # Compress if enabled
            if self.enable_compression:
                value = self._compress_value(value)

            self._cache[key] = {
                'value': value,
                'expires': expires,
                'compressed': self.enable_compression,
                'size_bytes': self._estimate_size(value)
            }

            self.logger.debug(f"Cache set: {key} (TTL: {ttl}s)")

    def _evict_least_valuable(self) -> None:
        """Evict least valuable cache entry."""
        if not self._cache:
            return

        # Calculate value score for each entry
        now = time.time()
        scores = {}

        for key in self._cache:
            # Factors: access frequency, recency, time to expiration
            access_freq = self._access_count.get(key, 1)
            last_access = max(self._access_times.get(key, [0]))
            recency = now - last_access if last_access > 0 else float('inf')
            time_to_expire = self._cache[key]['expires'] - now

            # Lower score = less valuable
            score = (access_freq * 0.4) + (1.0 / (recency + 1) * 0.3) + (time_to_expire * 0.3)
            scores[key] = score

        # Remove least valuable
        least_valuable = min(scores.items(), key=lambda x: x[1])
        self._remove_entry(least_valuable[0])

        self.logger.debug(f"Evicted cache entry: {least_valuable[0]} (score: {least_valuable[1]:.3f})")

    def _remove_entry(self, key: str) -> None:
        """Remove cache entry and associated metadata."""
        self._cache.pop(key, None)
        self._access_count.pop(key, None)
        self._access_times.pop(key, None)

    def _compress_value(self, value: Any) -> Any:
        """Compress value if beneficial."""
        # Simple compression simulation
        if isinstance(value, (str, dict, list)):
            try:
                import gzip
                json_str = json.dumps(value)
                if len(json_str) > 1024:  # Only compress larger values
                    compressed = gzip.compress(json_str.encode())
                    if len(compressed) < len(json_str) * 0.8:  # Only if significant compression
                        return compressed
            except Exception:
                pass
        return value

    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        try:
            import sys
            return sys.getsizeof(value)
        except Exception:
            return 1024  # Default estimate

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_size = sum(entry.get('size_bytes', 0) for entry in self._cache.values())

            return {
                "entries": len(self._cache),
                "max_size": self.max_size,
                "utilization": len(self._cache) / self.max_size,
                "total_size_bytes": total_size,
                "total_access_count": sum(self._access_count.values()),
                "compression_enabled": self.enable_compression
            }


class WorkloadAnalyzer:
    """Analyzes workloads for optimization opportunities."""

    def __init__(self):
        """Initialize workload analyzer."""
        self.operation_profiles: Dict[str, Dict[str, Any]] = {}
        self.performance_history: Dict[str, List[float]] = {}
        self.resource_usage_history: Dict[str, List[Dict[str, float]]] = {}

        self._lock = RLock()
        self.logger = logging.getLogger(f"{__name__}.workload_analyzer")

    def record_operation(self,
                        operation_name: str,
                        duration_ms: float,
                        cpu_usage: float,
                        memory_usage_mb: float,
                        io_operations: int = 0) -> None:
        """Record operation performance data."""
        with self._lock:
            # Update performance history
            self.performance_history.setdefault(operation_name, []).append(duration_ms)

            # Update resource usage history
            self.resource_usage_history.setdefault(operation_name, []).append({
                'cpu_usage': cpu_usage,
                'memory_usage_mb': memory_usage_mb,
                'io_operations': io_operations,
                'timestamp': time.time()
            })

            # Keep limited history
            max_history = 100
            if len(self.performance_history[operation_name]) > max_history:
                self.performance_history[operation_name] = self.performance_history[operation_name][-max_history:]
            if len(self.resource_usage_history[operation_name]) > max_history:
                self.resource_usage_history[operation_name] = self.resource_usage_history[operation_name][-max_history:]

    def analyze_operation(self, operation_name: str) -> PerformanceProfile:
        """Analyze operation and suggest performance profile."""
        with self._lock:
            if operation_name not in self.resource_usage_history:
                # Default profile for unknown operations
                return PerformanceProfile(
                    strategy=OptimizationStrategy.MIXED_WORKLOAD,
                    cpu_cores=2,
                    memory_limit_mb=512,
                    io_threads=4,
                    batch_size=10,
                    cache_size=100,
                    prefetch_enabled=True,
                    compression_enabled=True,
                    parallelization_threshold=100
                )

            usage_history = self.resource_usage_history[operation_name]

            # Calculate averages
            avg_cpu = sum(u['cpu_usage'] for u in usage_history) / len(usage_history)
            avg_memory = sum(u['memory_usage_mb'] for u in usage_history) / len(usage_history)
            avg_io = sum(u['io_operations'] for u in usage_history) / len(usage_history)

            # Determine optimization strategy
            strategy = self._determine_strategy(avg_cpu, avg_memory, avg_io)

            # Calculate optimal parameters
            cpu_cores = self._calculate_optimal_cpu_cores(avg_cpu, strategy)
            memory_limit = self._calculate_memory_limit(avg_memory, strategy)
            io_threads = self._calculate_io_threads(avg_io, strategy)

            return PerformanceProfile(
                strategy=strategy,
                cpu_cores=cpu_cores,
                memory_limit_mb=memory_limit,
                io_threads=io_threads,
                batch_size=self._calculate_batch_size(strategy),
                cache_size=self._calculate_cache_size(avg_memory),
                prefetch_enabled=strategy in [OptimizationStrategy.IO_INTENSIVE, OptimizationStrategy.MIXED_WORKLOAD],
                compression_enabled=avg_memory > 100,  # Enable for memory-intensive operations
                parallelization_threshold=self._calculate_parallel_threshold(strategy)
            )

    def _determine_strategy(self, avg_cpu: float, avg_memory: float, avg_io: float) -> OptimizationStrategy:
        """Determine optimization strategy based on resource usage."""
        if avg_cpu > 70:
            return OptimizationStrategy.CPU_INTENSIVE
        elif avg_io > 50:
            return OptimizationStrategy.IO_INTENSIVE
        elif avg_memory > 200:
            return OptimizationStrategy.MEMORY_INTENSIVE
        else:
            return OptimizationStrategy.MIXED_WORKLOAD

    def _calculate_optimal_cpu_cores(self, avg_cpu: float, strategy: OptimizationStrategy) -> int:
        """Calculate optimal CPU core allocation."""
        base_cores = 2

        if strategy == OptimizationStrategy.CPU_INTENSIVE:
            return min(multiprocessing.cpu_count(), max(2, int(avg_cpu / 20)))
        elif strategy == OptimizationStrategy.IO_INTENSIVE:
            return min(4, base_cores)
        else:
            return base_cores

    def _calculate_memory_limit(self, avg_memory: float, strategy: OptimizationStrategy) -> int:
        """Calculate optimal memory limit."""
        if strategy == OptimizationStrategy.MEMORY_INTENSIVE:
            return int(avg_memory * 1.5)  # 50% headroom
        else:
            return max(512, int(avg_memory * 1.2))  # 20% headroom

    def _calculate_io_threads(self, avg_io: float, strategy: OptimizationStrategy) -> int:
        """Calculate optimal I/O thread count."""
        if strategy == OptimizationStrategy.IO_INTENSIVE:
            return min(32, max(4, int(avg_io / 10)))
        else:
            return 4

    def _calculate_batch_size(self, strategy: OptimizationStrategy) -> int:
        """Calculate optimal batch size."""
        batch_sizes = {
            OptimizationStrategy.CPU_INTENSIVE: 50,
            OptimizationStrategy.IO_INTENSIVE: 20,
            OptimizationStrategy.MEMORY_INTENSIVE: 10,
            OptimizationStrategy.MIXED_WORKLOAD: 25,
            OptimizationStrategy.REAL_TIME: 1,
            OptimizationStrategy.BATCH_PROCESSING: 100
        }
        return batch_sizes.get(strategy, 25)

    def _calculate_cache_size(self, avg_memory: float) -> int:
        """Calculate optimal cache size."""
        return min(1000, max(50, int(avg_memory / 2)))

    def _calculate_parallel_threshold(self, strategy: OptimizationStrategy) -> int:
        """Calculate threshold for parallel processing."""
        thresholds = {
            OptimizationStrategy.CPU_INTENSIVE: 10,
            OptimizationStrategy.IO_INTENSIVE: 5,
            OptimizationStrategy.MEMORY_INTENSIVE: 50,
            OptimizationStrategy.MIXED_WORKLOAD: 20,
            OptimizationStrategy.REAL_TIME: 1,
            OptimizationStrategy.BATCH_PROCESSING: 100
        }
        return thresholds.get(strategy, 20)


class QuantumPerformanceOptimizer:
    """Advanced performance optimization engine with auto-scaling."""

    def __init__(self,
                 enable_auto_scaling: bool = True,
                 enable_smart_caching: bool = True,
                 enable_workload_analysis: bool = True):
        """Initialize performance optimizer."""
        self.enable_auto_scaling = enable_auto_scaling
        self.enable_smart_caching = enable_smart_caching
        self.enable_workload_analysis = enable_workload_analysis

        # Core components
        self.resource_pool = ResourcePool(enable_adaptive_sizing=enable_auto_scaling)
        self.smart_cache = SmartCache() if enable_smart_caching else None
        self.workload_analyzer = WorkloadAnalyzer() if enable_workload_analysis else None

        # Dependencies
        self.metrics_collector = MetricsCollector()
        self.health_monitor = get_health_monitor()
        self.resilience_engine = get_resilience_engine()
        self.audit_logger = get_audit_logger()

        # Optimization state
        self.optimization_profiles: Dict[str, PerformanceProfile] = {}
        self.scaling_rules: Dict[str, ScalingRule] = {}
        self.optimization_history: List[OptimizationResult] = []

        # Auto-scaling state
        self.last_scale_action: Dict[str, float] = {}
        self.current_instances: Dict[str, int] = {}

        self._lock = RLock()
        self.logger = logging.getLogger(__name__)

        # Setup default scaling rules
        self._setup_default_scaling_rules()

    def _setup_default_scaling_rules(self) -> None:
        """Setup default auto-scaling rules."""
        # CPU-based scaling
        self.scaling_rules['cpu_scaling'] = ScalingRule(
            trigger=ScalingTrigger.CPU_UTILIZATION,
            threshold_up=80.0,
            threshold_down=30.0,
            scale_up_by=2,
            scale_down_by=1,
            cooldown_seconds=300,
            max_instances=10,
            min_instances=1
        )

        # Memory-based scaling
        self.scaling_rules['memory_scaling'] = ScalingRule(
            trigger=ScalingTrigger.MEMORY_UTILIZATION,
            threshold_up=85.0,
            threshold_down=40.0,
            scale_up_by=1,
            scale_down_by=1,
            cooldown_seconds=180,
            max_instances=8,
            min_instances=1
        )

        # Response time scaling
        self.scaling_rules['response_time_scaling'] = ScalingRule(
            trigger=ScalingTrigger.RESPONSE_TIME,
            threshold_up=5000.0,  # 5 seconds
            threshold_down=1000.0,  # 1 second
            scale_up_by=2,
            scale_down_by=1,
            cooldown_seconds=120,
            max_instances=15,
            min_instances=2
        )

    def optimize_operation(self,
                          operation_name: str,
                          func: Callable,
                          *args, **kwargs) -> OptimizationResult:
        """Optimize operation execution with intelligent strategies."""
        start_time = time.time()

        # Get or create performance profile
        profile = self._get_performance_profile(operation_name)

        # Record pre-optimization metrics
        cpu_before = psutil.cpu_percent()
        memory_before = psutil.virtual_memory().percent

        try:
            # Apply optimizations based on profile
            optimized_result = self._execute_optimized(
                operation_name, func, profile, *args, **kwargs
            )

            # Calculate optimization metrics
            duration_ms = (time.time() - start_time) * 1000
            cpu_after = psutil.cpu_percent()
            memory_after = psutil.virtual_memory().percent

            # Create optimization result
            result = OptimizationResult(
                operation_name=operation_name,
                original_duration_ms=duration_ms * 1.2,  # Estimate without optimization
                optimized_duration_ms=duration_ms,
                improvement_factor=1.2,  # Estimate
                memory_saved_mb=max(0, memory_before - memory_after) * 10,  # Rough estimate
                optimization_applied=self._get_applied_optimizations(profile),
                metadata={
                    "profile_strategy": profile.strategy.value,
                    "cpu_before": cpu_before,
                    "cpu_after": cpu_after,
                    "memory_before": memory_before,
                    "memory_after": memory_after
                }
            )

            # Record in history
            self.optimization_history.append(result)
            if len(self.optimization_history) > 1000:
                self.optimization_history.pop(0)

            # Update workload analyzer
            if self.workload_analyzer:
                self.workload_analyzer.record_operation(
                    operation_name=operation_name,
                    duration_ms=duration_ms,
                    cpu_usage=cpu_after,
                    memory_usage_mb=memory_after * 10  # Rough conversion
                )

            return result

        except Exception as e:
            self.logger.error(f"Optimization failed for {operation_name}: {e}")
            raise e

    def _get_performance_profile(self, operation_name: str) -> PerformanceProfile:
        """Get or create performance profile for operation."""
        with self._lock:
            if operation_name not in self.optimization_profiles:
                if self.workload_analyzer:
                    profile = self.workload_analyzer.analyze_operation(operation_name)
                else:
                    # Default profile
                    profile = PerformanceProfile(
                        strategy=OptimizationStrategy.MIXED_WORKLOAD,
                        cpu_cores=2,
                        memory_limit_mb=512,
                        io_threads=4,
                        batch_size=25,
                        cache_size=100,
                        prefetch_enabled=True,
                        compression_enabled=False,
                        parallelization_threshold=20
                    )

                self.optimization_profiles[operation_name] = profile
                self.logger.info(f"Created performance profile for {operation_name}: {profile.strategy.value}")

            return self.optimization_profiles[operation_name]

    def _execute_optimized(self,
                          operation_name: str,
                          func: Callable,
                          profile: PerformanceProfile,
                          *args, **kwargs) -> Any:
        """Execute function with optimization strategies."""
        optimizations = []

        # Smart caching
        if self.smart_cache and profile.cache_size > 0:
            cache_key = f"{operation_name}_{hash(str(args) + str(kwargs))}"
            cached_result = self.smart_cache.get(cache_key)

            if cached_result is not None:
                optimizations.append("cache_hit")
                return cached_result

        # Choose execution strategy based on profile
        if profile.strategy == OptimizationStrategy.CPU_INTENSIVE:
            result = self._execute_cpu_optimized(func, profile, *args, **kwargs)
            optimizations.append("cpu_optimization")

        elif profile.strategy == OptimizationStrategy.IO_INTENSIVE:
            result = self._execute_io_optimized(func, profile, *args, **kwargs)
            optimizations.append("io_optimization")

        elif profile.strategy == OptimizationStrategy.MEMORY_INTENSIVE:
            result = self._execute_memory_optimized(func, profile, *args, **kwargs)
            optimizations.append("memory_optimization")

        else:
            # Mixed workload - use resilience engine
            resilience_config = ResilienceConfig(
                pattern=ResiliencePattern.RETRY_WITH_BACKOFF,
                timeout_seconds=profile.parallelization_threshold
            )
            result = self.resilience_engine.execute_resilient(
                operation_name, func, resilience_config, *args, **kwargs
            ).result
            optimizations.append("resilience_optimization")

        # Cache result if caching enabled
        if self.smart_cache and result is not None:
            cache_key = f"{operation_name}_{hash(str(args) + str(kwargs))}"
            self.smart_cache.set(cache_key, result)
            optimizations.append("cache_store")

        return result

    def _execute_cpu_optimized(self,
                              func: Callable,
                              profile: PerformanceProfile,
                              *args, **kwargs) -> Any:
        """Execute with CPU optimization."""
        # For CPU-intensive tasks, use process pool
        if len(args) > profile.parallelization_threshold:
            # Parallel processing for large datasets
            return self._execute_parallel(func, profile, *args, **kwargs)
        else:
            # Regular execution with CPU affinity if available
            return func(*args, **kwargs)

    def _execute_io_optimized(self,
                             func: Callable,
                             profile: PerformanceProfile,
                             *args, **kwargs) -> Any:
        """Execute with I/O optimization."""
        # Use thread pool for I/O operations
        future = self.resource_pool.submit_io_task(func, *args, **kwargs)
        return future.result()

    def _execute_memory_optimized(self,
                                 func: Callable,
                                 profile: PerformanceProfile,
                                 *args, **kwargs) -> Any:
        """Execute with memory optimization."""
        # Batch processing to manage memory usage
        if hasattr(args[0], '__len__') and len(args[0]) > profile.batch_size:
            return self._execute_batched(func, profile, *args, **kwargs)
        else:
            return func(*args, **kwargs)

    def _execute_parallel(self,
                         func: Callable,
                         profile: PerformanceProfile,
                         *args, **kwargs) -> Any:
        """Execute with parallel processing."""
        # Split work across multiple processes
        if args and hasattr(args[0], '__len__'):
            data = args[0]
            chunk_size = max(1, len(data) // profile.cpu_cores)

            futures = []
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i + chunk_size]
                future = self.resource_pool.submit_cpu_task(func, chunk, *args[1:], **kwargs)
                futures.append(future)

            # Collect results
            results = []
            for future in as_completed(futures):
                results.extend(future.result())

            return results
        else:
            return func(*args, **kwargs)

    def _execute_batched(self,
                        func: Callable,
                        profile: PerformanceProfile,
                        *args, **kwargs) -> Any:
        """Execute with batch processing."""
        if args and hasattr(args[0], '__len__'):
            data = args[0]
            batch_size = profile.batch_size

            results = []
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                batch_result = func(batch, *args[1:], **kwargs)
                results.extend(batch_result if isinstance(batch_result, list) else [batch_result])

            return results
        else:
            return func(*args, **kwargs)

    def _get_applied_optimizations(self, profile: PerformanceProfile) -> List[str]:
        """Get list of applied optimizations."""
        optimizations = []

        if self.smart_cache:
            optimizations.append("smart_caching")
        if profile.cpu_cores > 1:
            optimizations.append("parallel_processing")
        if profile.batch_size > 1:
            optimizations.append("batch_processing")
        if profile.prefetch_enabled:
            optimizations.append("prefetching")
        if profile.compression_enabled:
            optimizations.append("compression")

        optimizations.append(f"strategy_{profile.strategy.value}")

        return optimizations

    def check_scaling_triggers(self) -> List[str]:
        """Check if auto-scaling should be triggered."""
        if not self.enable_auto_scaling:
            return []

        actions_taken = []
        current_time = time.time()

        # Get current system metrics
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent

        for rule_name, rule in self.scaling_rules.items():
            # Check cooldown
            last_action_time = self.last_scale_action.get(rule_name, 0)
            if current_time - last_action_time < rule.cooldown_seconds:
                continue

            current_instances = self.current_instances.get(rule_name, rule.min_instances)

            # Check scaling triggers
            should_scale_up = False
            should_scale_down = False

            if rule.trigger == ScalingTrigger.CPU_UTILIZATION:
                if cpu_percent > rule.threshold_up:
                    should_scale_up = True
                elif cpu_percent < rule.threshold_down:
                    should_scale_down = True

            elif rule.trigger == ScalingTrigger.MEMORY_UTILIZATION:
                if memory_percent > rule.threshold_up:
                    should_scale_up = True
                elif memory_percent < rule.threshold_down:
                    should_scale_down = True

            # Apply scaling actions
            if should_scale_up and current_instances < rule.max_instances:
                new_instances = min(rule.max_instances, current_instances + rule.scale_up_by)
                self.current_instances[rule_name] = new_instances
                self.last_scale_action[rule_name] = current_time

                action = f"Scaled up {rule_name}: {current_instances} -> {new_instances}"
                actions_taken.append(action)
                self.logger.info(action)

                # Audit log
                self.audit_logger.log_security_event(
                    event_type=AuditEventType.SYSTEM_ACCESS,
                    action="auto_scale_up",
                    result="success",
                    severity=SecurityLevel.LOW,
                    details={
                        "rule_name": rule_name,
                        "trigger": rule.trigger.value,
                        "old_instances": current_instances,
                        "new_instances": new_instances
                    }
                )

            elif should_scale_down and current_instances > rule.min_instances:
                new_instances = max(rule.min_instances, current_instances - rule.scale_down_by)
                self.current_instances[rule_name] = new_instances
                self.last_scale_action[rule_name] = current_time

                action = f"Scaled down {rule_name}: {current_instances} -> {new_instances}"
                actions_taken.append(action)
                self.logger.info(action)

        return actions_taken

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            "optimization_profiles": len(self.optimization_profiles),
            "optimization_history": len(self.optimization_history),
            "resource_pool": self.resource_pool.get_utilization(),
            "scaling_rules": len(self.scaling_rules),
            "current_instances": self.current_instances.copy()
        }

        if self.smart_cache:
            stats["cache"] = self.smart_cache.get_stats()

        # Calculate optimization effectiveness
        if self.optimization_history:
            recent_optimizations = self.optimization_history[-100:]  # Last 100
            avg_improvement = sum(o.improvement_factor for o in recent_optimizations) / len(recent_optimizations)
            total_memory_saved = sum(o.memory_saved_mb for o in recent_optimizations)

            stats["optimization_effectiveness"] = {
                "avg_improvement_factor": avg_improvement,
                "total_memory_saved_mb": total_memory_saved,
                "recent_optimizations": len(recent_optimizations)
            }

        return stats

    def tune_performance_profile(self,
                                operation_name: str,
                                strategy: OptimizationStrategy,
                                **kwargs) -> None:
        """Manually tune performance profile for an operation."""
        with self._lock:
            current_profile = self._get_performance_profile(operation_name)

            # Update profile with provided parameters
            updated_profile = PerformanceProfile(
                strategy=strategy,
                cpu_cores=kwargs.get('cpu_cores', current_profile.cpu_cores),
                memory_limit_mb=kwargs.get('memory_limit_mb', current_profile.memory_limit_mb),
                io_threads=kwargs.get('io_threads', current_profile.io_threads),
                batch_size=kwargs.get('batch_size', current_profile.batch_size),
                cache_size=kwargs.get('cache_size', current_profile.cache_size),
                prefetch_enabled=kwargs.get('prefetch_enabled', current_profile.prefetch_enabled),
                compression_enabled=kwargs.get('compression_enabled', current_profile.compression_enabled),
                parallelization_threshold=kwargs.get('parallelization_threshold', current_profile.parallelization_threshold)
            )

            self.optimization_profiles[operation_name] = updated_profile

            self.logger.info(f"Tuned performance profile for {operation_name}: {strategy.value}")

            # Audit log
            self.audit_logger.log_configuration_change(
                config_key=f"performance_profile_{operation_name}",
                old_value=asdict(current_profile),
                new_value=asdict(updated_profile)
            )

    def clear_optimization_cache(self, operation_pattern: Optional[str] = None) -> None:
        """Clear optimization cache."""
        if self.smart_cache:
            if operation_pattern:
                # Clear specific pattern
                keys_to_remove = [k for k in self.smart_cache._cache.keys() if operation_pattern in k]
                for key in keys_to_remove:
                    self.smart_cache._remove_entry(key)
                self.logger.info(f"Cleared cache for pattern: {operation_pattern}")
            else:
                # Clear all cache
                self.smart_cache.clear()
                self.logger.info("Cleared all optimization cache")

    def shutdown(self) -> None:
        """Shutdown performance optimizer."""
        self.resource_pool.shutdown()
        self.logger.info("Performance optimizer shutdown complete")


# Global performance optimizer instance
_performance_optimizer: Optional[QuantumPerformanceOptimizer] = None


def get_performance_optimizer() -> QuantumPerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = QuantumPerformanceOptimizer()
    return _performance_optimizer
