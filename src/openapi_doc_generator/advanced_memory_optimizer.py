"""
Advanced Memory Optimization System

This module provides comprehensive memory management and optimization capabilities
including intelligent garbage collection tuning, memory leak detection, object
pooling, and advanced memory analytics for optimal performance.

Features:
- Intelligent garbage collection optimization
- Memory leak detection and prevention
- Advanced object pooling strategies
- Memory usage analytics and profiling
- Automatic memory pressure handling
- Weak reference management
- Memory-efficient data structures
- Smart caching with memory constraints
"""

import asyncio
import gc
import logging
import mmap
import os
import pickle
import sys
import threading
import time
import tracemalloc
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Type, WeakSet
import psutil

from .enhanced_monitoring import get_monitor

logger = logging.getLogger(__name__)


class MemoryStrategy(Enum):
    """Memory optimization strategies."""
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"
    ADAPTIVE = "adaptive"


class GCStrategy(Enum):
    """Garbage collection strategies."""
    DEFAULT = "default"
    PERFORMANCE = "performance"
    MEMORY_OPTIMIZED = "memory_optimized"
    LATENCY_OPTIMIZED = "latency_optimized"
    CUSTOM = "custom"


class MemoryPressureLevel(Enum):
    """Memory pressure levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MemoryMetrics:
    """Comprehensive memory metrics."""
    # System memory
    total_system_memory_gb: float
    available_system_memory_gb: float
    system_memory_percent: float
    
    # Process memory
    process_memory_mb: float
    process_memory_percent: float
    peak_memory_mb: float
    
    # Python-specific memory
    python_objects_count: int
    python_object_size_mb: float
    
    # Garbage collection
    gc_collections: Tuple[int, int, int]
    gc_collected: Tuple[int, int, int]
    gc_uncollectable: Tuple[int, int, int]
    
    # Memory tracking
    tracked_allocations_mb: float
    memory_leaks_detected: int
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MemoryLeak:
    """Detected memory leak information."""
    leak_id: str
    object_type: str
    allocation_site: str
    growth_rate_mb_per_hour: float
    total_leaked_mb: float
    first_detected: datetime
    last_updated: datetime
    severity: str  # "low", "medium", "high", "critical"
    stack_trace: List[str]


@dataclass
class GCConfiguration:
    """Garbage collection configuration."""
    strategy: GCStrategy
    generation_0_threshold: int
    generation_1_threshold: int
    generation_2_threshold: int
    
    # Custom settings
    enable_debug: bool = False
    track_stats: bool = True
    automatic_tuning: bool = True


class SmartObjectPool:
    """Intelligent object pool with automatic sizing and cleanup."""
    
    def __init__(self, 
                 object_factory: callable,
                 reset_func: Optional[callable] = None,
                 max_size: int = 100,
                 min_size: int = 5,
                 auto_cleanup: bool = True):
        
        self.object_factory = object_factory
        self.reset_func = reset_func or (lambda obj: None)
        self.max_size = max_size
        self.min_size = min_size
        self.auto_cleanup = auto_cleanup
        
        # Pool management
        self.available_objects: deque = deque()
        self.in_use_objects: WeakSet = weakref.WeakSet()
        self.total_created = 0
        self.total_reused = 0
        
        # Performance tracking
        self.hit_rate = 0.0
        self.last_cleanup = datetime.now()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Pre-populate pool
        self._initialize_pool()
        
        logger.debug(f"Object pool created for {object_factory.__name__}")
    
    def _initialize_pool(self):
        """Initialize pool with minimum objects."""
        with self.lock:
            for _ in range(self.min_size):
                try:
                    obj = self.object_factory()
                    self.available_objects.append(obj)
                    self.total_created += 1
                except Exception as e:
                    logger.warning(f"Failed to pre-create object: {e}")
                    break
    
    def acquire(self) -> Any:
        """Acquire an object from the pool."""
        with self.lock:
            if self.available_objects:
                obj = self.available_objects.popleft()
                self.total_reused += 1
            else:
                # Create new object if pool is empty
                try:
                    obj = self.object_factory()
                    self.total_created += 1
                except Exception as e:
                    logger.error(f"Failed to create new object: {e}")
                    raise
            
            self.in_use_objects.add(obj)
            self._update_hit_rate()
            
            return obj
    
    def release(self, obj: Any):
        """Release an object back to the pool."""
        with self.lock:
            try:
                # Reset object state
                self.reset_func(obj)
                
                # Add back to pool if not full
                if len(self.available_objects) < self.max_size:
                    self.available_objects.append(obj)
                
                # Remove from in-use tracking
                self.in_use_objects.discard(obj)
                
            except Exception as e:
                logger.warning(f"Failed to reset object: {e}")
                # Don't return potentially corrupted object to pool
    
    def _update_hit_rate(self):
        """Update pool hit rate statistics."""
        total_requests = self.total_created + self.total_reused
        self.hit_rate = (self.total_reused / total_requests * 100) if total_requests > 0 else 0
    
    def cleanup(self):
        """Cleanup unused objects from pool."""
        if not self.auto_cleanup:
            return
        
        with self.lock:
            # Keep only minimum objects
            while len(self.available_objects) > self.min_size:
                self.available_objects.popleft()
            
            self.last_cleanup = datetime.now()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self.lock:
            return {
                'available_objects': len(self.available_objects),
                'in_use_objects': len(self.in_use_objects),
                'total_created': self.total_created,
                'total_reused': self.total_reused,
                'hit_rate': self.hit_rate,
                'max_size': self.max_size,
                'min_size': self.min_size
            }


class MemoryEfficientCache:
    """Memory-efficient cache with automatic cleanup."""
    
    def __init__(self, 
                 max_memory_mb: float = 100.0,
                 max_items: int = 10000,
                 ttl_seconds: int = 3600):
        
        self.max_memory_mb = max_memory_mb
        self.max_items = max_items
        self.ttl_seconds = ttl_seconds
        
        # Cache storage
        self.cache: Dict[str, Tuple[Any, datetime, int]] = {}  # value, timestamp, size
        self.access_order: deque = deque()
        self.current_memory_mb = 0.0
        
        # Memory management
        self.memory_tracker = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.debug(f"Memory-efficient cache created: {max_memory_mb}MB limit")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key not in self.cache:
                return None
            
            value, timestamp, size = self.cache[key]
            
            # Check TTL
            if datetime.now() - timestamp > timedelta(seconds=self.ttl_seconds):
                self._remove_item(key)
                return None
            
            # Update access order
            self._update_access_order(key)
            
            return value
    
    def put(self, key: str, value: Any) -> bool:
        """Put item in cache."""
        with self.lock:
            # Calculate size
            size_bytes = sys.getsizeof(value)
            size_mb = size_bytes / (1024 * 1024)
            
            # Check if item would exceed memory limit
            if size_mb > self.max_memory_mb:
                logger.warning(f"Item too large for cache: {size_mb:.2f}MB")
                return False
            
            # Make space if necessary
            while (self.current_memory_mb + size_mb > self.max_memory_mb or 
                   len(self.cache) >= self.max_items):
                if not self._evict_lru():
                    break
            
            # Store item
            self.cache[key] = (value, datetime.now(), size_bytes)
            self.current_memory_mb += size_mb
            self._update_access_order(key)
            
            return True
    
    def _update_access_order(self, key: str):
        """Update access order for LRU."""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def _evict_lru(self) -> bool:
        """Evict least recently used item."""
        if not self.access_order:
            return False
        
        lru_key = self.access_order.popleft()
        self._remove_item(lru_key)
        return True
    
    def _remove_item(self, key: str):
        """Remove item from cache."""
        if key in self.cache:
            _, _, size_bytes = self.cache[key]
            size_mb = size_bytes / (1024 * 1024)
            
            del self.cache[key]
            self.current_memory_mb -= size_mb
            
            if key in self.access_order:
                self.access_order.remove(key)
    
    def cleanup_expired(self):
        """Cleanup expired items."""
        with self.lock:
            current_time = datetime.now()
            expired_keys = []
            
            for key, (value, timestamp, size) in self.cache.items():
                if current_time - timestamp > timedelta(seconds=self.ttl_seconds):
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_item(key)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        return {
            'current_memory_mb': self.current_memory_mb,
            'max_memory_mb': self.max_memory_mb,
            'utilization_percent': (self.current_memory_mb / self.max_memory_mb) * 100,
            'item_count': len(self.cache),
            'max_items': self.max_items
        }


class GarbageCollectionOptimizer:
    """Optimizes garbage collection settings for performance."""
    
    def __init__(self):
        self.current_config = self._get_current_gc_config()
        self.baseline_performance = None
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.gc_times: deque = deque(maxlen=100)
        self.collection_counts = [0, 0, 0]  # Track collections per generation
        
        logger.info("Garbage Collection Optimizer initialized")
    
    def _get_current_gc_config(self) -> GCConfiguration:
        """Get current garbage collection configuration."""
        thresholds = gc.get_threshold()
        
        return GCConfiguration(
            strategy=GCStrategy.DEFAULT,
            generation_0_threshold=thresholds[0],
            generation_1_threshold=thresholds[1],
            generation_2_threshold=thresholds[2],
            track_stats=True,
            automatic_tuning=False
        )
    
    def optimize_for_performance(self):
        """Optimize GC for performance (lower frequency, higher latency)."""
        config = GCConfiguration(
            strategy=GCStrategy.PERFORMANCE,
            generation_0_threshold=2000,  # Increased from default ~700
            generation_1_threshold=15,    # Increased from default ~10
            generation_2_threshold=15,    # Increased from default ~10
            automatic_tuning=True
        )
        
        self._apply_configuration(config)
        logger.info("GC optimized for performance")
    
    def optimize_for_memory(self):
        """Optimize GC for memory usage (higher frequency, lower latency)."""
        config = GCConfiguration(
            strategy=GCStrategy.MEMORY_OPTIMIZED,
            generation_0_threshold=400,   # Decreased from default ~700
            generation_1_threshold=5,     # Decreased from default ~10
            generation_2_threshold=5,     # Decreased from default ~10
            automatic_tuning=True
        )
        
        self._apply_configuration(config)
        logger.info("GC optimized for memory usage")
    
    def optimize_for_latency(self):
        """Optimize GC for low latency (balanced approach)."""
        config = GCConfiguration(
            strategy=GCStrategy.LATENCY_OPTIMIZED,
            generation_0_threshold=1000,  # Moderate frequency
            generation_1_threshold=8,     # Moderate frequency
            generation_2_threshold=8,     # Moderate frequency
            automatic_tuning=True
        )
        
        self._apply_configuration(config)
        logger.info("GC optimized for latency")
    
    def _apply_configuration(self, config: GCConfiguration):
        """Apply garbage collection configuration."""
        gc.set_threshold(
            config.generation_0_threshold,
            config.generation_1_threshold,
            config.generation_2_threshold
        )
        
        if config.enable_debug:
            gc.set_debug(gc.DEBUG_STATS)
        else:
            gc.set_debug(0)
        
        self.current_config = config
    
    def analyze_gc_performance(self) -> Dict[str, Any]:
        """Analyze garbage collection performance."""
        stats = gc.get_stats()
        
        analysis = {
            'current_config': {
                'strategy': self.current_config.strategy.value,
                'thresholds': gc.get_threshold()
            },
            'generation_stats': [],
            'performance_metrics': {
                'average_gc_time': sum(self.gc_times) / len(self.gc_times) if self.gc_times else 0,
                'total_collections': sum(stat['collections'] for stat in stats),
                'total_collected': sum(stat['collected'] for stat in stats),
                'total_uncollectable': sum(stat['uncollectable'] for stat in stats)
            },
            'recommendations': []
        }
        
        # Analyze each generation
        for i, stat in enumerate(stats):
            generation_info = {
                'generation': i,
                'collections': stat['collections'],
                'collected': stat['collected'],
                'uncollectable': stat['uncollectable'],
                'collection_rate': stat['collections'] / max(1, time.time() - (gc.get_stats()[0]['collections'] * 0.1))
            }
            analysis['generation_stats'].append(generation_info)
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_gc_recommendations(analysis)
        
        return analysis
    
    def _generate_gc_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate GC optimization recommendations."""
        recommendations = []
        
        perf_metrics = analysis['performance_metrics']
        
        # High collection rate recommendations
        total_collections = perf_metrics['total_collections']
        if total_collections > 1000:  # Arbitrary threshold
            recommendations.append("Consider optimizing for performance to reduce GC frequency")
        
        # Memory leak detection
        if perf_metrics['total_uncollectable'] > 100:
            recommendations.append("High number of uncollectable objects detected - possible memory leak")
        
        # Generation-specific recommendations
        gen_stats = analysis['generation_stats']
        
        if len(gen_stats) > 0 and gen_stats[0]['collection_rate'] > 10:  # Gen 0 collecting very frequently
            recommendations.append("Generation 0 collecting frequently - consider increasing threshold")
        
        if len(gen_stats) > 2 and gen_stats[2]['collections'] == 0:  # Gen 2 never collecting
            recommendations.append("Generation 2 not collecting - may indicate threshold too high")
        
        return recommendations
    
    def enable_automatic_tuning(self):
        """Enable automatic GC tuning based on performance metrics."""
        self.current_config.automatic_tuning = True
        # Implementation would monitor performance and adjust thresholds
        logger.info("Automatic GC tuning enabled")


class MemoryLeakDetector:
    """Detects and analyzes memory leaks."""
    
    def __init__(self):
        self.snapshots: List[tracemalloc.Snapshot] = []
        self.leak_candidates: Dict[str, MemoryLeak] = {}
        
        # Start memory tracking
        if not tracemalloc.is_tracing():
            tracemalloc.start(25)  # Track up to 25 frames
        
        # Baseline snapshot
        self.baseline_snapshot = tracemalloc.take_snapshot()
        self.last_analysis = datetime.now()
        
        logger.info("Memory leak detector initialized")
    
    def take_snapshot(self) -> str:
        """Take a memory snapshot and return snapshot ID."""
        snapshot = tracemalloc.take_snapshot()
        snapshot_id = f"snapshot_{len(self.snapshots)}_{int(time.time())}"
        
        self.snapshots.append(snapshot)
        
        # Keep only recent snapshots to manage memory
        if len(self.snapshots) > 20:
            self.snapshots.pop(0)
        
        return snapshot_id
    
    def analyze_memory_growth(self) -> List[MemoryLeak]:
        """Analyze memory growth and detect potential leaks."""
        if len(self.snapshots) < 2:
            return []
        
        current_snapshot = self.snapshots[-1]
        previous_snapshot = self.snapshots[-2]
        
        # Compare snapshots
        top_stats = current_snapshot.compare_to(previous_snapshot, 'lineno')
        
        detected_leaks = []
        
        for stat in top_stats[:20]:  # Top 20 memory growth items
            if stat.size_diff > 1024 * 1024:  # Growth > 1MB
                
                # Extract location information
                frame = stat.traceback[-1] if stat.traceback else None
                location = f"{frame.filename}:{frame.lineno}" if frame else "unknown"
                
                # Calculate growth rate
                time_diff = (datetime.now() - self.last_analysis).total_seconds() / 3600.0  # Hours
                growth_rate = (stat.size_diff / (1024 * 1024)) / max(time_diff, 0.001)  # MB/hour
                
                # Check if this is a new leak or update existing
                leak_id = f"leak_{hash(location)}"
                
                if leak_id in self.leak_candidates:
                    # Update existing leak
                    leak = self.leak_candidates[leak_id]
                    leak.total_leaked_mb += stat.size_diff / (1024 * 1024)
                    leak.growth_rate_mb_per_hour = growth_rate
                    leak.last_updated = datetime.now()
                else:
                    # New leak candidate
                    leak = MemoryLeak(
                        leak_id=leak_id,
                        object_type="unknown",
                        allocation_site=location,
                        growth_rate_mb_per_hour=growth_rate,
                        total_leaked_mb=stat.size_diff / (1024 * 1024),
                        first_detected=datetime.now(),
                        last_updated=datetime.now(),
                        severity=self._classify_leak_severity(growth_rate),
                        stack_trace=[str(frame) for frame in stat.traceback] if stat.traceback else []
                    )
                    
                    self.leak_candidates[leak_id] = leak
                
                detected_leaks.append(self.leak_candidates[leak_id])
        
        self.last_analysis = datetime.now()
        return detected_leaks
    
    def _classify_leak_severity(self, growth_rate_mb_per_hour: float) -> str:
        """Classify leak severity based on growth rate."""
        if growth_rate_mb_per_hour > 100:
            return "critical"
        elif growth_rate_mb_per_hour > 50:
            return "high"
        elif growth_rate_mb_per_hour > 10:
            return "medium"
        else:
            return "low"
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory usage summary."""
        if not self.snapshots:
            return {}
        
        current_snapshot = self.snapshots[-1]
        top_stats = current_snapshot.statistics('lineno')
        
        total_size = sum(stat.size for stat in top_stats)
        total_count = sum(stat.count for stat in top_stats)
        
        return {
            'total_memory_mb': total_size / (1024 * 1024),
            'total_allocations': total_count,
            'top_allocations': [
                {
                    'location': stat.traceback.format()[-1] if stat.traceback else 'unknown',
                    'size_mb': stat.size / (1024 * 1024),
                    'count': stat.count
                }
                for stat in top_stats[:10]
            ],
            'active_leaks': len(self.leak_candidates),
            'snapshots_taken': len(self.snapshots)
        }


class MemoryPressureHandler:
    """Handles memory pressure situations."""
    
    def __init__(self, 
                 warning_threshold: float = 0.8,  # 80% memory usage
                 critical_threshold: float = 0.9,  # 90% memory usage
                 emergency_threshold: float = 0.95):  # 95% memory usage
        
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.emergency_threshold = emergency_threshold
        
        self.current_pressure_level = MemoryPressureLevel.LOW
        self.pressure_callbacks: List[callable] = []
        
        # Memory management strategies
        self.object_pools: List[SmartObjectPool] = []
        self.caches: List[MemoryEfficientCache] = []
        
        logger.info("Memory pressure handler initialized")
    
    def register_pressure_callback(self, callback: callable):
        """Register callback for memory pressure events."""
        self.pressure_callbacks.append(callback)
    
    def register_object_pool(self, pool: SmartObjectPool):
        """Register object pool for pressure management."""
        self.object_pools.append(pool)
    
    def register_cache(self, cache: MemoryEfficientCache):
        """Register cache for pressure management."""
        self.caches.append(cache)
    
    def check_memory_pressure(self) -> MemoryPressureLevel:
        """Check current memory pressure level."""
        memory = psutil.virtual_memory()
        memory_percent = memory.percent / 100.0
        
        previous_level = self.current_pressure_level
        
        if memory_percent >= self.emergency_threshold:
            self.current_pressure_level = MemoryPressureLevel.CRITICAL
        elif memory_percent >= self.critical_threshold:
            self.current_pressure_level = MemoryPressureLevel.HIGH
        elif memory_percent >= self.warning_threshold:
            self.current_pressure_level = MemoryPressureLevel.MODERATE
        else:
            self.current_pressure_level = MemoryPressureLevel.LOW
        
        # Trigger callbacks if pressure level changed
        if self.current_pressure_level != previous_level:
            self._handle_pressure_change(previous_level, self.current_pressure_level)
        
        return self.current_pressure_level
    
    def _handle_pressure_change(self, 
                               old_level: MemoryPressureLevel,
                               new_level: MemoryPressureLevel):
        """Handle memory pressure level changes."""
        logger.info(f"Memory pressure changed: {old_level.value} -> {new_level.value}")
        
        # Take appropriate action based on new pressure level
        if new_level == MemoryPressureLevel.MODERATE:
            self._handle_moderate_pressure()
        elif new_level == MemoryPressureLevel.HIGH:
            self._handle_high_pressure()
        elif new_level == MemoryPressureLevel.CRITICAL:
            self._handle_critical_pressure()
        
        # Notify callbacks
        for callback in self.pressure_callbacks:
            try:
                callback(old_level, new_level)
            except Exception as e:
                logger.error(f"Memory pressure callback failed: {e}")
    
    def _handle_moderate_pressure(self):
        """Handle moderate memory pressure."""
        # Clean up caches
        for cache in self.caches:
            cache.cleanup_expired()
        
        # Run garbage collection
        gc.collect()
    
    def _handle_high_pressure(self):
        """Handle high memory pressure."""
        # More aggressive cache cleanup
        for cache in self.caches:
            cache.cleanup_expired()
            # Also clear some non-expired items
            if hasattr(cache, '_evict_lru'):
                for _ in range(10):  # Remove 10 LRU items
                    if not cache._evict_lru():
                        break
        
        # Clean up object pools
        for pool in self.object_pools:
            pool.cleanup()
        
        # Force garbage collection
        gc.collect()
        
        logger.warning("High memory pressure: performed aggressive cleanup")
    
    def _handle_critical_pressure(self):
        """Handle critical memory pressure."""
        logger.critical("Critical memory pressure: emergency cleanup initiated")
        
        # Emergency cache clearing
        for cache in self.caches:
            if hasattr(cache, 'clear'):
                cache.clear()
        
        # Emergency pool cleanup
        for pool in self.object_pools:
            pool.cleanup()
        
        # Multiple GC passes
        for _ in range(3):
            gc.collect()
        
        # Clear weak references
        import weakref
        try:
            weakref.finalize_cache()
        except:
            pass


class AdvancedMemoryOptimizer:
    """
    Main memory optimization system coordinating all memory management components.
    """
    
    def __init__(self, 
                 strategy: MemoryStrategy = MemoryStrategy.ADAPTIVE,
                 max_cache_memory_mb: float = 200.0,
                 gc_strategy: GCStrategy = GCStrategy.BALANCED):
        
        self.strategy = strategy
        self.max_cache_memory_mb = max_cache_memory_mb
        
        # Core components
        self.gc_optimizer = GarbageCollectionOptimizer()
        self.leak_detector = MemoryLeakDetector()
        self.pressure_handler = MemoryPressureHandler()
        
        # Memory management
        self.object_pools: Dict[str, SmartObjectPool] = {}
        self.caches: Dict[str, MemoryEfficientCache] = {}
        
        # Monitoring
        self.monitor = get_monitor()
        self.metrics_history: deque = deque(maxlen=1000)
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.optimization_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Configure initial GC strategy
        self._configure_gc_strategy(gc_strategy)
        
        logger.info(f"Advanced Memory Optimizer initialized with {strategy.value} strategy")
    
    async def start(self):
        """Start memory optimization system."""
        if self.running:
            return
        
        self.running = True
        
        # Start background tasks
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.optimization_task = asyncio.create_task(self._optimization_loop())
        
        # Configure automatic tuning based on strategy
        if self.strategy == MemoryStrategy.ADAPTIVE:
            self.gc_optimizer.enable_automatic_tuning()
        
        logger.info("Memory optimization system started")
    
    async def stop(self):
        """Stop memory optimization system."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel background tasks
        tasks = [self.monitoring_task, self.optimization_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("Memory optimization system stopped")
    
    def create_object_pool(self, 
                          name: str,
                          object_factory: callable,
                          reset_func: Optional[callable] = None,
                          max_size: int = 100,
                          min_size: int = 5) -> SmartObjectPool:
        """Create and register an object pool."""
        pool = SmartObjectPool(
            object_factory=object_factory,
            reset_func=reset_func,
            max_size=max_size,
            min_size=min_size
        )
        
        self.object_pools[name] = pool
        self.pressure_handler.register_object_pool(pool)
        
        logger.info(f"Object pool '{name}' created")
        return pool
    
    def create_cache(self,
                    name: str,
                    max_memory_mb: float = 50.0,
                    max_items: int = 5000,
                    ttl_seconds: int = 3600) -> MemoryEfficientCache:
        """Create and register a memory-efficient cache."""
        cache = MemoryEfficientCache(
            max_memory_mb=max_memory_mb,
            max_items=max_items,
            ttl_seconds=ttl_seconds
        )
        
        self.caches[name] = cache
        self.pressure_handler.register_cache(cache)
        
        logger.info(f"Memory-efficient cache '{name}' created")
        return cache
    
    def _configure_gc_strategy(self, strategy: GCStrategy):
        """Configure garbage collection strategy."""
        if strategy == GCStrategy.PERFORMANCE:
            self.gc_optimizer.optimize_for_performance()
        elif strategy == GCStrategy.MEMORY_OPTIMIZED:
            self.gc_optimizer.optimize_for_memory()
        elif strategy == GCStrategy.LATENCY_OPTIMIZED:
            self.gc_optimizer.optimize_for_latency()
        # DEFAULT strategy uses system defaults
    
    async def optimize_memory(self) -> Dict[str, Any]:
        """Perform comprehensive memory optimization."""
        optimization_start = time.time()
        
        # Collect current metrics
        metrics = await self._collect_memory_metrics()
        
        # Check memory pressure
        pressure_level = self.pressure_handler.check_memory_pressure()
        
        # Detect memory leaks
        self.leak_detector.take_snapshot()
        detected_leaks = self.leak_detector.analyze_memory_growth()
        
        # Analyze GC performance
        gc_analysis = self.gc_optimizer.analyze_gc_performance()
        
        # Perform optimization based on strategy
        optimization_actions = []
        
        if self.strategy == MemoryStrategy.AGGRESSIVE:
            optimization_actions.extend(await self._aggressive_optimization())
        elif self.strategy == MemoryStrategy.CONSERVATIVE:
            optimization_actions.extend(await self._conservative_optimization())
        elif self.strategy == MemoryStrategy.ADAPTIVE:
            optimization_actions.extend(await self._adaptive_optimization(metrics, pressure_level))
        else:  # BALANCED
            optimization_actions.extend(await self._balanced_optimization())
        
        # Calculate optimization results
        post_metrics = await self._collect_memory_metrics()
        optimization_time = time.time() - optimization_start
        
        memory_freed = (metrics.process_memory_mb - post_metrics.process_memory_mb)
        
        result = {
            'optimization_time': optimization_time,
            'memory_freed_mb': memory_freed,
            'pressure_level': pressure_level.value,
            'detected_leaks': len(detected_leaks),
            'gc_analysis': gc_analysis,
            'optimization_actions': optimization_actions,
            'before_metrics': metrics,
            'after_metrics': post_metrics
        }
        
        # Record metrics
        self.monitor.record_metric("memory_optimization_freed_mb", memory_freed, "gauge")
        self.monitor.record_metric("memory_optimization_time", optimization_time, "timer")
        
        return result
    
    async def _aggressive_optimization(self) -> List[str]:
        """Perform aggressive memory optimization."""
        actions = []
        
        # Aggressive cache cleanup
        for name, cache in self.caches.items():
            if hasattr(cache, 'cache'):
                initial_size = len(cache.cache)
                cache.cleanup_expired()
                # Also clear 50% of remaining items
                items_to_clear = len(cache.cache) // 2
                for _ in range(items_to_clear):
                    if not cache._evict_lru():
                        break
                
                actions.append(f"Aggressively cleaned cache '{name}': {initial_size} -> {len(cache.cache)} items")
        
        # Aggressive object pool cleanup
        for name, pool in self.object_pools.items():
            pool.cleanup()
            actions.append(f"Cleaned object pool '{name}'")
        
        # Force multiple GC passes
        for i in range(3):
            collected = gc.collect()
            actions.append(f"GC pass {i+1}: collected {collected} objects")
        
        actions.append("Performed aggressive memory optimization")
        return actions
    
    async def _conservative_optimization(self) -> List[str]:
        """Perform conservative memory optimization."""
        actions = []
        
        # Only clean expired items
        for name, cache in self.caches.items():
            cache.cleanup_expired()
            actions.append(f"Cleaned expired items from cache '{name}'")
        
        # Single GC pass
        collected = gc.collect()
        actions.append(f"Garbage collection: collected {collected} objects")
        
        actions.append("Performed conservative memory optimization")
        return actions
    
    async def _adaptive_optimization(self, 
                                   metrics: MemoryMetrics,
                                   pressure_level: MemoryPressureLevel) -> List[str]:
        """Perform adaptive optimization based on current conditions."""
        actions = []
        
        # Adapt optimization intensity based on pressure level
        if pressure_level in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]:
            actions.extend(await self._aggressive_optimization())
        elif pressure_level == MemoryPressureLevel.MODERATE:
            actions.extend(await self._balanced_optimization())
        else:
            actions.extend(await self._conservative_optimization())
        
        # Adaptive GC tuning based on performance
        if metrics.process_memory_mb > 500:  # High memory usage
            self.gc_optimizer.optimize_for_memory()
            actions.append("Adapted GC for memory optimization")
        elif len(detected_leaks := self.leak_detector.leak_candidates) > 5:
            self.gc_optimizer.optimize_for_memory()
            actions.append(f"Adapted GC due to {len(detected_leaks)} potential leaks")
        
        actions.append("Performed adaptive memory optimization")
        return actions
    
    async def _balanced_optimization(self) -> List[str]:
        """Perform balanced memory optimization."""
        actions = []
        
        # Clean expired cache items and some LRU items
        for name, cache in self.caches.items():
            initial_items = len(cache.cache) if hasattr(cache, 'cache') else 0
            cache.cleanup_expired()
            
            # Clean 25% of remaining items if cache is over 75% full
            if hasattr(cache, 'cache') and len(cache.cache) > cache.max_items * 0.75:
                items_to_clear = len(cache.cache) // 4
                for _ in range(items_to_clear):
                    if not cache._evict_lru():
                        break
            
            final_items = len(cache.cache) if hasattr(cache, 'cache') else 0
            actions.append(f"Cleaned cache '{name}': {initial_items} -> {final_items} items")
        
        # Moderate object pool cleanup
        for name, pool in self.object_pools.items():
            if (datetime.now() - pool.last_cleanup).total_seconds() > 300:  # 5 minutes
                pool.cleanup()
                actions.append(f"Cleaned object pool '{name}'")
        
        # Two GC passes
        for i in range(2):
            collected = gc.collect()
            actions.append(f"GC pass {i+1}: collected {collected} objects")
        
        actions.append("Performed balanced memory optimization")
        return actions
    
    async def _collect_memory_metrics(self) -> MemoryMetrics:
        """Collect comprehensive memory metrics."""
        # System memory
        system_memory = psutil.virtual_memory()
        
        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info()
        
        # Python objects
        python_objects = len(gc.get_objects())
        
        # GC stats
        gc_stats = gc.get_stats()
        gc_collections = tuple(stat['collections'] for stat in gc_stats)
        gc_collected = tuple(stat['collected'] for stat in gc_stats)
        gc_uncollectable = tuple(stat['uncollectable'] for stat in gc_stats)
        
        # Memory tracking
        tracked_memory = 0.0
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            tracked_memory = current / (1024 * 1024)  # MB
        
        metrics = MemoryMetrics(
            total_system_memory_gb=system_memory.total / (1024**3),
            available_system_memory_gb=system_memory.available / (1024**3),
            system_memory_percent=system_memory.percent,
            process_memory_mb=process_memory.rss / (1024**2),
            process_memory_percent=(process_memory.rss / system_memory.total) * 100,
            peak_memory_mb=process_memory.peak_wss / (1024**2) if hasattr(process_memory, 'peak_wss') else 0,
            python_objects_count=python_objects,
            python_object_size_mb=sys.getsizeof(gc.get_objects()) / (1024**2),
            gc_collections=gc_collections,
            gc_collected=gc_collected,
            gc_uncollectable=gc_uncollectable,
            tracked_allocations_mb=tracked_memory,
            memory_leaks_detected=len(self.leak_detector.leak_candidates)
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.running:
            try:
                # Collect metrics
                metrics = await self._collect_memory_metrics()
                
                # Record key metrics
                self.monitor.record_metric("memory_usage_mb", metrics.process_memory_mb, "gauge")
                self.monitor.record_metric("memory_usage_percent", metrics.process_memory_percent, "gauge")
                self.monitor.record_metric("python_objects_count", float(metrics.python_objects_count), "gauge")
                self.monitor.record_metric("memory_leaks_detected", float(metrics.memory_leaks_detected), "gauge")
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _optimization_loop(self):
        """Background optimization loop."""
        while self.running:
            try:
                # Perform periodic optimization
                optimization_result = await self.optimize_memory()
                
                if optimization_result['memory_freed_mb'] > 10:  # Significant memory freed
                    logger.info(f"Memory optimization freed {optimization_result['memory_freed_mb']:.1f}MB")
                
                # Check for memory leaks
                detected_leaks = self.leak_detector.analyze_memory_growth()
                if detected_leaks:
                    logger.warning(f"Detected {len(detected_leaks)} potential memory leaks")
                    for leak in detected_leaks[:3]:  # Log first 3 leaks
                        logger.warning(f"Memory leak: {leak.allocation_site} "
                                     f"({leak.growth_rate_mb_per_hour:.2f} MB/hour)")
                
                await asyncio.sleep(120)  # Optimize every 2 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Memory optimization loop error: {e}")
                await asyncio.sleep(300)
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory optimization report."""
        if not self.metrics_history:
            current_metrics = asyncio.run(self._collect_memory_metrics())
        else:
            current_metrics = self.metrics_history[-1]
        
        # Object pool statistics
        pool_stats = {}
        for name, pool in self.object_pools.items():
            pool_stats[name] = pool.get_stats()
        
        # Cache statistics
        cache_stats = {}
        for name, cache in self.caches.items():
            cache_stats[name] = cache.get_memory_usage()
        
        # Memory leak summary
        leak_summary = self.leak_detector.get_memory_summary()
        
        # GC analysis
        gc_analysis = self.gc_optimizer.analyze_gc_performance()
        
        return {
            'strategy': self.strategy.value,
            'current_metrics': current_metrics.__dict__,
            'pressure_level': self.pressure_handler.current_pressure_level.value,
            'object_pools': pool_stats,
            'caches': cache_stats,
            'memory_tracking': leak_summary,
            'gc_analysis': gc_analysis,
            'optimization_running': self.running,
            'metrics_collected': len(self.metrics_history)
        }


# Global memory optimizer
_global_memory_optimizer: Optional[AdvancedMemoryOptimizer] = None


def get_memory_optimizer(**kwargs) -> AdvancedMemoryOptimizer:
    """Get global memory optimizer instance."""
    global _global_memory_optimizer
    if _global_memory_optimizer is None:
        _global_memory_optimizer = AdvancedMemoryOptimizer(**kwargs)
    return _global_memory_optimizer


# Decorator for memory-efficient functions
def memory_optimized(pool_name: Optional[str] = None, cache_result: bool = False):
    """Decorator for memory optimization."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            optimizer = get_memory_optimizer()
            
            # Use object pooling if specified
            if pool_name and pool_name in optimizer.object_pools:
                pool = optimizer.object_pools[pool_name]
                # This is a simplified example - actual implementation would depend on function signature
                result = await func(*args, **kwargs)
            else:
                result = await func(*args, **kwargs)
            
            # Cache result if requested
            if cache_result:
                # Simple caching implementation - would need proper cache key generation
                pass
            
            return result
        
        def sync_wrapper(*args, **kwargs):
            optimizer = get_memory_optimizer()
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Example usage
if __name__ == "__main__":
    async def example_memory_optimization():
        """Example of using memory optimization."""
        
        # Create memory optimizer
        optimizer = AdvancedMemoryOptimizer(
            strategy=MemoryStrategy.ADAPTIVE,
            max_cache_memory_mb=100.0,
            gc_strategy=GCStrategy.PERFORMANCE
        )
        
        await optimizer.start()
        
        try:
            # Create object pool
            def create_list():
                return []
            
            def reset_list(lst):
                lst.clear()
            
            list_pool = optimizer.create_object_pool(
                "list_pool",
                create_list,
                reset_list,
                max_size=50
            )
            
            # Create cache
            cache = optimizer.create_cache(
                "example_cache",
                max_memory_mb=20.0,
                max_items=1000
            )
            
            # Simulate memory usage
            print("Simulating memory usage...")
            
            # Use object pool
            for i in range(20):
                lst = list_pool.acquire()
                lst.extend(range(100))
                # Use list...
                list_pool.release(lst)
            
            # Use cache
            for i in range(100):
                cache.put(f"key_{i}", f"value_{i}" * 100)
            
            # Perform optimization
            print("Performing memory optimization...")
            result = await optimizer.optimize_memory()
            
            print(f"Optimization results:")
            print(f"  Memory freed: {result['memory_freed_mb']:.2f} MB")
            print(f"  Pressure level: {result['pressure_level']}")
            print(f"  Detected leaks: {result['detected_leaks']}")
            print(f"  Optimization time: {result['optimization_time']:.3f} seconds")
            
            # Get memory report
            report = optimizer.get_memory_report()
            print(f"\nMemory Report:")
            print(f"  Current memory usage: {report['current_metrics']['process_memory_mb']:.1f} MB")
            print(f"  Python objects: {report['current_metrics']['python_objects_count']:,}")
            print(f"  Object pools: {len(report['object_pools'])}")
            print(f"  Caches: {len(report['caches'])}")
            
            # Pool statistics
            pool_stats = list_pool.get_stats()
            print(f"  List pool hit rate: {pool_stats['hit_rate']:.1f}%")
            
            # Cache statistics
            cache_stats = cache.get_memory_usage()
            print(f"  Cache utilization: {cache_stats['utilization_percent']:.1f}%")
            
        finally:
            await optimizer.stop()
    
    # Run example
    asyncio.run(example_memory_optimization())
    print("Memory optimization example completed!")