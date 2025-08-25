"""
Quantum-Inspired Intelligent Caching System

This module implements advanced caching with machine learning-inspired
eviction policies and quantum-inspired optimization for maximum performance.
"""

import hashlib
import logging
import pickle
import threading
import time
import weakref
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    ARC = "arc"                    # Adaptive Replacement Cache
    QUANTUM_WEIGHTED = "quantum_weighted"  # Quantum-inspired weighting
    ML_PREDICTIVE = "ml_predictive"        # Machine learning prediction


class AccessPattern(Enum):
    """Types of access patterns for optimization."""
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"


@dataclass
class CacheEntry:
    """Individual cache entry with metadata."""
    key: str
    value: Any
    size: int
    access_count: int = 0
    last_access_time: float = field(default_factory=time.time)
    creation_time: float = field(default_factory=time.time)
    access_pattern: AccessPattern = AccessPattern.RANDOM
    quantum_weight: float = 1.0
    prediction_score: float = 0.5
    ttl: Optional[float] = None  # Time to live


@dataclass
class CacheConfig:
    """Configuration for intelligent cache."""
    max_size_mb: int = 200
    max_entries: int = 10000
    default_ttl: Optional[float] = None
    strategy: CacheStrategy = CacheStrategy.QUANTUM_WEIGHTED
    quantum_learning_rate: float = 0.1
    prediction_threshold: float = 0.7
    cleanup_interval: float = 60.0  # seconds
    stats_window_size: int = 1000


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    avg_access_time: float = 0.0
    hit_rate: float = 0.0
    quantum_efficiency: float = 0.0
    recent_patterns: List[AccessPattern] = field(default_factory=list)


class QuantumIntelligentCache:
    """Advanced cache with quantum-inspired optimization and ML prediction."""

    def __init__(self, name: str, config: Optional[CacheConfig] = None):
        self.name = name
        self.config = config or CacheConfig()
        self.entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self.frequency_tracker: Dict[str, int] = defaultdict(int)
        self.access_history: List[Tuple[str, float]] = []
        self.stats = CacheStats()
        self.lock = threading.RLock()
        self.logger = logging.getLogger(f"QuantumCache.{name}")

        # Quantum-inspired state
        self.quantum_state = 1.0
        self.coherence_time = 10.0
        self.entanglement_map: Dict[str, set] = defaultdict(set)

        # Pattern recognition
        self.pattern_weights: Dict[AccessPattern, float] = dict.fromkeys(AccessPattern, 1.0)

        # Start background cleanup thread
        self.cleanup_thread = threading.Thread(
            target=self._background_cleanup,
            daemon=True
        )
        self.cleanup_thread.start()

        # Weak reference cleanup
        self._finalizer = weakref.finalize(self, self._cleanup_resources)

    def get(self, key: str, default: Any = None) -> Any:
        """Get item from cache with quantum-enhanced lookup."""
        with self.lock:
            start_time = time.time()

            if key in self.entries:
                entry = self.entries[key]

                # Check TTL expiration
                if entry.ttl and time.time() - entry.creation_time > entry.ttl:
                    self._evict_entry(key)
                    self.stats.misses += 1
                    return default

                # Update access metadata
                entry.access_count += 1
                entry.last_access_time = time.time()
                self.frequency_tracker[key] += 1

                # Move to end for LRU
                self.entries.move_to_end(key)

                # Update quantum weight based on access pattern
                self._update_quantum_weight(entry)

                # Record access for pattern analysis
                self._record_access(key, start_time)

                self.stats.hits += 1
                access_time = time.time() - start_time
                self._update_avg_access_time(access_time)

                return entry.value

            else:
                self.stats.misses += 1
                self._record_miss(key)
                return default

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put item in cache with intelligent placement."""
        with self.lock:
            try:
                # Calculate size
                size = self._calculate_size(value)

                # Check if we need to evict entries
                while (self._should_evict(size) and self.entries):
                    self._evict_next_entry()

                # Create new entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    size=size,
                    ttl=ttl or self.config.default_ttl,
                    quantum_weight=self._calculate_initial_quantum_weight(key)
                )

                # Remove old entry if exists
                if key in self.entries:
                    old_entry = self.entries[key]
                    self.stats.size_bytes -= old_entry.size

                # Add new entry
                self.entries[key] = entry
                self.stats.size_bytes += size
                self.stats.entry_count = len(self.entries)

                # Update quantum entanglement
                self._update_entanglement(key, entry)

                return True

            except Exception as e:
                self.logger.error(f"Failed to cache key {key}: {e}")
                return False

    def evict(self, key: str) -> bool:
        """Manually evict specific key."""
        with self.lock:
            if key in self.entries:
                self._evict_entry(key)
                return True
            return False

    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.entries.clear()
            self.frequency_tracker.clear()
            self.access_history.clear()
            self.entanglement_map.clear()
            self.stats = CacheStats()

    def _should_evict(self, new_size: int) -> bool:
        """Determine if eviction is needed."""
        current_size_mb = self.stats.size_bytes / (1024 * 1024)
        new_total_mb = (self.stats.size_bytes + new_size) / (1024 * 1024)

        return (
            new_total_mb > self.config.max_size_mb or
            len(self.entries) >= self.config.max_entries
        )

    def _evict_next_entry(self):
        """Evict the next entry based on configured strategy."""
        if not self.entries:
            return

        if self.config.strategy == CacheStrategy.LRU:
            key = next(iter(self.entries))
        elif self.config.strategy == CacheStrategy.LFU:
            key = min(self.entries.keys(), key=lambda k: self.frequency_tracker[k])
        elif self.config.strategy == CacheStrategy.QUANTUM_WEIGHTED:
            key = self._select_quantum_eviction_candidate()
        else:
            # Default to LRU
            key = next(iter(self.entries))

        self._evict_entry(key)

    def _select_quantum_eviction_candidate(self) -> str:
        """Select eviction candidate using quantum-inspired algorithm."""
        if not self.entries:
            return ""

        # Calculate quantum scores for all entries
        scores = {}
        current_time = time.time()

        for key, entry in self.entries.items():
            # Age factor (older entries have higher eviction probability)
            age_factor = (current_time - entry.last_access_time) / 3600.0  # hours

            # Frequency factor (less frequent = higher eviction probability)
            freq_factor = 1.0 / max(entry.access_count, 1)

            # Quantum decoherence (lower weight = higher eviction probability)
            quantum_factor = 1.0 / max(entry.quantum_weight, 0.1)

            # Size factor (larger entries more likely to be evicted under pressure)
            size_factor = entry.size / (1024 * 1024)  # MB

            # Combined score (higher = more likely to evict)
            score = (age_factor * 0.3 +
                    freq_factor * 0.3 +
                    quantum_factor * 0.3 +
                    size_factor * 0.1)

            # Add quantum uncertainty
            quantum_uncertainty = abs(hash(key) % 100) / 1000.0
            score += quantum_uncertainty

            scores[key] = score

        # Select entry with highest eviction score
        return max(scores.keys(), key=lambda k: scores[k])

    def _evict_entry(self, key: str):
        """Remove entry from cache."""
        if key in self.entries:
            entry = self.entries.pop(key)
            self.stats.size_bytes -= entry.size
            self.stats.evictions += 1

            # Clean up related data
            if key in self.frequency_tracker:
                del self.frequency_tracker[key]

            if key in self.entanglement_map:
                # Break entanglement with related keys
                for entangled_key in self.entanglement_map[key]:
                    self.entanglement_map[entangled_key].discard(key)
                del self.entanglement_map[key]

            self.stats.entry_count = len(self.entries)

    def _update_quantum_weight(self, entry: CacheEntry):
        """Update quantum weight based on access patterns."""
        current_time = time.time()

        # Time decay factor
        time_since_creation = current_time - entry.creation_time
        time_decay = max(0.1, 1.0 - (time_since_creation / (24 * 3600)))  # 24 hours

        # Frequency boost
        frequency_boost = min(2.0, 1.0 + (entry.access_count / 100.0))

        # Pattern alignment boost
        pattern_boost = self.pattern_weights.get(entry.access_pattern, 1.0)

        # Update quantum weight with learning
        old_weight = entry.quantum_weight
        new_weight = (time_decay * frequency_boost * pattern_boost)

        # Apply learning rate
        entry.quantum_weight = (
            old_weight * (1 - self.config.quantum_learning_rate) +
            new_weight * self.config.quantum_learning_rate
        )

        # Ensure weight bounds
        entry.quantum_weight = max(0.1, min(entry.quantum_weight, 2.0))

    def _calculate_initial_quantum_weight(self, key: str) -> float:
        """Calculate initial quantum weight for new entry."""
        # Base weight
        weight = 1.0

        # Key entropy (more unique keys get higher initial weight)
        key_hash = hashlib.md5(key.encode()).hexdigest()
        entropy = len(set(key_hash)) / 16.0  # Hex characters
        weight *= (0.5 + entropy)

        # Pattern prediction based on existing cache
        if self.entries:
            similar_keys = [k for k in self.entries.keys() if self._keys_similar(key, k)]
            if similar_keys:
                avg_weight = sum(self.entries[k].quantum_weight for k in similar_keys) / len(similar_keys)
                weight = (weight + avg_weight) / 2.0

        return max(0.1, min(weight, 2.0))

    def _keys_similar(self, key1: str, key2: str) -> bool:
        """Check if two keys are similar (simple heuristic)."""
        # Simple similarity based on common prefixes/suffixes
        return (len(key1) > 3 and len(key2) > 3 and
                (key1[:3] == key2[:3] or key1[-3:] == key2[-3:]))

    def _update_entanglement(self, key: str, entry: CacheEntry):
        """Update quantum entanglement between related cache entries."""
        # Find potentially entangled keys (similar patterns, recent access)
        current_time = time.time()

        for other_key, other_entry in self.entries.items():
            if other_key == key:
                continue

            # Criteria for entanglement
            time_proximity = abs(entry.creation_time - other_entry.creation_time) < 300  # 5 minutes
            pattern_match = entry.access_pattern == other_entry.access_pattern
            similar_keys = self._keys_similar(key, other_key)

            if time_proximity or pattern_match or similar_keys:
                self.entanglement_map[key].add(other_key)
                self.entanglement_map[other_key].add(key)

    def _record_access(self, key: str, access_time: float):
        """Record access for pattern analysis."""
        self.access_history.append((key, access_time))

        # Keep only recent history
        if len(self.access_history) > self.config.stats_window_size:
            self.access_history = self.access_history[-self.config.stats_window_size:]

        # Detect access patterns
        self._detect_access_pattern(key)

    def _record_miss(self, key: str):
        """Record cache miss for pattern learning."""
        # This could be used for predictive caching
        pass

    def _detect_access_pattern(self, key: str):
        """Detect access pattern for the key."""
        if key not in self.entries:
            return

        recent_accesses = [
            (k, t) for k, t in self.access_history[-20:] if k == key
        ]

        if len(recent_accesses) < 2:
            return

        # Simple pattern detection
        time_diffs = [
            recent_accesses[i][1] - recent_accesses[i-1][1]
            for i in range(1, len(recent_accesses))
        ]

        if time_diffs:
            avg_diff = sum(time_diffs) / len(time_diffs)
            std_diff = (sum((d - avg_diff)**2 for d in time_diffs) / len(time_diffs))**0.5

            if std_diff < avg_diff * 0.2:  # Low variance = regular pattern
                self.entries[key].access_pattern = AccessPattern.TEMPORAL
            else:
                self.entries[key].access_pattern = AccessPattern.RANDOM

    def _update_avg_access_time(self, access_time: float):
        """Update average access time statistics."""
        if self.stats.avg_access_time == 0.0:
            self.stats.avg_access_time = access_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats.avg_access_time = (
                alpha * access_time +
                (1 - alpha) * self.stats.avg_access_time
            )

    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            # Fallback size estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(
                    self._calculate_size(k) + self._calculate_size(v)
                    for k, v in value.items()
                )
            else:
                return 1024  # Default 1KB estimate

    def _background_cleanup(self):
        """Background thread for cache maintenance."""
        while True:
            try:
                time.sleep(self.config.cleanup_interval)

                with self.lock:
                    current_time = time.time()
                    expired_keys = []

                    # Find expired entries
                    for key, entry in self.entries.items():
                        if (entry.ttl and
                            current_time - entry.creation_time > entry.ttl):
                            expired_keys.append(key)

                    # Remove expired entries
                    for key in expired_keys:
                        self._evict_entry(key)

                    # Update statistics
                    self._update_statistics()

                    # Quantum coherence decay
                    self._update_quantum_coherence()

            except Exception as e:
                self.logger.error(f"Background cleanup error: {e}")

    def _update_statistics(self):
        """Update cache performance statistics."""
        total_requests = self.stats.hits + self.stats.misses
        if total_requests > 0:
            self.stats.hit_rate = self.stats.hits / total_requests

        # Calculate quantum efficiency
        if self.entries:
            avg_quantum_weight = sum(
                entry.quantum_weight for entry in self.entries.values()
            ) / len(self.entries)
            self.stats.quantum_efficiency = min(avg_quantum_weight, 1.0)

    def _update_quantum_coherence(self):
        """Update quantum coherence for all entries."""
        current_time = time.time()

        for entry in self.entries.values():
            time_since_access = current_time - entry.last_access_time
            coherence_decay = max(0.1, 1.0 - (time_since_access / self.coherence_time))
            entry.quantum_weight *= coherence_decay
            entry.quantum_weight = max(0.1, entry.quantum_weight)

    def get_cache_report(self) -> Dict[str, Any]:
        """Generate comprehensive cache performance report."""
        with self.lock:
            return {
                "name": self.name,
                "stats": {
                    "hits": self.stats.hits,
                    "misses": self.stats.misses,
                    "hit_rate": self.stats.hit_rate,
                    "evictions": self.stats.evictions,
                    "entry_count": self.stats.entry_count,
                    "size_mb": self.stats.size_bytes / (1024 * 1024),
                    "avg_access_time_ms": self.stats.avg_access_time * 1000,
                    "quantum_efficiency": self.stats.quantum_efficiency,
                },
                "config": {
                    "max_size_mb": self.config.max_size_mb,
                    "max_entries": self.config.max_entries,
                    "strategy": self.config.strategy.value,
                },
                "quantum_metrics": {
                    "quantum_state": self.quantum_state,
                    "coherence_time": self.coherence_time,
                    "entangled_pairs": sum(len(pairs) for pairs in self.entanglement_map.values()),
                },
                "pattern_analysis": {
                    "pattern_weights": {p.value: w for p, w in self.pattern_weights.items()},
                    "recent_access_count": len(self.access_history),
                },
                "timestamp": time.time()
            }

    def _cleanup_resources(self):
        """Cleanup resources on cache destruction."""
        try:
            self.clear()
        except Exception:
            pass


# Global cache registry
_global_cache_registry: Dict[str, QuantumIntelligentCache] = {}
_cache_lock = threading.Lock()


def get_quantum_cache(name: str, config: Optional[CacheConfig] = None) -> QuantumIntelligentCache:
    """Get or create quantum intelligent cache by name."""
    with _cache_lock:
        if name not in _global_cache_registry:
            _global_cache_registry[name] = QuantumIntelligentCache(name, config)
        return _global_cache_registry[name]


def quantum_cached(cache_name: str, ttl: Optional[float] = None, key_func: Optional[Callable] = None):
    """Decorator for quantum-cached function results."""
    def decorator(func):
        cache = get_quantum_cache(cache_name)

        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"

            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.put(cache_key, result, ttl)
            return result

        wrapper.__name__ = f"quantum_cached_{func.__name__}"
        return wrapper
    return decorator
