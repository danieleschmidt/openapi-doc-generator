"""
Advanced Multi-Level Caching System for Performance Optimization

Provides intelligent caching with LRU, TTL, and adaptive algorithms to maximize
performance for documentation generation operations.
"""

import asyncio
import hashlib
import time
import threading
import weakref
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable, Generic, TypeVar, Union
from collections import OrderedDict
from pathlib import Path

T = TypeVar('T')


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    max_size_bytes: int = 0
    avg_access_time_ms: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


@dataclass
class CacheEntry(Generic[T]):
    """Individual cache entry with metadata."""
    value: T
    created_at: float
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self) -> None:
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1


class LRUTTLCache(Generic[T]):
    """LRU cache with TTL support and memory management."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100, default_ttl: float = None):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats(max_size_bytes=self.max_memory_bytes)
        
    def get(self, key: str, default: T = None) -> Optional[T]:
        """Get value from cache with LRU ordering."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check if expired
                if entry.is_expired():
                    del self._cache[key]
                    self._stats.size_bytes -= entry.size_bytes
                    self._stats.misses += 1
                    return default
                
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                entry.touch()
                
                self._stats.hits += 1
                return entry.value
            
            self._stats.misses += 1
            return default
    
    def put(self, key: str, value: T, ttl: float = None) -> None:
        """Put value in cache with optional TTL."""
        with self._lock:
            # Calculate entry size
            size_bytes = self._estimate_size(value)
            
            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache[key]
                self._stats.size_bytes -= old_entry.size_bytes
                del self._cache[key]
            
            # Create new entry
            entry = CacheEntry(
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                size_bytes=size_bytes,
                ttl=ttl or self.default_ttl
            )
            
            # Evict entries if necessary
            self._evict_if_needed(size_bytes)
            
            # Add new entry
            self._cache[key] = entry
            self._stats.size_bytes += size_bytes
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        try:
            import sys
            if hasattr(sys, 'getsizeof'):
                return sys.getsizeof(value)
            else:
                # Fallback estimation
                if isinstance(value, str):
                    return len(value.encode('utf-8'))
                elif isinstance(value, (list, tuple)):
                    return sum(self._estimate_size(item) for item in value)
                elif isinstance(value, dict):
                    return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in value.items())
                else:
                    return 64  # Default estimate
        except Exception:
            return 64  # Safe fallback
    
    def _evict_if_needed(self, incoming_size: int) -> None:
        """Evict entries to make room for new entry."""
        # Size-based eviction
        while (len(self._cache) >= self.max_size or 
               self._stats.size_bytes + incoming_size > self.max_memory_bytes):
            if not self._cache:
                break
            
            # Remove least recently used
            oldest_key, oldest_entry = self._cache.popitem(last=False)
            self._stats.size_bytes -= oldest_entry.size_bytes
            self._stats.evictions += 1
        
        # TTL-based cleanup
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            entry = self._cache.pop(key)
            self._stats.size_bytes -= entry.size_bytes
            self._stats.evictions += 1
    
    def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            self._cache.clear()
            self._stats = CacheStats(max_size_bytes=self.max_memory_bytes)
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                size_bytes=self._stats.size_bytes,
                max_size_bytes=self._stats.max_size_bytes,
                avg_access_time_ms=0.0  # Could implement if needed
            )


class AdaptiveCache:
    """Adaptive caching system with multiple strategies."""
    
    def __init__(self, max_memory_mb: int = 200):
        self.max_memory_mb = max_memory_mb
        
        # Different cache levels for different data types
        self.ast_cache = LRUTTLCache[Any](max_size=500, max_memory_mb=max_memory_mb // 4, default_ttl=3600)  # 1 hour
        self.route_cache = LRUTTLCache[Any](max_size=1000, max_memory_mb=max_memory_mb // 4, default_ttl=1800)  # 30 min
        self.schema_cache = LRUTTLCache[Any](max_size=2000, max_memory_mb=max_memory_mb // 4, default_ttl=900)   # 15 min
        self.doc_cache = LRUTTLCache[Any](max_size=100, max_memory_mb=max_memory_mb // 4, default_ttl=600)    # 10 min
        
        self._lock = threading.RLock()
    
    def get_ast(self, file_path: str, file_mtime: float) -> Optional[Any]:
        """Get cached AST for file."""
        cache_key = f"ast:{file_path}:{file_mtime}"
        return self.ast_cache.get(cache_key)
    
    def put_ast(self, file_path: str, file_mtime: float, ast_tree: Any) -> None:
        """Cache AST for file."""
        cache_key = f"ast:{file_path}:{file_mtime}"
        self.ast_cache.put(cache_key, ast_tree, ttl=3600)  # 1 hour TTL
    
    def get_routes(self, content_hash: str) -> Optional[Any]:
        """Get cached routes for content."""
        return self.route_cache.get(f"routes:{content_hash}")
    
    def put_routes(self, content_hash: str, routes: Any) -> None:
        """Cache routes for content."""
        self.route_cache.put(f"routes:{content_hash}", routes, ttl=1800)  # 30 min TTL
    
    def get_schema(self, schema_key: str) -> Optional[Any]:
        """Get cached schema information."""
        return self.schema_cache.get(f"schema:{schema_key}")
    
    def put_schema(self, schema_key: str, schema: Any) -> None:
        """Cache schema information."""
        self.schema_cache.put(f"schema:{schema_key}", schema, ttl=900)  # 15 min TTL
    
    def get_documentation(self, doc_key: str) -> Optional[str]:
        """Get cached documentation."""
        return self.doc_cache.get(f"doc:{doc_key}")
    
    def put_documentation(self, doc_key: str, documentation: str) -> None:
        """Cache generated documentation."""
        self.doc_cache.put(f"doc:{doc_key}", documentation, ttl=600)  # 10 min TTL
    
    def compute_file_hash(self, file_path: Union[str, Path]) -> str:
        """Compute hash of file content for cache keys."""
        try:
            path = Path(file_path)
            if not path.exists():
                return "missing_file"
            
            # Use file size and mtime for quick hash
            stat = path.stat()
            content = f"{path}:{stat.st_size}:{stat.st_mtime}"
            return hashlib.md5(content.encode()).hexdigest()[:16]
        except Exception:
            return "error_hash"
    
    def invalidate_file(self, file_path: str) -> None:
        """Invalidate all cache entries related to a file."""
        # This is a simplified implementation
        # In production, you'd want to track dependencies more precisely
        self.clear_all()
    
    def clear_all(self) -> None:
        """Clear all caches."""
        with self._lock:
            self.ast_cache.clear()
            self.route_cache.clear()
            self.schema_cache.clear()
            self.doc_cache.clear()
    
    def get_overall_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all cache levels."""
        return {
            "ast_cache": self.ast_cache.get_stats(),
            "route_cache": self.route_cache.get_stats(),
            "schema_cache": self.schema_cache.get_stats(),
            "doc_cache": self.doc_cache.get_stats()
        }


# Global cache instance
_global_cache: Optional[AdaptiveCache] = None
_cache_lock = threading.RLock()


def get_cache(max_memory_mb: int = 200) -> AdaptiveCache:
    """Get global adaptive cache instance."""
    global _global_cache
    with _cache_lock:
        if _global_cache is None:
            _global_cache = AdaptiveCache(max_memory_mb)
        return _global_cache


def cached_operation(cache_type: str, ttl: float = None):
    """Decorator for caching operation results."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Create cache key from function name and arguments
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()[:16]
            
            # Try to get from appropriate cache
            if cache_type == "ast":
                result = cache.ast_cache.get(cache_key)
            elif cache_type == "routes":
                result = cache.route_cache.get(cache_key)
            elif cache_type == "schema":
                result = cache.schema_cache.get(cache_key)
            elif cache_type == "doc":
                result = cache.doc_cache.get(cache_key)
            else:
                result = None
            
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            # Put in appropriate cache
            if cache_type == "ast":
                cache.ast_cache.put(cache_key, result, ttl)
            elif cache_type == "routes":
                cache.route_cache.put(cache_key, result, ttl)
            elif cache_type == "schema":
                cache.schema_cache.put(cache_key, result, ttl)
            elif cache_type == "doc":
                cache.doc_cache.put(cache_key, result, ttl)
            
            return result
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    return decorator