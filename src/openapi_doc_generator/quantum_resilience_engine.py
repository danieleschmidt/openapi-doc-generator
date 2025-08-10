"""Advanced resilience and fault tolerance engine."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout

from .quantum_audit_logger import AuditEventType, get_audit_logger
from .quantum_health_monitor import get_health_monitor, HealthStatus
from .quantum_recovery import RecoveryStrategy, QuantumRecoveryManager
from .quantum_security import SecurityLevel


class ResiliencePattern(Enum):
    """Resilience patterns for fault tolerance."""
    CIRCUIT_BREAKER = "circuit_breaker"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    BULKHEAD = "bulkhead"
    TIMEOUT = "timeout"
    FALLBACK = "fallback"
    RATE_LIMITER = "rate_limiter"
    CACHE_ASIDE = "cache_aside"
    SAGA_PATTERN = "saga_pattern"


@dataclass
class ResilienceConfig:
    """Configuration for resilience patterns."""
    pattern: ResiliencePattern
    max_retries: int = 3
    timeout_seconds: float = 30.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    backoff_multiplier: float = 2.0
    max_backoff: float = 60.0
    rate_limit_per_second: int = 100
    bulkhead_max_concurrent: int = 10
    cache_ttl_seconds: int = 300
    enable_jitter: bool = True


@dataclass
class OperationResult:
    """Result of a resilient operation."""
    success: bool
    result: Any
    error: Optional[Exception]
    attempts: int
    total_duration_ms: float
    pattern_used: Optional[ResiliencePattern]
    metadata: Dict[str, Any]


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """Circuit breaker implementation."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 half_open_max_calls: int = 3):
        """Initialize circuit breaker."""
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.half_open_call_count = 0
        
        self.logger = logging.getLogger(f"{__name__}.circuit_breaker")
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time < self.recovery_timeout:
                raise RuntimeError("Circuit breaker is OPEN - calls blocked")
            else:
                # Transition to half-open
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_call_count = 0
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
                
        if self.state == CircuitBreakerState.HALF_OPEN:
            if self.half_open_call_count >= self.half_open_max_calls:
                raise RuntimeError("Circuit breaker HALF_OPEN - max calls exceeded")
                
        try:
            result = func(*args, **kwargs)
            
            # Success - reset or close circuit
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.logger.info("Circuit breaker CLOSED after successful half-open test")
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = max(0, self.failure_count - 1)
                
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                self.logger.warning("Circuit breaker OPEN after half-open failure")
            elif (self.state == CircuitBreakerState.CLOSED and 
                  self.failure_count >= self.failure_threshold):
                self.state = CircuitBreakerState.OPEN
                self.logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")
                
            raise e


class BulkheadExecutor:
    """Bulkhead pattern executor for resource isolation."""
    
    def __init__(self, max_concurrent: int = 10, queue_size: int = 100):
        """Initialize bulkhead executor."""
        self.max_concurrent = max_concurrent
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.active_tasks = 0
        self.queue_size = queue_size
        self.logger = logging.getLogger(f"{__name__}.bulkhead")
        
    def submit(self, func: Callable, *args, **kwargs) -> Any:
        """Submit task to bulkhead executor."""
        if self.active_tasks >= self.max_concurrent:
            raise RuntimeError(f"Bulkhead limit exceeded: {self.active_tasks}/{self.max_concurrent}")
            
        self.active_tasks += 1
        
        try:
            future = self.executor.submit(func, *args, **kwargs)
            result = future.result()  # Wait for completion
            return result
        finally:
            self.active_tasks -= 1
            
    def shutdown(self):
        """Shutdown bulkhead executor."""
        self.executor.shutdown(wait=True)


class RateLimiter:
    """Rate limiter implementation."""
    
    def __init__(self, rate_per_second: int = 100):
        """Initialize rate limiter."""
        self.rate_per_second = rate_per_second
        self.tokens = rate_per_second
        self.last_refill = time.time()
        self.logger = logging.getLogger(f"{__name__}.rate_limiter")
        
    def acquire(self) -> bool:
        """Acquire a rate limit token."""
        now = time.time()
        
        # Refill tokens based on elapsed time
        elapsed = now - self.last_refill
        self.tokens = min(self.rate_per_second, 
                         self.tokens + elapsed * self.rate_per_second)
        self.last_refill = now
        
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        else:
            return False
            
    def wait_for_token(self, timeout: float = 5.0) -> bool:
        """Wait for a rate limit token."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.acquire():
                return True
            time.sleep(0.01)  # Small delay
            
        return False


class ResilienceCache:
    """Simple in-memory cache for fallback data."""
    
    def __init__(self, default_ttl: int = 300):
        """Initialize cache."""
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(f"{__name__}.cache")
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            entry = self.cache[key]
            if time.time() < entry['expires']:
                self.logger.debug(f"Cache hit: {key}")
                return entry['value']
            else:
                # Expired entry
                del self.cache[key]
                self.logger.debug(f"Cache expired: {key}")
                
        return None
        
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        ttl = ttl or self.default_ttl
        expires = time.time() + ttl
        
        self.cache[key] = {
            'value': value,
            'expires': expires
        }
        
        self.logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
        
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.logger.info("Cache cleared")


class QuantumResilienceEngine:
    """Advanced resilience engine with multiple fault tolerance patterns."""
    
    def __init__(self):
        """Initialize resilience engine."""
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.bulkhead_executors: Dict[str, BulkheadExecutor] = {}
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.cache = ResilienceCache()
        
        self.recovery_manager = QuantumRecoveryManager()
        self.health_monitor = get_health_monitor()
        self.audit_logger = get_audit_logger()
        
        self.logger = logging.getLogger(__name__)
        
        # Operation metrics
        self.operation_stats: Dict[str, Dict[str, Any]] = {}
        
    def execute_resilient(self,
                         operation_name: str,
                         func: Callable,
                         config: Optional[ResilienceConfig] = None,
                         *args, **kwargs) -> OperationResult:
        """Execute operation with resilience patterns."""
        start_time = time.time()
        config = config or ResilienceConfig(pattern=ResiliencePattern.RETRY_WITH_BACKOFF)
        
        try:
            result = self._execute_with_patterns(
                operation_name, func, config, *args, **kwargs
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Update success metrics
            self._update_operation_stats(operation_name, success=True, duration_ms=duration_ms)
            
            return OperationResult(
                success=True,
                result=result,
                error=None,
                attempts=1,
                total_duration_ms=duration_ms,
                pattern_used=config.pattern,
                metadata={"operation_name": operation_name}
            )
            
        except Exception as error:
            duration_ms = (time.time() - start_time) * 1000
            
            # Update failure metrics
            self._update_operation_stats(operation_name, success=False, duration_ms=duration_ms)
            
            # Log security event for critical failures
            if duration_ms > 30000:  # >30 seconds
                self.audit_logger.log_security_event(
                    event_type=AuditEventType.ERROR_CONDITION,
                    action=f"resilient_operation_failed",
                    result="timeout",
                    severity=SecurityLevel.HIGH,
                    details={
                        "operation_name": operation_name,
                        "duration_ms": duration_ms,
                        "error": str(error)
                    }
                )
                
            return OperationResult(
                success=False,
                result=None,
                error=error,
                attempts=1,
                total_duration_ms=duration_ms,
                pattern_used=config.pattern,
                metadata={"operation_name": operation_name, "error": str(error)}
            )
            
    def _execute_with_patterns(self,
                              operation_name: str,
                              func: Callable,
                              config: ResilienceConfig,
                              *args, **kwargs) -> Any:
        """Execute operation with configured resilience patterns."""
        if config.pattern == ResiliencePattern.CIRCUIT_BREAKER:
            return self._execute_with_circuit_breaker(operation_name, func, config, *args, **kwargs)
        elif config.pattern == ResiliencePattern.RETRY_WITH_BACKOFF:
            return self._execute_with_retry(operation_name, func, config, *args, **kwargs)
        elif config.pattern == ResiliencePattern.BULKHEAD:
            return self._execute_with_bulkhead(operation_name, func, config, *args, **kwargs)
        elif config.pattern == ResiliencePattern.TIMEOUT:
            return self._execute_with_timeout(operation_name, func, config, *args, **kwargs)
        elif config.pattern == ResiliencePattern.FALLBACK:
            return self._execute_with_fallback(operation_name, func, config, *args, **kwargs)
        elif config.pattern == ResiliencePattern.RATE_LIMITER:
            return self._execute_with_rate_limit(operation_name, func, config, *args, **kwargs)
        elif config.pattern == ResiliencePattern.CACHE_ASIDE:
            return self._execute_with_cache(operation_name, func, config, *args, **kwargs)
        else:
            # Default: direct execution
            return func(*args, **kwargs)
            
    def _execute_with_circuit_breaker(self,
                                    operation_name: str,
                                    func: Callable,
                                    config: ResilienceConfig,
                                    *args, **kwargs) -> Any:
        """Execute with circuit breaker pattern."""
        if operation_name not in self.circuit_breakers:
            self.circuit_breakers[operation_name] = CircuitBreaker(
                failure_threshold=config.circuit_breaker_threshold,
                recovery_timeout=config.circuit_breaker_timeout
            )
            
        breaker = self.circuit_breakers[operation_name]
        return breaker.call(func, *args, **kwargs)
        
    def _execute_with_retry(self,
                           operation_name: str,
                           func: Callable,
                           config: ResilienceConfig,
                           *args, **kwargs) -> Any:
        """Execute with retry and exponential backoff."""
        last_exception = None
        
        for attempt in range(config.max_retries + 1):
            try:
                if attempt > 0:
                    # Calculate backoff delay
                    delay = min(config.max_backoff,
                               (config.backoff_multiplier ** (attempt - 1)))
                    
                    # Add jitter if enabled
                    if config.enable_jitter:
                        import random
                        delay *= (0.5 + random.random() * 0.5)
                        
                    self.logger.info(f"Retry {attempt}/{config.max_retries} for {operation_name} "
                                   f"after {delay:.2f}s delay")
                    time.sleep(delay)
                    
                return func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Attempt {attempt + 1} failed for {operation_name}: {e}")
                
                # Don't retry certain types of errors
                if self._is_non_retryable_error(e):
                    break
                    
        # All retries exhausted
        raise last_exception
        
    def _execute_with_bulkhead(self,
                              operation_name: str,
                              func: Callable,
                              config: ResilienceConfig,
                              *args, **kwargs) -> Any:
        """Execute with bulkhead resource isolation."""
        if operation_name not in self.bulkhead_executors:
            self.bulkhead_executors[operation_name] = BulkheadExecutor(
                max_concurrent=config.bulkhead_max_concurrent
            )
            
        executor = self.bulkhead_executors[operation_name]
        return executor.submit(func, *args, **kwargs)
        
    def _execute_with_timeout(self,
                             operation_name: str,
                             func: Callable,
                             config: ResilienceConfig,
                             *args, **kwargs) -> Any:
        """Execute with timeout protection."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=config.timeout_seconds)
            except FuturesTimeout:
                raise TimeoutError(f"Operation {operation_name} timed out after {config.timeout_seconds}s")
                
    def _execute_with_fallback(self,
                              operation_name: str,
                              func: Callable,
                              config: ResilienceConfig,
                              *args, **kwargs) -> Any:
        """Execute with fallback strategy."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.logger.warning(f"Primary operation failed for {operation_name}: {e}")
            
            # Try to get cached result as fallback
            cache_key = f"{operation_name}_{hash(str(args) + str(kwargs))}"
            cached_result = self.cache.get(cache_key)
            
            if cached_result is not None:
                self.logger.info(f"Using cached fallback for {operation_name}")
                return cached_result
            else:
                # No fallback available
                raise e
                
    def _execute_with_rate_limit(self,
                                operation_name: str,
                                func: Callable,
                                config: ResilienceConfig,
                                *args, **kwargs) -> Any:
        """Execute with rate limiting."""
        if operation_name not in self.rate_limiters:
            self.rate_limiters[operation_name] = RateLimiter(
                rate_per_second=config.rate_limit_per_second
            )
            
        limiter = self.rate_limiters[operation_name]
        
        if not limiter.wait_for_token(timeout=5.0):
            raise RuntimeError(f"Rate limit exceeded for {operation_name}")
            
        return func(*args, **kwargs)
        
    def _execute_with_cache(self,
                           operation_name: str,
                           func: Callable,
                           config: ResilienceConfig,
                           *args, **kwargs) -> Any:
        """Execute with cache-aside pattern."""
        # Generate cache key
        cache_key = f"{operation_name}_{hash(str(args) + str(kwargs))}"
        
        # Try cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self.logger.debug(f"Cache hit for {operation_name}")
            return cached_result
            
        # Execute and cache result
        result = func(*args, **kwargs)
        self.cache.set(cache_key, result, ttl=config.cache_ttl_seconds)
        
        return result
        
    def _is_non_retryable_error(self, error: Exception) -> bool:
        """Check if error should not be retried."""
        non_retryable_types = [
            ValueError,
            TypeError,
            AttributeError,
            ImportError,
            SyntaxError
        ]
        
        return any(isinstance(error, error_type) for error_type in non_retryable_types)
        
    def _update_operation_stats(self,
                               operation_name: str,
                               success: bool,
                               duration_ms: float) -> None:
        """Update operation statistics."""
        if operation_name not in self.operation_stats:
            self.operation_stats[operation_name] = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_duration_ms": 0.0,
                "avg_duration_ms": 0.0,
                "max_duration_ms": 0.0,
                "min_duration_ms": float('inf')
            }
            
        stats = self.operation_stats[operation_name]
        stats["total_calls"] += 1
        stats["total_duration_ms"] += duration_ms
        stats["avg_duration_ms"] = stats["total_duration_ms"] / stats["total_calls"]
        stats["max_duration_ms"] = max(stats["max_duration_ms"], duration_ms)
        stats["min_duration_ms"] = min(stats["min_duration_ms"], duration_ms)
        
        if success:
            stats["successful_calls"] += 1
        else:
            stats["failed_calls"] += 1
            
    @contextmanager
    def resilient_context(self,
                         operation_name: str,
                         config: Optional[ResilienceConfig] = None):
        """Context manager for resilient operations."""
        try:
            self.logger.info(f"Starting resilient operation: {operation_name}")
            yield
            self.logger.info(f"Completed resilient operation: {operation_name}")
        except Exception as e:
            self.logger.error(f"Resilient operation failed: {operation_name} - {e}")
            
            # Attempt recovery
            if config and config.pattern == ResiliencePattern.FALLBACK:
                self.logger.info(f"Attempting fallback recovery for {operation_name}")
                # Recovery logic would go here
                
            raise e
            
    def get_operation_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get operation statistics."""
        return self.operation_stats.copy()
        
    def get_circuit_breaker_states(self) -> Dict[str, str]:
        """Get current circuit breaker states."""
        return {
            name: breaker.state.value 
            for name, breaker in self.circuit_breakers.items()
        }
        
    def reset_circuit_breaker(self, operation_name: str) -> bool:
        """Reset a circuit breaker."""
        if operation_name in self.circuit_breakers:
            breaker = self.circuit_breakers[operation_name]
            breaker.state = CircuitBreakerState.CLOSED
            breaker.failure_count = 0
            
            self.logger.info(f"Reset circuit breaker for {operation_name}")
            self.audit_logger.log_security_event(
                event_type=AuditEventType.SYSTEM_ACCESS,
                action="circuit_breaker_reset",
                result="success",
                severity=SecurityLevel.LOW,
                details={"operation_name": operation_name}
            )
            return True
            
        return False
        
    def clear_cache(self, pattern: Optional[str] = None) -> None:
        """Clear cache entries."""
        if pattern:
            # Remove specific pattern
            keys_to_remove = [k for k in self.cache.cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self.cache.cache[key]
            self.logger.info(f"Cleared cache entries matching pattern: {pattern}")
        else:
            # Clear all
            self.cache.clear()
            self.logger.info("Cleared all cache entries")
            
    def shutdown(self) -> None:
        """Shutdown resilience engine."""
        # Shutdown bulkhead executors
        for executor in self.bulkhead_executors.values():
            executor.shutdown()
            
        self.logger.info("Resilience engine shutdown complete")


# Global resilience engine instance
_resilience_engine: Optional[QuantumResilienceEngine] = None


def get_resilience_engine() -> QuantumResilienceEngine:
    """Get global resilience engine instance."""
    global _resilience_engine
    if _resilience_engine is None:
        _resilience_engine = QuantumResilienceEngine()
    return _resilience_engine