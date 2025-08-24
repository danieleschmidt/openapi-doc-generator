"""Robust retry mechanisms with exponential backoff and smart exception handling."""

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
from .enhanced_monitoring import get_monitor


class RetryableException(Exception):
    """Base class for exceptions that should trigger retries."""
    pass


class NonRetryableException(Exception):
    """Base class for exceptions that should not trigger retries."""
    pass


class RetryStrategy(Enum):
    """Retry strategy types."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_INTERVAL = "fixed_interval"
    FIBONACCI = "fibonacci"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_range: float = 0.1
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    retryable_exceptions: List[Type[Exception]] = field(default_factory=lambda: [
        ConnectionError, TimeoutError, OSError, RetryableException
    ])
    non_retryable_exceptions: List[Type[Exception]] = field(default_factory=lambda: [
        ValueError, TypeError, NonRetryableException, KeyboardInterrupt
    ])
    backoff_multiplier: float = 1.0
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5


@dataclass
class RetryResult:
    """Result of retry operation."""
    success: bool
    attempts_made: int
    total_duration: float
    result: Any = None
    last_exception: Optional[Exception] = None
    retry_history: List[Dict[str, Any]] = field(default_factory=list)


class RetryMechanisms:
    """Comprehensive retry mechanism with various strategies."""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        self.logger = logging.getLogger(__name__)
        self.monitor = get_monitor()
        self.operation_stats: Dict[str, Dict] = {}
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay based on retry strategy."""
        if self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * attempt
        elif self.config.strategy == RetryStrategy.FIXED_INTERVAL:
            delay = self.config.base_delay
        elif self.config.strategy == RetryStrategy.FIBONACCI:
            delay = self.config.base_delay * self._fibonacci(attempt + 1)
        else:
            delay = self.config.base_delay
        
        # Apply backoff multiplier
        delay *= self.config.backoff_multiplier
        
        # Cap at max delay
        delay = min(delay, self.config.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            jitter_amount = delay * self.config.jitter_range
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)
    
    def _fibonacci(self, n: int) -> int:
        """Calculate Fibonacci number for Fibonacci backoff strategy."""
        if n <= 1:
            return n
        return self._fibonacci(n - 1) + self._fibonacci(n - 2)
    
    def _is_retryable_exception(self, exception: Exception) -> bool:
        """Determine if exception should trigger a retry."""
        # Check non-retryable first (takes precedence)
        if any(isinstance(exception, exc_type) for exc_type in self.config.non_retryable_exceptions):
            return False
        
        # Check retryable
        return any(isinstance(exception, exc_type) for exc_type in self.config.retryable_exceptions)
    
    def _update_operation_stats(self, operation_name: str, success: bool, 
                              attempts: int, duration: float):
        """Update operation statistics for monitoring."""
        if operation_name not in self.operation_stats:
            self.operation_stats[operation_name] = {
                'total_calls': 0,
                'successful_calls': 0,
                'failed_calls': 0,
                'total_attempts': 0,
                'total_duration': 0.0,
                'average_attempts': 0.0,
                'success_rate': 0.0
            }
        
        stats = self.operation_stats[operation_name]
        stats['total_calls'] += 1
        stats['total_attempts'] += attempts
        stats['total_duration'] += duration
        
        if success:
            stats['successful_calls'] += 1
        else:
            stats['failed_calls'] += 1
        
        # Update derived metrics
        stats['average_attempts'] = stats['total_attempts'] / stats['total_calls']
        stats['success_rate'] = stats['successful_calls'] / stats['total_calls']
    
    def retry_operation(self, func: Callable, operation_name: str = None, 
                       *args, **kwargs) -> RetryResult:
        """Execute function with retry logic."""
        operation_name = operation_name or func.__name__
        start_time = time.time()
        retry_history = []
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            attempt_start = time.time()
            
            try:
                with self.monitor.operation_timer(f"{operation_name}_attempt_{attempt + 1}"):
                    result = func(*args, **kwargs)
                
                # Success case
                total_duration = time.time() - start_time
                self._update_operation_stats(operation_name, True, attempt + 1, total_duration)
                
                self.logger.info(
                    f"Operation '{operation_name}' succeeded on attempt {attempt + 1}/{self.config.max_attempts}"
                )
                
                return RetryResult(
                    success=True,
                    attempts_made=attempt + 1,
                    total_duration=total_duration,
                    result=result,
                    retry_history=retry_history
                )
                
            except Exception as e:
                attempt_duration = time.time() - attempt_start
                last_exception = e
                
                retry_history.append({
                    'attempt': attempt + 1,
                    'exception': str(e),
                    'exception_type': type(e).__name__,
                    'duration': attempt_duration,
                    'timestamp': time.time()
                })
                
                # Check if this exception should trigger a retry
                if not self._is_retryable_exception(e):
                    self.logger.warning(
                        f"Non-retryable exception in '{operation_name}': {e}"
                    )
                    break
                
                # Check if we have more attempts
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    
                    self.logger.warning(
                        f"Operation '{operation_name}' failed on attempt {attempt + 1}/{self.config.max_attempts}. "
                        f"Retrying in {delay:.2f}s. Error: {e}"
                    )
                    
                    time.sleep(delay)
                else:
                    self.logger.error(
                        f"Operation '{operation_name}' failed after {self.config.max_attempts} attempts. "
                        f"Final error: {e}"
                    )
        
        # All attempts failed
        total_duration = time.time() - start_time
        self._update_operation_stats(operation_name, False, self.config.max_attempts, total_duration)
        
        return RetryResult(
            success=False,
            attempts_made=self.config.max_attempts,
            total_duration=total_duration,
            last_exception=last_exception,
            retry_history=retry_history
        )
    
    async def async_retry_operation(self, func: Callable, operation_name: str = None, 
                                  *args, **kwargs) -> RetryResult:
        """Execute async function with retry logic."""
        operation_name = operation_name or func.__name__
        start_time = time.time()
        retry_history = []
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            attempt_start = time.time()
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Success case
                total_duration = time.time() - start_time
                self._update_operation_stats(operation_name, True, attempt + 1, total_duration)
                
                return RetryResult(
                    success=True,
                    attempts_made=attempt + 1,
                    total_duration=total_duration,
                    result=result,
                    retry_history=retry_history
                )
                
            except Exception as e:
                attempt_duration = time.time() - attempt_start
                last_exception = e
                
                retry_history.append({
                    'attempt': attempt + 1,
                    'exception': str(e),
                    'exception_type': type(e).__name__,
                    'duration': attempt_duration,
                    'timestamp': time.time()
                })
                
                if not self._is_retryable_exception(e):
                    break
                
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    await asyncio.sleep(delay)
        
        # All attempts failed
        total_duration = time.time() - start_time
        self._update_operation_stats(operation_name, False, self.config.max_attempts, total_duration)
        
        return RetryResult(
            success=False,
            attempts_made=self.config.max_attempts,
            total_duration=total_duration,
            last_exception=last_exception,
            retry_history=retry_history
        )
    
    def get_operation_stats(self) -> Dict[str, Dict]:
        """Get retry statistics for all operations."""
        return self.operation_stats.copy()
    
    def reset_stats(self):
        """Reset operation statistics."""
        self.operation_stats.clear()


# Global retry mechanism instance
_default_retry_mechanism = RetryMechanisms()


def retry_with_exponential_backoff(config: RetryConfig = None):
    """Decorator for automatic retry with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        retry_config = config or RetryConfig()
        retry_mechanism = RetryMechanisms(retry_config)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = retry_mechanism.retry_operation(func, func.__name__, *args, **kwargs)
            
            if result.success:
                return result.result
            else:
                # Re-raise the last exception if all retries failed
                if result.last_exception:
                    raise result.last_exception
                else:
                    raise RuntimeError(f"Operation {func.__name__} failed after {result.attempts_made} attempts")
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await retry_mechanism.async_retry_operation(func, func.__name__, *args, **kwargs)
            
            if result.success:
                return result.result
            else:
                if result.last_exception:
                    raise result.last_exception
                else:
                    raise RuntimeError(f"Operation {func.__name__} failed after {result.attempts_made} attempts")
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def robust_file_operation(max_attempts: int = 3, base_delay: float = 1.0):
    """Decorator specifically for file operations with appropriate retry settings."""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        retryable_exceptions=[OSError, IOError, PermissionError, FileNotFoundError],
        non_retryable_exceptions=[ValueError, TypeError, IsADirectoryError]
    )
    return retry_with_exponential_backoff(config)


def robust_network_operation(max_attempts: int = 5, base_delay: float = 2.0):
    """Decorator specifically for network operations with appropriate retry settings."""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=30.0,
        retryable_exceptions=[ConnectionError, TimeoutError, OSError],
        non_retryable_exceptions=[ValueError, TypeError, KeyboardInterrupt]
    )
    return retry_with_exponential_backoff(config)


def get_retry_mechanism() -> RetryMechanisms:
    """Get the global retry mechanism instance."""
    return _default_retry_mechanism