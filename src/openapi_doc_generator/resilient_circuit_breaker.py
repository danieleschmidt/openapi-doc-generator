"""
Resilient Circuit Breaker System

This module implements advanced circuit breaker patterns for fault tolerance,
automatic recovery, and system resilience under failure conditions.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Callable, Dict, Optional


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failures detected, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5          # Failures before opening
    recovery_timeout: float = 60.0      # Seconds before trying half-open
    success_threshold: int = 3          # Successes to close from half-open
    timeout: float = 30.0               # Request timeout
    sliding_window_size: int = 100      # Size of metrics window
    min_request_threshold: int = 10     # Min requests before considering failures


@dataclass
class CircuitMetrics:
    """Metrics for circuit breaker operation."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    circuit_opens: int = 0
    circuit_closes: int = 0
    half_opens: int = 0
    last_failure_time: Optional[float] = None
    recent_results: deque = field(default_factory=lambda: deque(maxlen=100))


class ResilientCircuitBreaker:
    """Advanced circuit breaker with adaptive behavior."""

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.metrics = CircuitMetrics()
        self.lock = Lock()
        self.logger = logging.getLogger(f"CircuitBreaker.{name}")

    def __call__(self, func: Callable) -> Callable:
        """Decorator usage for circuit breaker."""
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        wrapper.__name__ = f"circuit_breaker_{func.__name__}"
        return wrapper

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.metrics.half_opens += 1
                    self.logger.info(f"Circuit breaker {self.name} entering HALF_OPEN state")
                else:
                    self.logger.warning(f"Circuit breaker {self.name} is OPEN, rejecting request")
                    raise CircuitBreakerError(
                        f"Circuit breaker {self.name} is OPEN. "
                        f"Last failure: {self.metrics.last_failure_time}"
                    )

        # Execute the function
        start_time = time.time()
        try:
            result = self._execute_with_timeout(func, *args, **kwargs)
            self._record_success(time.time() - start_time)
            return result

        except TimeoutError:
            self._record_timeout(time.time() - start_time)
            raise
        except Exception as e:
            self._record_failure(time.time() - start_time, e)
            raise

    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection."""
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.metrics.half_opens += 1
                    self.logger.info(f"Circuit breaker {self.name} entering HALF_OPEN state")
                else:
                    self.logger.warning(f"Circuit breaker {self.name} is OPEN, rejecting async request")
                    raise CircuitBreakerError(f"Circuit breaker {self.name} is OPEN")

        # Execute the async function
        start_time = time.time()
        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            self._record_success(time.time() - start_time)
            return result

        except asyncio.TimeoutError:
            self._record_timeout(time.time() - start_time)
            raise TimeoutError(f"Function {func.__name__} timed out after {self.config.timeout}s")
        except Exception as e:
            self._record_failure(time.time() - start_time, e)
            raise

    def _execute_with_timeout(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with timeout protection."""
        # For simplicity, we'll implement a basic timeout
        # In production, you might want to use threading.Timer or signal
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Function {func.__name__} timed out after {self.config.timeout}s")

        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(self.config.timeout))

        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # Cancel the alarm
            return result
        finally:
            signal.signal(signal.SIGALRM, old_handler)

    def _record_success(self, execution_time: float):
        """Record successful execution."""
        with self.lock:
            self.metrics.total_requests += 1
            self.metrics.successful_requests += 1
            self.metrics.recent_results.append(True)

            if self.state == CircuitState.HALF_OPEN:
                consecutive_successes = sum(
                    1 for result in list(self.metrics.recent_results)[-self.config.success_threshold:]
                    if result
                )
                if consecutive_successes >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.metrics.circuit_closes += 1
                    self.logger.info(f"Circuit breaker {self.name} closed after successful recovery")

    def _record_failure(self, execution_time: float, exception: Exception):
        """Record failed execution."""
        with self.lock:
            self.metrics.total_requests += 1
            self.metrics.failed_requests += 1
            self.metrics.last_failure_time = time.time()
            self.metrics.recent_results.append(False)

            self.logger.warning(f"Circuit breaker {self.name} recorded failure: {exception}")

            if self.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN]:
                if self._should_open_circuit():
                    self.state = CircuitState.OPEN
                    self.metrics.circuit_opens += 1
                    self.logger.error(f"Circuit breaker {self.name} opened due to failures")

    def _record_timeout(self, execution_time: float):
        """Record timeout execution."""
        with self.lock:
            self.metrics.total_requests += 1
            self.metrics.timeout_requests += 1
            self.metrics.last_failure_time = time.time()
            self.metrics.recent_results.append(False)

            self.logger.warning(f"Circuit breaker {self.name} recorded timeout")

            if self.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN]:
                if self._should_open_circuit():
                    self.state = CircuitState.OPEN
                    self.metrics.circuit_opens += 1
                    self.logger.error(f"Circuit breaker {self.name} opened due to timeouts")

    def _should_open_circuit(self) -> bool:
        """Determine if circuit should open based on failure rate."""
        recent_results = list(self.metrics.recent_results)

        if len(recent_results) < self.config.min_request_threshold:
            return False

        recent_failures = sum(1 for result in recent_results if not result)
        failure_rate = recent_failures / len(recent_results)

        return recent_failures >= self.config.failure_threshold or failure_rate > 0.5

    def _should_attempt_reset(self) -> bool:
        """Determine if we should attempt to reset from OPEN to HALF_OPEN."""
        if self.metrics.last_failure_time is None:
            return True

        time_since_failure = time.time() - self.metrics.last_failure_time
        return time_since_failure >= self.config.recovery_timeout

    def get_metrics(self) -> Dict[str, Any]:
        """Get current circuit breaker metrics."""
        with self.lock:
            failure_rate = (
                self.metrics.failed_requests / max(self.metrics.total_requests, 1)
            )
            recent_results = list(self.metrics.recent_results)
            recent_failure_rate = (
                sum(1 for r in recent_results if not r) / max(len(recent_results), 1)
            )

            return {
                "name": self.name,
                "state": self.state.value,
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "timeout_requests": self.metrics.timeout_requests,
                "failure_rate": failure_rate,
                "recent_failure_rate": recent_failure_rate,
                "circuit_opens": self.metrics.circuit_opens,
                "circuit_closes": self.metrics.circuit_closes,
                "half_opens": self.metrics.half_opens,
                "last_failure_time": self.metrics.last_failure_time,
                "time_in_current_state": time.time() - (self.metrics.last_failure_time or 0),
            }

    def reset(self):
        """Manually reset circuit breaker to closed state."""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.metrics.recent_results.clear()
            self.logger.info(f"Circuit breaker {self.name} manually reset")


class CircuitBreakerError(Exception):
    """Raised when circuit breaker blocks a request."""
    pass


class CircuitBreakerManager:
    """Manages multiple circuit breakers."""

    def __init__(self):
        self.circuit_breakers: Dict[str, ResilientCircuitBreaker] = {}
        self.lock = Lock()

    def get_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> ResilientCircuitBreaker:
        """Get or create circuit breaker by name."""
        with self.lock:
            if name not in self.circuit_breakers:
                self.circuit_breakers[name] = ResilientCircuitBreaker(name, config)
            return self.circuit_breakers[name]

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers."""
        with self.lock:
            return {
                name: breaker.get_metrics()
                for name, breaker in self.circuit_breakers.items()
            }

    def reset_all(self):
        """Reset all circuit breakers."""
        with self.lock:
            for breaker in self.circuit_breakers.values():
                breaker.reset()


# Global circuit breaker manager
_global_manager: Optional[CircuitBreakerManager] = None


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get global circuit breaker manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = CircuitBreakerManager()
    return _global_manager


def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator for applying circuit breaker to functions."""
    def decorator(func):
        breaker = get_circuit_breaker_manager().get_breaker(name, config)
        return breaker(func)
    return decorator


def resilient_operation(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Context manager for resilient operations."""
    from contextlib import contextmanager

    @contextmanager
    def context():
        breaker = get_circuit_breaker_manager().get_breaker(name, config)
        yield breaker

    return context()
