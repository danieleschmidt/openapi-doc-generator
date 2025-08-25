"""
Circuit Breaker Pattern Implementation for Robust Operation

Provides automatic failure detection and recovery for all system operations,
preventing cascading failures and enabling graceful degradation.
"""

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict

from .enhanced_error_handling import ErrorContext, get_error_handler
from .enhanced_monitoring import get_monitor


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failure threshold exceeded, circuit open
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5         # Failures before opening circuit
    success_threshold: int = 3         # Successes needed to close circuit
    timeout: float = 60.0             # Seconds before trying half-open
    operation_timeout: float = 30.0   # Maximum operation time


class CircuitBreaker:
    """Circuit breaker implementation with monitoring and metrics."""

    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.RLock()

        self.error_handler = get_error_handler()
        self.monitor = get_monitor()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._state

    def is_available(self) -> bool:
        """Check if circuit is available for operations."""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            elif self._state == CircuitState.OPEN:
                # Check if timeout has passed
                if time.time() - self._last_failure_time >= self.config.timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    return True
                return False
            else:  # HALF_OPEN
                return True

    def record_success(self) -> None:
        """Record successful operation."""
        with self._lock:
            self._failure_count = 0

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    if hasattr(self.monitor, 'record_metric'):
                        self.monitor.record_metric(f"circuit_breaker.{self.name}.closed", 1, None)

    def record_failure(self, error: Exception) -> None:
        """Record failed operation."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                if hasattr(self.monitor, 'record_metric'):
                    self.monitor.record_metric(f"circuit_breaker.{self.name}.opened", 1, None)
            elif (self._state == CircuitState.CLOSED and
                  self._failure_count >= self.config.failure_threshold):
                self._state = CircuitState.OPEN
                if hasattr(self.monitor, 'record_metric'):
                    self.monitor.record_metric(f"circuit_breaker.{self.name}.opened", 1, None)

            # Log the failure with context
            context = ErrorContext(
                operation=f"circuit_breaker_{self.name}",
                component="circuit_breaker",
                performance_metrics={
                    "failure_count": self._failure_count,
                    "state": self._state.value
                }
            )
            # Log error with context
            if hasattr(self.error_handler, 'handle_exception'):
                self.error_handler.handle_exception(error, context)
            else:
                # Fallback to basic logging
                import logging
                logger = logging.getLogger("circuit_breaker")
                logger.error(f"Circuit breaker {self.name} failure: {str(error)}")

    @contextmanager
    def protect(self, operation_name: str = None):
        """Context manager for circuit breaker protection."""
        if not self.is_available():
            raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")

        operation_name = operation_name or "unknown_operation"
        start_time = time.time()

        try:
            start_time = time.time()
            yield
            duration = time.time() - start_time
            self.monitor.record_operation_time(f"circuit_breaker.{self.name}.{operation_name}", duration)

            # Record success
            self.record_success()
            if hasattr(self.monitor, 'record_metric'):
                self.monitor.record_metric(f"circuit_breaker.{self.name}.success", 1, None)

        except Exception as e:
            # Record failure
            self.record_failure(e)
            if hasattr(self.monitor, 'record_metric'):
                self.monitor.record_metric(f"circuit_breaker.{self.name}.failure", 1, None)

            # Add circuit breaker context to error
            if hasattr(e, '__circuit_breaker_context__'):
                e.__circuit_breaker_context__ = {
                    'circuit_name': self.name,
                    'state': self._state.value,
                    'failure_count': self._failure_count
                }

            raise

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self.protect(func.__name__):
            return func(*args, **kwargs)

    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics for monitoring."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure_time": self._last_failure_time,
                "is_available": self.is_available()
            }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and operation is blocked."""
    pass


# Global circuit breaker registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_registry_lock = threading.RLock()


def get_circuit_breaker(name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
    """Get or create a circuit breaker instance."""
    with _registry_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(name, config)
        return _circuit_breakers[name]


def protect_operation(circuit_name: str, config: CircuitBreakerConfig = None):
    """Decorator for protecting operations with circuit breakers."""
    def decorator(func: Callable) -> Callable:
        circuit = get_circuit_breaker(circuit_name, config)

        def wrapper(*args, **kwargs):
            return circuit.execute(func, *args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator


def get_all_circuit_breaker_metrics() -> Dict[str, Dict[str, Any]]:
    """Get metrics for all registered circuit breakers."""
    with _registry_lock:
        return {name: cb.get_metrics() for name, cb in _circuit_breakers.items()}
