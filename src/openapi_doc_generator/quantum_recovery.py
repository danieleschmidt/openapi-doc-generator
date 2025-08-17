"""Error recovery and resilience mechanisms for quantum task planning."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAIL_FAST = "fail_fast"
    CIRCUIT_BREAKER = "circuit_breaker"


@dataclass
class RecoveryContext:
    """Context information for error recovery."""
    operation_name: str
    attempt_count: int
    max_attempts: int
    error_history: list[str]
    recovery_strategy: RecoveryStrategy
    fallback_available: bool = False


class QuantumRecoveryManager:
    """Manages error recovery and resilience for quantum operations."""

    def __init__(self):
        """Initialize recovery manager."""
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.retry_policies: dict[str, RetryPolicy] = {}
        self.recovery_stats: dict[str, dict[str, Any]] = {}

        # Default recovery configurations
        self._setup_default_policies()

    def _setup_default_policies(self):
        """Setup default retry and circuit breaker policies."""
        # Quantum annealing retry policy
        self.retry_policies["quantum_annealing"] = RetryPolicy(
            max_attempts=3,
            base_delay=1.0,
            max_delay=10.0,
            exponential_backoff=True,
            jitter=True
        )

        # Resource allocation retry policy
        self.retry_policies["resource_allocation"] = RetryPolicy(
            max_attempts=5,
            base_delay=0.5,
            max_delay=5.0,
            exponential_backoff=True,
            jitter=False
        )

        # Validation operations
        self.retry_policies["validation"] = RetryPolicy(
            max_attempts=2,
            base_delay=0.1,
            max_delay=1.0,
            exponential_backoff=False,
            jitter=False
        )

        # Circuit breakers for external dependencies
        self.circuit_breakers["monitoring"] = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0,
            half_open_max_calls=3
        )

        self.circuit_breakers["optimization"] = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=60.0,
            half_open_max_calls=2
        )

    @contextmanager
    def resilient_execution(self,
                           operation_name: str,
                           strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
                           fallback: Callable | None = None):
        """Context manager for resilient operation execution."""
        context = RecoveryContext(
            operation_name=operation_name,
            attempt_count=0,
            max_attempts=self.retry_policies.get(operation_name, RetryPolicy()).max_attempts,
            error_history=[],
            recovery_strategy=strategy,
            fallback_available=fallback is not None
        )

        try:
            # Check circuit breaker if available
            if operation_name in self.circuit_breakers:
                circuit_breaker = self.circuit_breakers[operation_name]
                if not circuit_breaker.can_execute():
                    raise CircuitBreakerOpenError(f"Circuit breaker open for {operation_name}")

            yield context

            # Mark success in circuit breaker
            if operation_name in self.circuit_breakers:
                self.circuit_breakers[operation_name].record_success()

            # Update recovery stats
            self._update_recovery_stats(operation_name, "success", context)

        except Exception as e:
            context.error_history.append(str(e))

            # Mark failure in circuit breaker
            if operation_name in self.circuit_breakers:
                self.circuit_breakers[operation_name].record_failure()

            # Apply recovery strategy
            recovered_result = self._apply_recovery_strategy(context, e, fallback)
            if recovered_result is not None:
                logger.info(f"Recovery successful for {operation_name} using {strategy.value}")
                return recovered_result

            # Update recovery stats
            self._update_recovery_stats(operation_name, "failure", context)

            # Re-raise if no recovery possible
            raise

    def retry_with_backoff(self,
                          operation: Callable,
                          operation_name: str,
                          *args, **kwargs) -> Any:
        """Execute operation with retry and exponential backoff."""
        policy = self.retry_policies.get(operation_name, RetryPolicy())
        last_exception = None

        for attempt in range(policy.max_attempts):
            try:
                result = operation(*args, **kwargs)

                if attempt > 0:
                    logger.info(f"Operation {operation_name} succeeded on attempt {attempt + 1}")

                return result

            except Exception as e:
                last_exception = e

                if attempt < policy.max_attempts - 1:  # Not the last attempt
                    delay = policy.calculate_delay(attempt)
                    logger.warning(
                        f"Operation {operation_name} failed on attempt {attempt + 1}: {str(e)}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Operation {operation_name} failed after {policy.max_attempts} attempts")

        # All attempts failed
        if last_exception:
            raise last_exception

    def _apply_recovery_strategy(self,
                               context: RecoveryContext,
                               error: Exception,
                               fallback: Callable | None) -> Any:
        """Apply appropriate recovery strategy."""
        if context.recovery_strategy == RecoveryStrategy.RETRY:
            return self._handle_retry_recovery(context, error)

        elif context.recovery_strategy == RecoveryStrategy.FALLBACK:
            return self._handle_fallback_recovery(context, error, fallback)

        elif context.recovery_strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return self._handle_graceful_degradation(context, error)

        elif context.recovery_strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            return self._handle_circuit_breaker_recovery(context, error, fallback)

        elif context.recovery_strategy == RecoveryStrategy.FAIL_FAST:
            return None  # No recovery, fail immediately

        return None

    def _handle_retry_recovery(self, context: RecoveryContext, error: Exception) -> Any:
        """Handle retry-based recovery."""
        if context.attempt_count < context.max_attempts:
            policy = self.retry_policies.get(context.operation_name, RetryPolicy())
            delay = policy.calculate_delay(context.attempt_count)

            logger.warning(f"Retrying {context.operation_name} in {delay:.2f}s (attempt {context.attempt_count + 1})")
            time.sleep(delay)

            context.attempt_count += 1
            return "retry"

        return None

    def _handle_fallback_recovery(self,
                                context: RecoveryContext,
                                error: Exception,
                                fallback: Callable | None) -> Any:
        """Handle fallback-based recovery."""
        if fallback and context.fallback_available:
            try:
                logger.info(f"Executing fallback for {context.operation_name}")
                return fallback()
            except Exception as fallback_error:
                logger.error(f"Fallback failed for {context.operation_name}: {str(fallback_error)}")
                return None

        return None

    def _handle_graceful_degradation(self, context: RecoveryContext, error: Exception) -> Any:
        """Handle graceful degradation recovery."""
        logger.warning(f"Graceful degradation for {context.operation_name}: {str(error)}")

        # Return minimal viable result based on operation type
        if "quantum_annealing" in context.operation_name:
            # Return empty schedule with low fidelity
            from .quantum_scheduler import QuantumScheduleResult
            return QuantumScheduleResult([], 0.0, 0.0, 0.1, 0)

        elif "validation" in context.operation_name:
            # Return permissive validation result
            return [], True

        elif "monitoring" in context.operation_name:
            # Return basic metrics
            return {"status": "degraded", "metrics": {}}

        return {"status": "degraded", "message": f"Operation {context.operation_name} running in degraded mode"}

    def _handle_circuit_breaker_recovery(self,
                                       context: RecoveryContext,
                                       error: Exception,
                                       fallback: Callable | None) -> Any:
        """Handle circuit breaker recovery."""
        circuit_breaker = self.circuit_breakers.get(context.operation_name)

        if circuit_breaker and circuit_breaker.state == CircuitBreakerState.OPEN:
            logger.warning(f"Circuit breaker open for {context.operation_name}, attempting fallback")
            return self._handle_fallback_recovery(context, error, fallback)

        return None

    def _update_recovery_stats(self, operation_name: str, status: str, context: RecoveryContext):
        """Update recovery statistics."""
        if operation_name not in self.recovery_stats:
            self.recovery_stats[operation_name] = {
                "total_attempts": 0,
                "successful_recoveries": 0,
                "failed_recoveries": 0,
                "recovery_strategies_used": {},
                "average_attempts": 0.0
            }

        stats = self.recovery_stats[operation_name]
        stats["total_attempts"] += 1

        if status == "success":
            if context.attempt_count > 0:  # Was recovered
                stats["successful_recoveries"] += 1

                strategy = context.recovery_strategy.value
                if strategy not in stats["recovery_strategies_used"]:
                    stats["recovery_strategies_used"][strategy] = 0
                stats["recovery_strategies_used"][strategy] += 1
        else:
            stats["failed_recoveries"] += 1

        # Update average attempts
        total_operations = stats["successful_recoveries"] + stats["failed_recoveries"]
        if total_operations > 0:
            stats["average_attempts"] = stats["total_attempts"] / total_operations

    def get_recovery_statistics(self) -> dict[str, Any]:
        """Get comprehensive recovery statistics."""
        return {
            "recovery_stats": dict(self.recovery_stats),
            "circuit_breaker_states": {
                name: {
                    "state": cb.state.value,
                    "failure_count": cb.failure_count,
                    "last_failure_time": cb.last_failure_time
                }
                for name, cb in self.circuit_breakers.items()
            },
            "active_policies": {
                name: {
                    "max_attempts": policy.max_attempts,
                    "base_delay": policy.base_delay,
                    "max_delay": policy.max_delay
                }
                for name, policy in self.retry_policies.items()
            }
        }


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_backoff: bool = True
    jitter: bool = True

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        if self.exponential_backoff:
            delay = self.base_delay * (2 ** attempt)
        else:
            delay = self.base_delay

        # Apply max delay limit
        delay = min(delay, self.max_delay)

        # Add jitter if enabled
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # 50-100% of calculated delay

        return delay


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures."""

    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 half_open_max_calls: int = 3):
        """Initialize circuit breaker."""
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.failure_count = 0
        self.last_failure_time = 0.0
        self.half_open_calls = 0
        self.state = CircuitBreakerState.CLOSED

    def can_execute(self) -> bool:
        """Check if operation can be executed."""
        current_time = time.time()

        if self.state == CircuitBreakerState.CLOSED:
            return True

        elif self.state == CircuitBreakerState.OPEN:
            if current_time - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_calls = 0
                logger.info("Circuit breaker transitioning to half-open")
                return True
            return False

        elif self.state == CircuitBreakerState.HALF_OPEN:
            return self.half_open_calls < self.half_open_max_calls

        return False

    def record_success(self):
        """Record successful operation."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max_calls:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker closed - service recovered")
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)  # Gradual recovery

    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            logger.warning("Circuit breaker opened - service still failing")
        elif self.state == CircuitBreakerState.CLOSED and self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker opened - failure threshold ({self.failure_threshold}) exceeded")


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


# Global recovery manager instance
_recovery_manager = None


def get_recovery_manager() -> QuantumRecoveryManager:
    """Get global recovery manager instance."""
    global _recovery_manager
    if _recovery_manager is None:
        _recovery_manager = QuantumRecoveryManager()
    return _recovery_manager
