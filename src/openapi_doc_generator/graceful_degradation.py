"""Graceful degradation system for handling non-critical failures."""

import functools
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional

from .enhanced_monitoring import get_monitor


class OperationCriticality(Enum):
    """Criticality levels for operations."""
    CRITICAL = "critical"        # Must succeed, failure stops execution
    IMPORTANT = "important"      # Should succeed, fallback on failure
    OPTIONAL = "optional"        # Nice to have, continue without on failure
    ENHANCEMENT = "enhancement"  # Pure optimization, graceful fallback always


@dataclass
class DegradationPolicy:
    """Policy for handling operation degradation."""
    criticality: OperationCriticality
    fallback_value: Any = None
    fallback_function: Optional[Callable] = None
    warning_message: str = ""
    max_failures_before_disable: int = 5
    disable_duration_seconds: float = 300.0  # 5 minutes
    enable_monitoring: bool = True
    custom_exception_handler: Optional[Callable] = None


@dataclass
class DegradationState:
    """State tracking for degraded operations."""
    operation_name: str
    is_disabled: bool = False
    failure_count: int = 0
    last_failure_time: float = 0.0
    disabled_until: float = 0.0
    total_degradations: int = 0
    successful_recoveries: int = 0
    policy: Optional[DegradationPolicy] = None


class GracefulDegradationManager:
    """Manager for graceful degradation across the application."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.monitor = get_monitor()
        self.degradation_states: Dict[str, DegradationState] = {}
        self.global_policies: Dict[str, DegradationPolicy] = {}
        self._initialize_default_policies()

    def _initialize_default_policies(self):
        """Initialize default degradation policies."""
        self.global_policies.update({
            "schema_inference": DegradationPolicy(
                criticality=OperationCriticality.IMPORTANT,
                fallback_value=[],
                warning_message="Schema inference failed. Documentation generated without detailed type information."
            ),
            "plugin_loading": DegradationPolicy(
                criticality=OperationCriticality.OPTIONAL,
                fallback_value=None,
                warning_message="Plugin loading failed. Using core functionality only."
            ),
            "performance_optimization": DegradationPolicy(
                criticality=OperationCriticality.ENHANCEMENT,
                fallback_value=None,
                warning_message="Performance optimization disabled. Using standard processing."
            ),
            "advanced_caching": DegradationPolicy(
                criticality=OperationCriticality.ENHANCEMENT,
                fallback_value=None,
                warning_message="Advanced caching unavailable. Using basic caching."
            ),
            "ml_analysis": DegradationPolicy(
                criticality=OperationCriticality.OPTIONAL,
                fallback_value={},
                warning_message="ML analysis failed. Documentation generated with basic analysis only."
            ),
            "graphql_introspection": DegradationPolicy(
                criticality=OperationCriticality.IMPORTANT,
                fallback_value={},
                warning_message="GraphQL introspection failed. Basic schema analysis used."
            ),
            "external_validation": DegradationPolicy(
                criticality=OperationCriticality.OPTIONAL,
                fallback_value=True,
                warning_message="External validation unavailable. Skipping validation checks."
            ),
        })

    def register_policy(self, operation_name: str, policy: DegradationPolicy):
        """Register a degradation policy for an operation."""
        self.global_policies[operation_name] = policy

        # Initialize state if not exists
        if operation_name not in self.degradation_states:
            self.degradation_states[operation_name] = DegradationState(
                operation_name=operation_name,
                policy=policy
            )

    def _get_state(self, operation_name: str) -> DegradationState:
        """Get or create degradation state for operation."""
        if operation_name not in self.degradation_states:
            policy = self.global_policies.get(operation_name)
            self.degradation_states[operation_name] = DegradationState(
                operation_name=operation_name,
                policy=policy
            )
        return self.degradation_states[operation_name]

    def _should_skip_operation(self, operation_name: str) -> bool:
        """Check if operation should be skipped due to previous failures."""
        state = self._get_state(operation_name)
        current_time = time.time()

        # Check if operation is temporarily disabled
        if state.is_disabled and current_time < state.disabled_until:
            return True

        # Re-enable if disable period has passed
        if state.is_disabled and current_time >= state.disabled_until:
            state.is_disabled = False
            self.logger.info(f"Re-enabling degraded operation: {operation_name}")
            if state.policy and state.policy.enable_monitoring:
                self.monitor.record_metric(f"{operation_name}_re_enabled", 1)

        return False

    def _handle_operation_failure(self, operation_name: str, exception: Exception,
                                policy: DegradationPolicy) -> Any:
        """Handle operation failure according to degradation policy."""
        state = self._get_state(operation_name)
        state.failure_count += 1
        state.last_failure_time = time.time()
        state.total_degradations += 1

        # Log the failure
        self.logger.warning(
            f"Operation '{operation_name}' failed (attempt {state.failure_count}): {exception}"
        )

        # Check if we should disable the operation
        if state.failure_count >= policy.max_failures_before_disable:
            state.is_disabled = True
            state.disabled_until = time.time() + policy.disable_duration_seconds

            self.logger.warning(
                f"Disabling operation '{operation_name}' for {policy.disable_duration_seconds}s "
                f"due to {state.failure_count} consecutive failures"
            )

            if policy.enable_monitoring:
                self.monitor.record_metric(f"{operation_name}_disabled", 1)

        # Handle based on criticality
        if policy.criticality == OperationCriticality.CRITICAL:
            # Critical operations must succeed
            raise exception

        # Show warning message if provided
        if policy.warning_message:
            self.logger.warning(policy.warning_message)

        # Try custom exception handler first
        if policy.custom_exception_handler:
            try:
                return policy.custom_exception_handler(exception)
            except Exception as handler_error:
                self.logger.error(f"Custom exception handler failed: {handler_error}")

        # Use fallback function if available
        if policy.fallback_function:
            try:
                self.logger.info(f"Using fallback function for '{operation_name}'")
                return policy.fallback_function()
            except Exception as fallback_error:
                self.logger.error(f"Fallback function failed: {fallback_error}")

        # Return fallback value
        self.logger.info(f"Using fallback value for '{operation_name}': {policy.fallback_value}")
        return policy.fallback_value

    def _handle_operation_success(self, operation_name: str):
        """Handle successful operation execution."""
        state = self._get_state(operation_name)

        # Reset failure count on success
        if state.failure_count > 0:
            self.logger.info(f"Operation '{operation_name}' recovered after {state.failure_count} failures")
            state.successful_recoveries += 1
            state.failure_count = 0

            if state.policy and state.policy.enable_monitoring:
                self.monitor.record_metric(f"{operation_name}_recovered", 1)

    def execute_with_degradation(self, operation_name: str, func: Callable,
                                policy: Optional[DegradationPolicy] = None,
                                *args, **kwargs) -> Any:
        """Execute function with graceful degradation."""
        # Use provided policy or global policy
        effective_policy = policy or self.global_policies.get(operation_name)
        if not effective_policy:
            # No policy defined, execute normally
            return func(*args, **kwargs)

        # Check if operation should be skipped
        if self._should_skip_operation(operation_name):
            self.logger.debug(f"Skipping disabled operation: {operation_name}")
            return self._handle_operation_failure(operation_name,
                                                Exception("Operation disabled"),
                                                effective_policy)

        # Execute the operation
        try:
            if effective_policy.enable_monitoring:
                with self.monitor.operation_timer(f"{operation_name}_degradation"):
                    result = func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Handle success
            self._handle_operation_success(operation_name)
            return result

        except Exception as e:
            # Handle failure according to policy
            return self._handle_operation_failure(operation_name, e, effective_policy)

    def get_degradation_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current degradation status for all operations."""
        current_time = time.time()
        status = {}

        for operation_name, state in self.degradation_states.items():
            status[operation_name] = {
                'is_disabled': state.is_disabled,
                'failure_count': state.failure_count,
                'total_degradations': state.total_degradations,
                'successful_recoveries': state.successful_recoveries,
                'disabled_until': state.disabled_until if state.is_disabled else None,
                'seconds_until_re_enable': max(0, state.disabled_until - current_time) if state.is_disabled else 0,
                'policy': {
                    'criticality': state.policy.criticality.value if state.policy else None,
                    'max_failures': state.policy.max_failures_before_disable if state.policy else None,
                } if state.policy else None
            }

        return status

    def reset_operation_state(self, operation_name: str):
        """Reset degradation state for an operation."""
        if operation_name in self.degradation_states:
            state = self.degradation_states[operation_name]
            policy = state.policy
            self.degradation_states[operation_name] = DegradationState(
                operation_name=operation_name,
                policy=policy
            )
            self.logger.info(f"Reset degradation state for: {operation_name}")

    def force_enable_operation(self, operation_name: str):
        """Force enable a disabled operation."""
        state = self._get_state(operation_name)
        state.is_disabled = False
        state.disabled_until = 0.0
        state.failure_count = 0
        self.logger.info(f"Force enabled operation: {operation_name}")


# Global degradation manager
_degradation_manager = GracefulDegradationManager()


def with_graceful_degradation(operation_name: str, policy: DegradationPolicy = None):
    """Decorator for graceful degradation."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return _degradation_manager.execute_with_degradation(
                operation_name, func, policy, *args, **kwargs
            )
        return wrapper
    return decorator


def get_degradation_manager() -> GracefulDegradationManager:
    """Get the global degradation manager."""
    return _degradation_manager


# Convenience decorators for common patterns
def optional_operation(operation_name: str, fallback_value: Any = None,
                      warning_message: str = ""):
    """Decorator for optional operations that can fail gracefully."""
    policy = DegradationPolicy(
        criticality=OperationCriticality.OPTIONAL,
        fallback_value=fallback_value,
        warning_message=warning_message
    )
    return with_graceful_degradation(operation_name, policy)


def enhancement_operation(operation_name: str, warning_message: str = ""):
    """Decorator for enhancement operations that should degrade gracefully."""
    policy = DegradationPolicy(
        criticality=OperationCriticality.ENHANCEMENT,
        fallback_value=None,
        warning_message=warning_message or f"Enhancement '{operation_name}' unavailable, using standard processing"
    )
    return with_graceful_degradation(operation_name, policy)


def important_operation_with_fallback(operation_name: str, fallback_function: Callable,
                                    warning_message: str = ""):
    """Decorator for important operations with custom fallback functions."""
    policy = DegradationPolicy(
        criticality=OperationCriticality.IMPORTANT,
        fallback_function=fallback_function,
        warning_message=warning_message
    )
    return with_graceful_degradation(operation_name, policy)
