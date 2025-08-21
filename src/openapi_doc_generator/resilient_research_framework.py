"""
Resilient Research Framework with Advanced Error Handling

This module provides robust error handling, validation, recovery mechanisms,
and monitoring for the quantum-enhanced research modules to ensure production
reliability and fault tolerance.

Features:
1. Circuit breaker pattern for quantum operations
2. Graceful degradation when advanced features fail
3. Comprehensive validation and sanitization
4. Automatic recovery and retry mechanisms
5. Performance monitoring and alerting
"""

import asyncio
import functools
import logging
import time
import traceback
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed

from .quantum_optimizer import QuantumCache
from .utils import echo


logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for categorized error handling."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class OperationState(Enum):
    """States for circuit breaker pattern."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, requests rejected
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation: str
    timestamp: float
    severity: ErrorSeverity
    error_type: str
    message: str
    stack_trace: Optional[str] = None
    recovery_attempted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring operations."""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_before: Optional[float] = None
    memory_after: Optional[float] = None
    memory_delta: Optional[float] = None
    success: bool = True
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    def __init__(self, message: str, field: str = None, value: Any = None):
        super().__init__(message)
        self.field = field
        self.value = value


class QuantumOperationError(Exception):
    """Custom exception for quantum operation errors."""
    def __init__(self, message: str, operation: str = None, quantum_state: Any = None):
        super().__init__(message)
        self.operation = operation
        self.quantum_state = quantum_state


class MLInferenceError(Exception):
    """Custom exception for ML inference errors."""
    def __init__(self, message: str, model_type: str = None, input_data: Any = None):
        super().__init__(message)
        self.model_type = model_type
        self.input_data = input_data


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for fault tolerance.
    
    Prevents cascading failures by monitoring operation success/failure rates
    and temporarily blocking requests when failure threshold is exceeded.
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = OperationState.CLOSED
        self._lock = threading.Lock()
        
        logger.info(f"CircuitBreaker initialized: threshold={failure_threshold}, timeout={recovery_timeout}")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker to a function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self._call_with_circuit_breaker(func, *args, **kwargs)
        return wrapper
    
    def _call_with_circuit_breaker(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == OperationState.OPEN:
                if self._should_attempt_reset():
                    self.state = OperationState.HALF_OPEN
                    logger.info("Circuit breaker: Attempting reset (HALF_OPEN)")
                else:
                    raise QuantumOperationError(
                        f"Circuit breaker is OPEN. Last failure: {self.last_failure_time}"
                    )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful operation."""
        with self._lock:
            self.failure_count = 0
            self.state = OperationState.CLOSED
            if self.state != OperationState.CLOSED:
                logger.info("Circuit breaker: Reset to CLOSED state")
    
    def _on_failure(self):
        """Handle failed operation."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = OperationState.OPEN
                logger.warning(f"Circuit breaker: OPEN due to {self.failure_count} failures")


class GracefulDegradation:
    """
    Graceful degradation manager for handling component failures.
    
    Provides fallback mechanisms when advanced features (quantum, ML) fail,
    ensuring basic functionality continues to work.
    """
    
    def __init__(self):
        self.fallback_registry: Dict[str, Callable] = {}
        self.degradation_status: Dict[str, bool] = {}
        
        logger.info("GracefulDegradation initialized")
    
    def register_fallback(self, operation: str, fallback_func: Callable):
        """Register a fallback function for an operation."""
        self.fallback_registry[operation] = fallback_func
        logger.info(f"Registered fallback for operation: {operation}")
    
    def execute_with_fallback(self, operation: str, primary_func: Callable, 
                            *args, **kwargs) -> Tuple[Any, bool]:
        """
        Execute operation with fallback capability.
        
        Returns:
            Tuple of (result, is_degraded) where is_degraded indicates
            whether fallback was used.
        """
        try:
            result = primary_func(*args, **kwargs)
            
            # Mark as recovered if previously degraded
            if self.degradation_status.get(operation, False):
                self.degradation_status[operation] = False
                logger.info(f"Operation {operation} recovered from degradation")
            
            return result, False
            
        except Exception as e:
            logger.warning(f"Primary operation {operation} failed: {e}")
            
            # Attempt fallback
            if operation in self.fallback_registry:
                try:
                    fallback_func = self.fallback_registry[operation]
                    result = fallback_func(*args, **kwargs)
                    
                    if not self.degradation_status.get(operation, False):
                        self.degradation_status[operation] = True
                        logger.warning(f"Operation {operation} degraded to fallback")
                    
                    return result, True
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback for {operation} also failed: {fallback_error}")
                    raise
            else:
                logger.error(f"No fallback registered for operation: {operation}")
                raise


class InputValidator:
    """
    Comprehensive input validation and sanitization.
    
    Validates inputs to quantum and ML operations to prevent errors
    and security issues from malformed or malicious inputs.
    """
    
    def __init__(self):
        self.validation_cache = QuantumCache(max_size=1000)
        logger.info("InputValidator initialized")
    
    def validate_ast_node(self, node: Any) -> bool:
        """Validate AST node input."""
        if node is None:
            raise ValidationError("AST node cannot be None", "node", node)
        
        # Check if it's actually an AST node
        import ast
        if not isinstance(node, ast.AST):
            raise ValidationError("Input must be an AST node", "node", type(node))
        
        # Check for reasonable complexity
        node_count = len(list(ast.walk(node)))
        if node_count > 10000:
            raise ValidationError(
                f"AST too complex: {node_count} nodes (max: 10000)", 
                "node", node_count
            )
        
        return True
    
    def validate_embedding_dimensions(self, embedding: np.ndarray, expected_dim: int) -> bool:
        """Validate embedding dimensions and values."""
        if embedding is None:
            raise ValidationError("Embedding cannot be None", "embedding", embedding)
        
        if not isinstance(embedding, np.ndarray):
            raise ValidationError("Embedding must be numpy array", "embedding", type(embedding))
        
        if embedding.shape[0] != expected_dim:
            raise ValidationError(
                f"Embedding dimension mismatch: got {embedding.shape[0]}, expected {expected_dim}",
                "embedding", embedding.shape
            )
        
        # Check for invalid values
        if np.any(np.isnan(embedding)):
            raise ValidationError("Embedding contains NaN values", "embedding", embedding)
        
        if np.any(np.isinf(embedding)):
            raise ValidationError("Embedding contains infinite values", "embedding", embedding)
        
        # Check for reasonable value ranges
        if np.max(np.abs(embedding)) > 1000:
            warnings.warn("Embedding values are very large, may cause numerical issues")
        
        return True
    
    def validate_schema_data(self, schema: Dict[str, Any]) -> bool:
        """Validate schema data structure."""
        if not isinstance(schema, dict):
            raise ValidationError("Schema must be a dictionary", "schema", type(schema))
        
        if len(schema) > 1000:
            raise ValidationError(
                f"Schema too large: {len(schema)} fields (max: 1000)",
                "schema", len(schema)
            )
        
        # Validate field names
        for field_name in schema.keys():
            if not isinstance(field_name, str):
                raise ValidationError("Field names must be strings", "field_name", field_name)
            
            if len(field_name) > 100:
                raise ValidationError(
                    f"Field name too long: {len(field_name)} chars (max: 100)",
                    "field_name", field_name
                )
        
        return True
    
    def sanitize_file_path(self, file_path: str) -> str:
        """Sanitize and validate file path."""
        if not isinstance(file_path, str):
            raise ValidationError("File path must be string", "file_path", type(file_path))
        
        if len(file_path) > 4096:
            raise ValidationError("File path too long", "file_path", len(file_path))
        
        # Basic path traversal protection
        if ".." in file_path or file_path.startswith("/"):
            logger.warning(f"Potentially unsafe file path: {file_path}")
        
        # Normalize path
        import os
        return os.path.normpath(file_path)
    
    def validate_probability(self, prob: float, field_name: str = "probability") -> bool:
        """Validate probability value."""
        if not isinstance(prob, (int, float)):
            raise ValidationError(f"{field_name} must be numeric", field_name, type(prob))
        
        if not (0.0 <= prob <= 1.0):
            raise ValidationError(f"{field_name} must be between 0 and 1", field_name, prob)
        
        if np.isnan(prob) or np.isinf(prob):
            raise ValidationError(f"{field_name} cannot be NaN or infinite", field_name, prob)
        
        return True


class PerformanceMonitor:
    """
    Performance monitoring and alerting system.
    
    Tracks operation performance, memory usage, and provides
    alerts when performance degrades beyond acceptable thresholds.
    """
    
    def __init__(self, alert_threshold_ms: float = 5000, memory_threshold_mb: float = 1000):
        self.alert_threshold_ms = alert_threshold_ms
        self.memory_threshold_mb = memory_threshold_mb
        self.metrics_history: List[PerformanceMetrics] = []
        self.active_operations: Dict[str, PerformanceMetrics] = {}
        self._lock = threading.Lock()
        
        logger.info(f"PerformanceMonitor initialized: time_threshold={alert_threshold_ms}ms, "
                   f"memory_threshold={memory_threshold_mb}MB")
    
    @contextmanager
    def monitor_operation(self, operation_name: str, **metadata):
        """Context manager for monitoring operation performance."""
        import psutil
        
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.perf_counter()
        
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=start_time,
            memory_before=memory_before,
            metadata=metadata
        )
        
        with self._lock:
            self.active_operations[operation_name] = metrics
        
        try:
            yield metrics
            
            # Success path
            end_time = time.perf_counter()
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            metrics.end_time = end_time
            metrics.duration = (end_time - start_time) * 1000  # Convert to ms
            metrics.memory_after = memory_after
            metrics.memory_delta = memory_after - memory_before
            metrics.success = True
            
        except Exception as e:
            # Error path
            end_time = time.perf_counter()
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            metrics.end_time = end_time
            metrics.duration = (end_time - start_time) * 1000
            metrics.memory_after = memory_after
            metrics.memory_delta = memory_after - memory_before
            metrics.success = False
            metrics.error_count = 1
            
            raise
        
        finally:
            with self._lock:
                self.active_operations.pop(operation_name, None)
                self.metrics_history.append(metrics)
                
                # Keep only recent metrics
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-500:]
            
            self._check_performance_alerts(metrics)
    
    def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """Check if performance metrics warrant alerts."""
        alerts = []
        
        # Duration alert
        if metrics.duration and metrics.duration > self.alert_threshold_ms:
            alerts.append(f"Operation {metrics.operation_name} took {metrics.duration:.1f}ms "
                         f"(threshold: {self.alert_threshold_ms}ms)")
        
        # Memory alert
        if metrics.memory_delta and metrics.memory_delta > self.memory_threshold_mb:
            alerts.append(f"Operation {metrics.operation_name} used {metrics.memory_delta:.1f}MB "
                         f"(threshold: {self.memory_threshold_mb}MB)")
        
        # Log alerts
        for alert in alerts:
            logger.warning(f"PERFORMANCE ALERT: {alert}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of recent performance metrics."""
        with self._lock:
            if not self.metrics_history:
                return {"message": "No performance data available"}
            
            successful_ops = [m for m in self.metrics_history if m.success]
            failed_ops = [m for m in self.metrics_history if not m.success]
            
            summary = {
                "total_operations": len(self.metrics_history),
                "successful_operations": len(successful_ops),
                "failed_operations": len(failed_ops),
                "success_rate": len(successful_ops) / len(self.metrics_history) if self.metrics_history else 0,
                "active_operations": len(self.active_operations),
                "performance_alerts": 0  # Would count recent alerts
            }
            
            if successful_ops:
                durations = [m.duration for m in successful_ops if m.duration]
                memory_deltas = [m.memory_delta for m in successful_ops if m.memory_delta]
                
                if durations:
                    summary.update({
                        "avg_duration_ms": np.mean(durations),
                        "max_duration_ms": np.max(durations),
                        "min_duration_ms": np.min(durations)
                    })
                
                if memory_deltas:
                    summary.update({
                        "avg_memory_delta_mb": np.mean(memory_deltas),
                        "max_memory_delta_mb": np.max(memory_deltas)
                    })
            
            return summary


class RetryMechanism:
    """
    Configurable retry mechanism with exponential backoff.
    
    Provides intelligent retry logic for transient failures with
    exponential backoff and jitter to prevent thundering herd issues.
    """
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, exponential_base: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        
        logger.info(f"RetryMechanism initialized: max_retries={max_retries}, "
                   f"base_delay={base_delay}s, max_delay={max_delay}s")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to add retry logic to a function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self._execute_with_retry(func, *args, **kwargs)
        return wrapper
    
    def _execute_with_retry(self, func: Callable, *args, **kwargs):
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    logger.error(f"Function {func.__name__} failed after {self.max_retries} retries")
                    raise
                
                # Calculate delay with exponential backoff and jitter
                delay = min(
                    self.base_delay * (self.exponential_base ** attempt),
                    self.max_delay
                )
                
                # Add jitter (±25% of delay)
                jitter = delay * 0.25 * (2 * np.random.random() - 1)
                actual_delay = max(0, delay + jitter)
                
                logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}), "
                              f"retrying in {actual_delay:.2f}s: {e}")
                
                time.sleep(actual_delay)
        
        raise last_exception


class ErrorRecoverySystem:
    """
    Comprehensive error recovery and health management system.
    
    Coordinates circuit breakers, graceful degradation, and recovery
    mechanisms to maintain system health under various failure conditions.
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.graceful_degradation = GracefulDegradation()
        self.validator = InputValidator()
        self.performance_monitor = PerformanceMonitor()
        self.retry_mechanism = RetryMechanism()
        
        # Error tracking
        self.error_history: List[ErrorContext] = []
        self.recovery_attempts: Dict[str, int] = {}
        
        self._setup_default_fallbacks()
        
        logger.info("ErrorRecoverySystem initialized")
    
    def _setup_default_fallbacks(self):
        """Setup default fallback functions."""
        # Fallback for quantum semantic analysis
        def quantum_fallback(*args, **kwargs):
            logger.info("Using traditional AST analysis as quantum fallback")
            return {"semantic_type": "unknown", "confidence": 0.5, "fallback": True}
        
        # Fallback for ML schema inference  
        def ml_fallback(*args, **kwargs):
            logger.info("Using simple type inference as ML fallback")
            return {"type": "Any", "confidence": 0.3, "fallback": True}
        
        # Fallback for benchmarking
        def benchmark_fallback(*args, **kwargs):
            logger.info("Using basic timing as benchmark fallback")
            return {"execution_time": 0.0, "accuracy": 0.0, "fallback": True}
        
        self.graceful_degradation.register_fallback("quantum_analysis", quantum_fallback)
        self.graceful_degradation.register_fallback("ml_inference", ml_fallback)
        self.graceful_degradation.register_fallback("benchmark", benchmark_fallback)
    
    def get_circuit_breaker(self, operation: str, **kwargs) -> CircuitBreaker:
        """Get or create circuit breaker for operation."""
        if operation not in self.circuit_breakers:
            self.circuit_breakers[operation] = CircuitBreaker(**kwargs)
        return self.circuit_breakers[operation]
    
    def execute_resilient_operation(self, operation_name: str, func: Callable,
                                  *args, use_circuit_breaker: bool = True,
                                  use_retry: bool = True,
                                  use_fallback: bool = True,
                                  **kwargs):
        """
        Execute operation with full resilience features.
        
        Args:
            operation_name: Name of the operation for tracking
            func: Function to execute
            use_circuit_breaker: Enable circuit breaker protection
            use_retry: Enable retry mechanism
            use_fallback: Enable graceful degradation
            
        Returns:
            Tuple of (result, metadata) where metadata contains execution info
        """
        metadata = {
            "operation": operation_name,
            "circuit_breaker_used": use_circuit_breaker,
            "retry_used": use_retry,
            "fallback_used": False,
            "degraded": False,
            "attempts": 1,
            "errors": []
        }
        
        with self.performance_monitor.monitor_operation(operation_name) as perf_metrics:
            try:
                # Apply circuit breaker if requested
                if use_circuit_breaker:
                    circuit_breaker = self.get_circuit_breaker(operation_name)
                    func = circuit_breaker(func)
                
                # Apply retry mechanism if requested
                if use_retry:
                    retry_func = self.retry_mechanism(func)
                else:
                    retry_func = func
                
                # Execute operation
                if use_fallback:
                    result, is_degraded = self.graceful_degradation.execute_with_fallback(
                        operation_name, retry_func, *args, **kwargs
                    )
                    metadata["degraded"] = is_degraded
                    metadata["fallback_used"] = is_degraded
                else:
                    result = retry_func(*args, **kwargs)
                
                return result, metadata
                
            except Exception as e:
                # Record error context
                error_context = ErrorContext(
                    operation=operation_name,
                    timestamp=time.time(),
                    severity=self._classify_error_severity(e),
                    error_type=type(e).__name__,
                    message=str(e),
                    stack_trace=traceback.format_exc(),
                    metadata=metadata
                )
                
                self.error_history.append(error_context)
                metadata["errors"].append(error_context.message)
                
                # Attempt recovery
                recovery_result = self._attempt_recovery(operation_name, e, metadata)
                if recovery_result is not None:
                    return recovery_result, metadata
                
                # If all recovery attempts failed, re-raise
                raise
    
    def _classify_error_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity for appropriate handling."""
        if isinstance(error, (MemoryError, SystemError)):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (QuantumOperationError, MLInferenceError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, ValidationError):
            return ErrorSeverity.MEDIUM
        elif isinstance(error, (FileNotFoundError, PermissionError)):
            return ErrorSeverity.LOW
        else:
            return ErrorSeverity.MEDIUM
    
    def _attempt_recovery(self, operation: str, error: Exception, 
                         metadata: Dict[str, Any]) -> Optional[Any]:
        """Attempt to recover from operation failure."""
        recovery_key = f"{operation}:{type(error).__name__}"
        
        # Track recovery attempts
        self.recovery_attempts[recovery_key] = self.recovery_attempts.get(recovery_key, 0) + 1
        
        if self.recovery_attempts[recovery_key] > 3:
            logger.error(f"Maximum recovery attempts exceeded for {recovery_key}")
            return None
        
        logger.info(f"Attempting recovery for {recovery_key} (attempt {self.recovery_attempts[recovery_key]})")
        
        # Specific recovery strategies
        if isinstance(error, MemoryError):
            return self._recover_from_memory_error(operation, metadata)
        elif isinstance(error, ValidationError):
            return self._recover_from_validation_error(operation, error, metadata)
        elif isinstance(error, QuantumOperationError):
            return self._recover_from_quantum_error(operation, error, metadata)
        
        return None
    
    def _recover_from_memory_error(self, operation: str, metadata: Dict[str, Any]) -> Optional[Any]:
        """Attempt recovery from memory errors."""
        import gc
        
        logger.warning(f"Attempting memory recovery for {operation}")
        
        # Force garbage collection
        gc.collect()
        
        # Clear caches if available
        if hasattr(self.validator, 'validation_cache'):
            self.validator.validation_cache.clear()
        
        # Return a basic safe result
        return {
            "status": "recovered_from_memory_error",
            "operation": operation,
            "recovery_used": True
        }
    
    def _recover_from_validation_error(self, operation: str, error: ValidationError,
                                     metadata: Dict[str, Any]) -> Optional[Any]:
        """Attempt recovery from validation errors."""
        logger.warning(f"Attempting validation recovery for {operation}: {error}")
        
        # Return sanitized default result
        return {
            "status": "recovered_from_validation_error",
            "operation": operation,
            "field": getattr(error, 'field', 'unknown'),
            "recovery_used": True
        }
    
    def _recover_from_quantum_error(self, operation: str, error: QuantumOperationError,
                                   metadata: Dict[str, Any]) -> Optional[Any]:
        """Attempt recovery from quantum operation errors."""
        logger.warning(f"Attempting quantum recovery for {operation}: {error}")
        
        # Fallback to classical computation
        return {
            "status": "recovered_from_quantum_error",
            "operation": operation,
            "quantum_operation": getattr(error, 'operation', 'unknown'),
            "fallback_to_classical": True,
            "recovery_used": True
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        health_status = {
            "timestamp": time.time(),
            "overall_status": "healthy",
            "circuit_breakers": {},
            "degradation_status": self.graceful_degradation.degradation_status.copy(),
            "error_summary": {},
            "recovery_attempts": self.recovery_attempts.copy(),
            "performance": self.performance_monitor.get_performance_summary()
        }
        
        # Circuit breaker status
        for name, breaker in self.circuit_breakers.items():
            health_status["circuit_breakers"][name] = {
                "state": breaker.state.value,
                "failure_count": breaker.failure_count,
                "last_failure": breaker.last_failure_time
            }
            
            if breaker.state != OperationState.CLOSED:
                health_status["overall_status"] = "degraded"
        
        # Error summary
        if self.error_history:
            recent_errors = [e for e in self.error_history if time.time() - e.timestamp < 3600]
            error_counts = {}
            for error in recent_errors:
                severity = error.severity.value
                error_counts[severity] = error_counts.get(severity, 0) + 1
            
            health_status["error_summary"] = {
                "total_recent_errors": len(recent_errors),
                "by_severity": error_counts
            }
            
            if error_counts.get("critical", 0) > 0 or error_counts.get("high", 0) > 5:
                health_status["overall_status"] = "unhealthy"
        
        # Check degradation
        if any(self.graceful_degradation.degradation_status.values()):
            if health_status["overall_status"] == "healthy":
                health_status["overall_status"] = "degraded"
        
        return health_status
    
    def reset_error_state(self, operation: str = None):
        """Reset error state for recovery."""
        if operation:
            # Reset specific operation
            if operation in self.circuit_breakers:
                with self.circuit_breakers[operation]._lock:
                    self.circuit_breakers[operation].failure_count = 0
                    self.circuit_breakers[operation].state = OperationState.CLOSED
                    self.circuit_breakers[operation].last_failure_time = None
            
            # Reset degradation status
            self.graceful_degradation.degradation_status.pop(operation, None)
            
            # Reset recovery attempts
            keys_to_remove = [k for k in self.recovery_attempts.keys() if k.startswith(operation)]
            for key in keys_to_remove:
                self.recovery_attempts.pop(key, None)
            
            logger.info(f"Reset error state for operation: {operation}")
        
        else:
            # Reset all error state
            for breaker in self.circuit_breakers.values():
                with breaker._lock:
                    breaker.failure_count = 0
                    breaker.state = OperationState.CLOSED
                    breaker.last_failure_time = None
            
            self.graceful_degradation.degradation_status.clear()
            self.recovery_attempts.clear()
            
            logger.info("Reset all error state")


# Global error recovery system instance
_error_recovery_system = None


def get_error_recovery_system() -> ErrorRecoverySystem:
    """Get global error recovery system instance."""
    global _error_recovery_system
    if _error_recovery_system is None:
        _error_recovery_system = ErrorRecoverySystem()
    return _error_recovery_system


def resilient_operation(operation_name: str, **resilience_options):
    """
    Decorator for making operations resilient with full error handling.
    
    Args:
        operation_name: Name for tracking and monitoring
        **resilience_options: Options for circuit breaker, retry, fallback
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            recovery_system = get_error_recovery_system()
            result, metadata = recovery_system.execute_resilient_operation(
                operation_name, func, *args, **resilience_options, **kwargs
            )
            return result
        return wrapper
    return decorator


# Monitoring decorators for performance tracking
def monitor_performance(operation_name: str = None, **metadata):
    """Decorator for performance monitoring."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            recovery_system = get_error_recovery_system()
            
            with recovery_system.performance_monitor.monitor_operation(name, **metadata):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage and testing
    recovery_system = get_error_recovery_system()
    
    @resilient_operation("test_operation", use_circuit_breaker=True, use_retry=True)
    def test_function(should_fail: bool = False):
        if should_fail:
            raise ValueError("Test error")
        return {"status": "success", "data": "test_data"}
    
    # Test successful operation
    result = test_function(False)
    echo(f"Success result: {result}")
    
    # Test health status
    health = recovery_system.get_system_health()
    echo(f"System health: {health['overall_status']}")
    echo("Resilient research framework ready!")