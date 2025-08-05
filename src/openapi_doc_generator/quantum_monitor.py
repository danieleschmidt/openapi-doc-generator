"""Monitoring and health checking for quantum-inspired task planning."""

from __future__ import annotations

import time
import logging
import threading
import psutil
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json

from .quantum_scheduler import QuantumTask, QuantumScheduleResult

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class PerformanceMetrics:
    """Performance metrics for quantum planning operations."""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    memory_before_mb: Optional[float] = None
    memory_after_mb: Optional[float] = None
    memory_delta_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    task_count: Optional[int] = None
    quantum_fidelity: Optional[float] = None
    convergence_iterations: Optional[int] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    component: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class QuantumPlanningMonitor:
    """Monitor for quantum planning system performance and health."""
    
    def __init__(self, max_metrics_history: int = 1000):
        """Initialize monitoring system."""
        self.max_metrics_history = max_metrics_history
        self.metrics_history: List[PerformanceMetrics] = []
        self.active_operations: Dict[str, PerformanceMetrics] = {}
        self.health_checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self.alert_thresholds = {
            "max_duration_ms": 60000,  # 1 minute
            "max_memory_mb": 1000,     # 1GB
            "min_quantum_fidelity": 0.5,
            "max_cpu_percent": 80
        }
        self.lock = threading.Lock()
        
        # Register default health checks
        self._register_default_health_checks()
    
    def start_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start monitoring an operation."""
        operation_id = f"{operation_name}_{int(time.time() * 1000)}"
        
        try:
            process = psutil.Process()
            memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            memory_before = None
        
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=time.time(),
            memory_before_mb=memory_before,
            metadata=metadata or {}
        )
        
        with self.lock:
            self.active_operations[operation_id] = metrics
        
        logger.debug(f"Started monitoring operation: {operation_name} ({operation_id})")
        return operation_id
    
    def end_operation(self, operation_id: str, 
                     result: Optional[QuantumScheduleResult] = None,
                     error: Optional[str] = None) -> Optional[PerformanceMetrics]:
        """End monitoring an operation and record metrics."""
        with self.lock:
            if operation_id not in self.active_operations:
                logger.warning(f"Operation {operation_id} not found in active operations")
                return None
            
            metrics = self.active_operations.pop(operation_id)
        
        # Update metrics
        metrics.end_time = time.time()
        metrics.duration_ms = (metrics.end_time - metrics.start_time) * 1000
        metrics.error = error
        
        # Memory and CPU metrics
        try:
            process = psutil.Process()
            metrics.memory_after_mb = process.memory_info().rss / (1024 * 1024)
            if metrics.memory_before_mb:
                metrics.memory_delta_mb = metrics.memory_after_mb - metrics.memory_before_mb
            metrics.cpu_percent = process.cpu_percent()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        
        # Result-specific metrics
        if result:
            metrics.task_count = len(result.optimized_tasks)
            metrics.quantum_fidelity = result.quantum_fidelity
            metrics.convergence_iterations = result.convergence_iterations
        
        # Store in history
        with self.lock:
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_metrics_history:
                self.metrics_history.pop(0)
        
        # Check for alerts
        self._check_performance_alerts(metrics)
        
        logger.info(
            f"Operation completed: {metrics.operation_name} "
            f"({metrics.duration_ms:.2f}ms, "
            f"memory_delta: {metrics.memory_delta_mb:.2f}MB)"
        )
        
        return metrics
    
    def get_metrics_summary(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        with self.lock:
            if operation_name:
                relevant_metrics = [m for m in self.metrics_history 
                                  if m.operation_name == operation_name]
            else:
                relevant_metrics = self.metrics_history.copy()
        
        if not relevant_metrics:
            return {"message": "No metrics available"}
        
        # Calculate statistics
        durations = [m.duration_ms for m in relevant_metrics if m.duration_ms is not None]
        memory_deltas = [m.memory_delta_mb for m in relevant_metrics if m.memory_delta_mb is not None]
        fidelities = [m.quantum_fidelity for m in relevant_metrics if m.quantum_fidelity is not None]
        
        summary = {
            "total_operations": len(relevant_metrics),
            "operation_name": operation_name or "all",
            "time_range": {
                "start": min(m.start_time for m in relevant_metrics),
                "end": max(m.end_time for m in relevant_metrics if m.end_time)
            } if relevant_metrics else None
        }
        
        if durations:
            summary["duration_stats"] = {
                "avg_ms": sum(durations) / len(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "total_ms": sum(durations)
            }
        
        if memory_deltas:
            summary["memory_stats"] = {
                "avg_delta_mb": sum(memory_deltas) / len(memory_deltas),
                "max_delta_mb": max(memory_deltas),
                "min_delta_mb": min(memory_deltas)
            }
        
        if fidelities:
            summary["quantum_stats"] = {
                "avg_fidelity": sum(fidelities) / len(fidelities),
                "min_fidelity": min(fidelities),
                "max_fidelity": max(fidelities)
            }
        
        # Error statistics
        errors = [m for m in relevant_metrics if m.error]
        if errors:
            error_counts = defaultdict(int)
            for m in errors:
                error_counts[m.error] += 1
            summary["error_stats"] = dict(error_counts)
        
        return summary
    
    def run_health_checks(self) -> List[HealthCheckResult]:
        """Run all registered health checks."""
        results = []
        
        for check_name, check_func in self.health_checks.items():
            try:
                result = check_func()
                results.append(result)
            except Exception as e:
                logger.error(f"Health check {check_name} failed: {e}")
                results.append(HealthCheckResult(
                    component=check_name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(e)}"
                ))
        
        return results
    
    def get_system_status(self) -> HealthStatus:
        """Get overall system health status."""
        health_results = self.run_health_checks()
        
        if any(r.status == HealthStatus.CRITICAL for r in health_results):
            return HealthStatus.CRITICAL
        elif any(r.status == HealthStatus.WARNING for r in health_results):
            return HealthStatus.WARNING
        elif all(r.status == HealthStatus.HEALTHY for r in health_results):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def register_health_check(self, name: str, check_func: Callable[[], HealthCheckResult]) -> None:
        """Register a custom health check."""
        self.health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    def _register_default_health_checks(self) -> None:
        """Register default health checks."""
        
        def check_memory_usage() -> HealthCheckResult:
            try:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 * 1024)
                
                if memory_mb > 2000:  # 2GB
                    status = HealthStatus.CRITICAL
                    message = f"High memory usage: {memory_mb:.2f}MB"
                elif memory_mb > 1000:  # 1GB
                    status = HealthStatus.WARNING
                    message = f"Elevated memory usage: {memory_mb:.2f}MB"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Normal memory usage: {memory_mb:.2f}MB"
                
                return HealthCheckResult(
                    component="memory",
                    status=status,
                    message=message,
                    details={"memory_mb": memory_mb}
                )
            except Exception as e:
                return HealthCheckResult(
                    component="memory",
                    status=HealthStatus.UNKNOWN,
                    message=f"Unable to check memory: {e}"
                )
        
        def check_active_operations() -> HealthCheckResult:
            with self.lock:
                active_count = len(self.active_operations)
                oldest_operation = None
                
                if self.active_operations:
                    current_time = time.time()
                    oldest_start = min(op.start_time for op in self.active_operations.values())
                    oldest_duration = current_time - oldest_start
                    oldest_operation = oldest_duration
            
            if active_count > 10:
                status = HealthStatus.WARNING
                message = f"Many active operations: {active_count}"
            elif oldest_operation and oldest_operation > 300:  # 5 minutes
                status = HealthStatus.WARNING
                message = f"Long-running operation detected: {oldest_operation:.1f}s"
            else:
                status = HealthStatus.HEALTHY
                message = f"Active operations: {active_count}"
            
            return HealthCheckResult(
                component="active_operations",
                status=status,
                message=message,
                details={
                    "active_count": active_count,
                    "oldest_duration_s": oldest_operation
                }
            )
        
        def check_recent_errors() -> HealthCheckResult:
            with self.lock:
                recent_metrics = self.metrics_history[-100:]  # Last 100 operations
            
            recent_errors = [m for m in recent_metrics if m.error and m.end_time]
            
            if not recent_metrics:
                return HealthCheckResult(
                    component="recent_errors",
                    status=HealthStatus.HEALTHY,
                    message="No recent operations to analyze"
                )
            
            error_rate = len(recent_errors) / len(recent_metrics)
            
            if error_rate > 0.2:  # 20% error rate
                status = HealthStatus.CRITICAL
                message = f"High error rate: {error_rate:.1%}"
            elif error_rate > 0.1:  # 10% error rate
                status = HealthStatus.WARNING
                message = f"Elevated error rate: {error_rate:.1%}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Low error rate: {error_rate:.1%}"
            
            return HealthCheckResult(
                component="recent_errors",
                status=status,
                message=message,
                details={
                    "error_count": len(recent_errors),
                    "total_operations": len(recent_metrics),
                    "error_rate": error_rate
                }
            )
        
        self.register_health_check("memory", check_memory_usage)
        self.register_health_check("active_operations", check_active_operations)
        self.register_health_check("recent_errors", check_recent_errors)
    
    def _check_performance_alerts(self, metrics: PerformanceMetrics) -> None:
        """Check for performance alerts and log warnings."""
        alerts = []
        
        if metrics.duration_ms and metrics.duration_ms > self.alert_thresholds["max_duration_ms"]:
            alerts.append(f"Long duration: {metrics.duration_ms:.2f}ms")
        
        if metrics.memory_delta_mb and metrics.memory_delta_mb > self.alert_thresholds["max_memory_mb"]:
            alerts.append(f"High memory usage: {metrics.memory_delta_mb:.2f}MB")
        
        if metrics.quantum_fidelity and metrics.quantum_fidelity < self.alert_thresholds["min_quantum_fidelity"]:
            alerts.append(f"Low quantum fidelity: {metrics.quantum_fidelity:.3f}")
        
        if metrics.cpu_percent and metrics.cpu_percent > self.alert_thresholds["max_cpu_percent"]:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if alerts:
            logger.warning(f"Performance alerts for {metrics.operation_name}: {'; '.join(alerts)}")
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        with self.lock:
            metrics_data = []
            for m in self.metrics_history:
                metric_dict = {
                    "operation_name": m.operation_name,
                    "start_time": m.start_time,
                    "end_time": m.end_time,
                    "duration_ms": m.duration_ms,
                    "memory_before_mb": m.memory_before_mb,
                    "memory_after_mb": m.memory_after_mb,
                    "memory_delta_mb": m.memory_delta_mb,
                    "cpu_percent": m.cpu_percent,
                    "task_count": m.task_count,
                    "quantum_fidelity": m.quantum_fidelity,
                    "convergence_iterations": m.convergence_iterations,
                    "error": m.error,
                    "metadata": m.metadata
                }
                metrics_data.append(metric_dict)
        
        if format.lower() == "json":
            return json.dumps(metrics_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global monitor instance
_global_monitor = None


def get_monitor() -> QuantumPlanningMonitor:
    """Get global monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = QuantumPlanningMonitor()
    return _global_monitor


def monitor_operation(operation_name: str, metadata: Optional[Dict[str, Any]] = None):
    """Decorator to monitor quantum planning operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_monitor()
            operation_id = monitor.start_operation(operation_name, metadata)
            
            try:
                result = func(*args, **kwargs)
                
                # Extract quantum result if available
                quantum_result = None
                if isinstance(result, QuantumScheduleResult):
                    quantum_result = result
                elif hasattr(result, 'quantum_result'):
                    quantum_result = result.quantum_result
                
                monitor.end_operation(operation_id, quantum_result)
                return result
            
            except Exception as e:
                monitor.end_operation(operation_id, error=str(e))
                raise
        
        return wrapper
    return decorator