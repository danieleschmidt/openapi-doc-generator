"""
Enhanced Monitoring and Health Check System

This module provides comprehensive monitoring, health checks, metrics collection,
and alerting capabilities for the OpenAPI documentation generator.
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import psutil

from .enhanced_error_handling import get_error_handler


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class HealthCheck:
    """Individual health check configuration."""
    name: str
    check_function: Callable[[], bool]
    description: str
    critical: bool = False
    timeout: float = 5.0
    interval: float = 30.0


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert configuration and state."""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    message: str
    severity: str = "warning"
    cooldown: float = 300.0  # 5 minutes
    last_triggered: Optional[datetime] = None


class EnhancedMonitor:
    """Comprehensive monitoring and health checking system."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_handler = get_error_handler()

        # Health checks
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_results: Dict[str, bool] = {}
        self.health_last_check: Dict[str, datetime] = {}

        # Metrics
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.metric_aggregates: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Alerts
        self.alerts: Dict[str, Alert] = {}
        self.alert_callbacks: List[Callable[[Alert, Dict[str, Any]], None]] = []

        # System monitoring
        self.start_time = datetime.now()
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None

        # Performance tracking
        self.operation_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.resource_usage_history: deque = deque(maxlen=100)

        self._setup_default_health_checks()
        self._setup_default_alerts()

    def _setup_default_health_checks(self):
        """Setup default health checks."""
        self.register_health_check(HealthCheck(
            name="memory_usage",
            check_function=self._check_memory_usage,
            description="Check system memory usage",
            critical=True
        ))

        self.register_health_check(HealthCheck(
            name="disk_space",
            check_function=self._check_disk_space,
            description="Check available disk space",
            critical=False
        ))

        self.register_health_check(HealthCheck(
            name="error_rate",
            check_function=self._check_error_rate,
            description="Check error rate",
            critical=True
        ))

    def _setup_default_alerts(self):
        """Setup default alerts."""
        self.register_alert(Alert(
            name="high_memory_usage",
            condition=lambda metrics: metrics.get("memory_percent", 0) > 90,
            message="High memory usage detected: {memory_percent}%",
            severity="critical"
        ))

        self.register_alert(Alert(
            name="high_error_rate",
            condition=lambda metrics: metrics.get("error_rate", 0) > 0.1,
            message="High error rate detected: {error_rate}",
            severity="warning"
        ))

    def register_health_check(self, health_check: HealthCheck):
        """Register a new health check."""
        self.health_checks[health_check.name] = health_check
        self.logger.info(f"Registered health check: {health_check.name}")

    def register_alert(self, alert: Alert):
        """Register a new alert."""
        self.alerts[alert.name] = alert
        self.logger.info(f"Registered alert: {alert.name}")

    def add_alert_callback(self, callback: Callable[[Alert, Dict[str, Any]], None]):
        """Add callback for alert notifications."""
        self.alert_callbacks.append(callback)

    async def run_health_checks(self) -> Dict[str, bool]:
        """Run all health checks and return results."""
        results = {}

        for name, check in self.health_checks.items():
            try:
                # Check if we need to run this check (based on interval)
                last_check = self.health_last_check.get(name)
                if last_check and datetime.now() - last_check < timedelta(seconds=check.interval):
                    results[name] = self.health_results.get(name, False)
                    continue

                # Run the health check with timeout
                start_time = time.time()
                result = await asyncio.wait_for(
                    asyncio.to_thread(check.check_function),
                    timeout=check.timeout
                )

                duration = time.time() - start_time
                self.record_metric(f"health_check_{name}_duration", duration, MetricType.TIMER)

                results[name] = result
                self.health_results[name] = result
                self.health_last_check[name] = datetime.now()

                self.logger.debug(f"Health check {name}: {'PASS' if result else 'FAIL'} ({duration:.3f}s)")

            except asyncio.TimeoutError:
                self.logger.warning(f"Health check {name} timed out")
                results[name] = False
                self.health_results[name] = False
            except Exception as e:
                self.logger.error(f"Health check {name} failed with error: {e}")
                results[name] = False
                self.health_results[name] = False

        return results

    def get_health_status(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.health_results:
            return HealthStatus.UNKNOWN

        # Check critical health checks
        critical_checks = [
            name for name, check in self.health_checks.items()
            if check.critical
        ]

        failed_critical = [
            name for name in critical_checks
            if not self.health_results.get(name, False)
        ]

        if failed_critical:
            return HealthStatus.CRITICAL

        # Check non-critical health checks
        failed_checks = [
            name for name, result in self.health_results.items()
            if not result
        ]

        if failed_checks:
            if len(failed_checks) > len(self.health_results) * 0.5:
                return HealthStatus.DEGRADED
            else:
                return HealthStatus.WARNING

        return HealthStatus.HEALTHY

    def record_metric(self, name: str, value: float, metric_type: MetricType,
                     labels: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.now(),
            labels=labels or {}
        )

        self.metrics[name].append(metric)
        self._update_aggregates(name, value)

        # Check alerts
        self._check_alerts({name: value})

    def record_operation_time(self, operation: str, duration: float):
        """Record operation execution time."""
        self.operation_times[operation].append(duration)
        self.record_metric(f"operation_{operation}_duration", duration, MetricType.TIMER)

    def _update_aggregates(self, name: str, value: float):
        """Update metric aggregates."""
        metrics = [m.value for m in self.metrics[name]]

        if metrics:
            self.metric_aggregates[name] = {
                "count": len(metrics),
                "sum": sum(metrics),
                "avg": sum(metrics) / len(metrics),
                "min": min(metrics),
                "max": max(metrics),
                "latest": metrics[-1]
            }

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        # System metrics
        system_metrics = self._get_system_metrics()

        # Operation metrics
        operation_metrics = {}
        for operation, times in self.operation_times.items():
            if times:
                operation_metrics[operation] = {
                    "count": len(times),
                    "avg_duration": sum(times) / len(times),
                    "min_duration": min(times),
                    "max_duration": max(times),
                    "latest_duration": times[-1]
                }

        # Error metrics
        error_summary = self.error_handler.get_error_summary()

        return {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "health_status": self.get_health_status().value,
            "system": system_metrics,
            "operations": operation_metrics,
            "errors": error_summary,
            "custom_metrics": dict(self.metric_aggregates)
        }

    def _get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_mb = memory.available / (1024 * 1024)

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024 * 1024 * 1024)

            # Process metrics
            process = psutil.Process()
            process_memory_mb = process.memory_info().rss / (1024 * 1024)
            process_cpu_percent = process.cpu_percent()

            metrics = {
                "memory_percent": memory_percent,
                "memory_available_mb": memory_available_mb,
                "cpu_percent": cpu_percent,
                "disk_percent": disk_percent,
                "disk_free_gb": disk_free_gb,
                "process_memory_mb": process_memory_mb,
                "process_cpu_percent": process_cpu_percent
            }

            # Store for health checks
            self.resource_usage_history.append(metrics)

            return metrics

        except Exception as e:
            self.logger.warning(f"Failed to get system metrics: {e}")
            return {}

    def _check_memory_usage(self) -> bool:
        """Health check for memory usage."""
        try:
            memory = psutil.virtual_memory()
            return memory.percent < 90  # Less than 90% memory usage
        except Exception:
            return False

    def _check_disk_space(self) -> bool:
        """Health check for disk space."""
        try:
            disk = psutil.disk_usage('/')
            used_percent = (disk.used / disk.total) * 100
            return used_percent < 95  # Less than 95% disk usage
        except Exception:
            return False

    def _check_error_rate(self) -> bool:
        """Health check for error rate."""
        error_summary = self.error_handler.get_error_summary()

        # If no errors recorded, assume healthy
        if error_summary["error_count"] == 0:
            return True

        # Calculate error rate (errors per minute)
        uptime_minutes = (datetime.now() - self.start_time).total_seconds() / 60
        if uptime_minutes < 1:
            uptime_minutes = 1

        error_rate = error_summary["error_count"] / uptime_minutes
        return error_rate < 1  # Less than 1 error per minute

    def _check_alerts(self, current_metrics: Dict[str, float]):
        """Check alert conditions and trigger if necessary."""
        for alert_name, alert in self.alerts.items():
            try:
                # Check cooldown
                if (alert.last_triggered and
                    datetime.now() - alert.last_triggered < timedelta(seconds=alert.cooldown)):
                    continue

                # Check condition
                if alert.condition(current_metrics):
                    # Trigger alert
                    alert.last_triggered = datetime.now()
                    self._trigger_alert(alert, current_metrics)

            except Exception as e:
                self.logger.error(f"Error checking alert {alert_name}: {e}")

    def _trigger_alert(self, alert: Alert, metrics: Dict[str, Any]):
        """Trigger an alert."""
        formatted_message = alert.message.format(**metrics)

        self.logger.log(
            logging.CRITICAL if alert.severity == "critical" else logging.WARNING,
            f"ALERT [{alert.severity.upper()}] {alert.name}: {formatted_message}"
        )

        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert, metrics)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")

    def start_monitoring(self):
        """Start continuous monitoring in background thread."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Enhanced monitoring started")

    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("Enhanced monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                system_metrics = self._get_system_metrics()
                for name, value in system_metrics.items():
                    self.record_metric(name, value, MetricType.GAUGE)

                # Run health checks (async)
                asyncio.run(self.run_health_checks())

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")

            time.sleep(30)  # Run every 30 seconds


# Global monitor instance
_global_monitor = EnhancedMonitor()


def get_monitor() -> EnhancedMonitor:
    """Get global monitor instance."""
    return _global_monitor


def monitor_operation(operation_name: str):
    """Decorator for monitoring operation performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                get_monitor().record_operation_time(operation_name, duration)
                return result
            except Exception:
                duration = time.time() - start_time
                get_monitor().record_operation_time(f"{operation_name}_error", duration)
                raise
        return wrapper
    return decorator
