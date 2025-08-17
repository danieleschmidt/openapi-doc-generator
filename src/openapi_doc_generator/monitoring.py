"""Monitoring and observability features for OpenAPI Doc Generator."""

import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import psutil


@dataclass
class HealthCheck:
    """Health check result."""
    service: str
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: float
    response_time_ms: float
    details: Optional[Dict[str, Any]] = None


@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_total_mb: float
    disk_usage_percent: float
    process_count: int
    timestamp: float


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    operation: str
    duration_ms: float
    memory_usage_mb: float
    cpu_time_ms: float
    timestamp: float
    correlation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MetricsCollector:
    """Collects and manages application metrics."""

    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.health_checks: List[HealthCheck] = []
        self.logger = logging.getLogger(__name__)

    def record_performance(self,
                          operation: str,
                          duration_ms: float,
                          memory_usage_mb: float = 0,
                          cpu_time_ms: float = 0,
                          correlation_id: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record performance metrics for an operation."""
        metric = PerformanceMetrics(
            operation=operation,
            duration_ms=duration_ms,
            memory_usage_mb=memory_usage_mb,
            cpu_time_ms=cpu_time_ms,
            timestamp=time.time(),
            correlation_id=correlation_id,
            metadata=metadata or {}
        )
        self.metrics.append(metric)

        # Log performance metric
        self.logger.info(
            f"Performance: {operation} completed in {duration_ms:.2f}ms",
            extra={
                "operation": operation,
                "duration_ms": duration_ms,
                "memory_mb": memory_usage_mb,
                "correlation_id": correlation_id
            }
        )

    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        disk_usage = psutil.disk_usage('/')

        return SystemMetrics(
            cpu_percent=process.cpu_percent(),
            memory_percent=system_memory.percent,
            memory_used_mb=memory_info.rss / 1024 / 1024,
            memory_total_mb=system_memory.total / 1024 / 1024,
            disk_usage_percent=disk_usage.percent,
            process_count=len(psutil.pids()),
            timestamp=time.time()
        )

    def check_health(self) -> Dict[str, HealthCheck]:
        """Perform comprehensive health checks."""
        checks = {}

        # System health
        start_time = time.time()
        try:
            metrics = self.get_system_metrics()

            status = "healthy"
            if metrics.memory_percent > 90:
                status = "degraded"
            if metrics.memory_percent > 95 or metrics.disk_usage_percent > 95:
                status = "unhealthy"

            checks["system"] = HealthCheck(
                service="system",
                status=status,
                timestamp=time.time(),
                response_time_ms=(time.time() - start_time) * 1000,
                details=asdict(metrics)
            )
        except Exception as e:
            checks["system"] = HealthCheck(
                service="system",
                status="unhealthy",
                timestamp=time.time(),
                response_time_ms=(time.time() - start_time) * 1000,
                details={"error": str(e)}
            )

        # Application health
        start_time = time.time()
        try:
            from openapi_doc_generator.documentator import APIDocumentator

            # Quick functional test
            APIDocumentator()

            status = "healthy"
            checks["application"] = HealthCheck(
                service="application",
                status=status,
                timestamp=time.time(),
                response_time_ms=(time.time() - start_time) * 1000,
                details={"version": "1.2.0"}
            )
        except Exception as e:
            checks["application"] = HealthCheck(
                service="application",
                status="unhealthy",
                timestamp=time.time(),
                response_time_ms=(time.time() - start_time) * 1000,
                details={"error": str(e)}
            )

        # Dependencies health
        start_time = time.time()
        try:
            import graphql
            import jinja2

            checks["dependencies"] = HealthCheck(
                service="dependencies",
                status="healthy",
                timestamp=time.time(),
                response_time_ms=(time.time() - start_time) * 1000,
                details={
                    "jinja2_version": jinja2.__version__,
                    "graphql_version": graphql.__version__
                }
            )
        except Exception as e:
            checks["dependencies"] = HealthCheck(
                service="dependencies",
                status="unhealthy",
                timestamp=time.time(),
                response_time_ms=(time.time() - start_time) * 1000,
                details={"error": str(e)}
            )

        return checks

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        if not self.metrics:
            return {"total_operations": 0}

        durations = [m.duration_ms for m in self.metrics]
        memory_usages = [m.memory_usage_mb for m in self.metrics if m.memory_usage_mb > 0]

        operations_count = {}
        for metric in self.metrics:
            operations_count[metric.operation] = operations_count.get(metric.operation, 0) + 1

        summary = {
            "total_operations": len(self.metrics),
            "operations_by_type": operations_count,
            "performance": {
                "avg_duration_ms": sum(durations) / len(durations),
                "min_duration_ms": min(durations),
                "max_duration_ms": max(durations),
                "p95_duration_ms": sorted(durations)[int(0.95 * len(durations))] if durations else 0,
                "p99_duration_ms": sorted(durations)[int(0.99 * len(durations))] if durations else 0
            }
        }

        if memory_usages:
            summary["memory"] = {
                "avg_usage_mb": sum(memory_usages) / len(memory_usages),
                "peak_usage_mb": max(memory_usages)
            }

        return summary

    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        if format == "json":
            return json.dumps({
                "metrics": [asdict(m) for m in self.metrics],
                "summary": self.get_metrics_summary(),
                "timestamp": time.time()
            }, indent=2)
        elif format == "prometheus":
            return self._export_prometheus_format()
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        # Performance metrics
        lines.append("# HELP openapi_doc_generator_operation_duration_seconds Duration of operations")
        lines.append("# TYPE openapi_doc_generator_operation_duration_seconds histogram")

        for metric in self.metrics:
            duration_seconds = metric.duration_ms / 1000
            lines.append(
                f'openapi_doc_generator_operation_duration_seconds{{operation="{metric.operation}"}} {duration_seconds}'
            )

        # System metrics
        system_metrics = self.get_system_metrics()
        lines.extend([
            "# HELP openapi_doc_generator_memory_usage_percent Memory usage percentage",
            "# TYPE openapi_doc_generator_memory_usage_percent gauge",
            f"openapi_doc_generator_memory_usage_percent {system_metrics.memory_percent}",
            "",
            "# HELP openapi_doc_generator_cpu_usage_percent CPU usage percentage",
            "# TYPE openapi_doc_generator_cpu_usage_percent gauge",
            f"openapi_doc_generator_cpu_usage_percent {system_metrics.cpu_percent}"
        ])

        return "\n".join(lines)

    def clear_metrics(self) -> None:
        """Clear collected metrics."""
        self.metrics.clear()
        self.health_checks.clear()


# Global metrics collector instance
metrics_collector = MetricsCollector()


def performance_timer(operation: str, correlation_id: Optional[str] = None):
    """Decorator for timing operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024

            try:
                result = func(*args, **kwargs)

                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024

                duration_ms = (end_time - start_time) * 1000
                memory_delta = end_memory - start_memory

                metrics_collector.record_performance(
                    operation=operation,
                    duration_ms=duration_ms,
                    memory_usage_mb=memory_delta,
                    correlation_id=correlation_id,
                    metadata={"success": True}
                )

                return result

            except Exception as e:
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000

                metrics_collector.record_performance(
                    operation=operation,
                    duration_ms=duration_ms,
                    correlation_id=correlation_id,
                    metadata={"success": False, "error": str(e)}
                )

                raise

        return wrapper
    return decorator


def get_health_status() -> Dict[str, Any]:
    """Get overall health status."""
    checks = metrics_collector.check_health()

    overall_status = "healthy"
    if any(check.status == "unhealthy" for check in checks.values()):
        overall_status = "unhealthy"
    elif any(check.status == "degraded" for check in checks.values()):
        overall_status = "degraded"

    return {
        "status": overall_status,
        "timestamp": time.time(),
        "checks": {name: asdict(check) for name, check in checks.items()},
        "metrics_summary": metrics_collector.get_metrics_summary()
    }


def get_readiness_status() -> Dict[str, Any]:
    """Get readiness status for deployment."""
    checks = metrics_collector.check_health()

    # Readiness is more strict - all checks must be healthy
    is_ready = all(check.status == "healthy" for check in checks.values())

    return {
        "ready": is_ready,
        "timestamp": time.time(),
        "checks": {name: check.status for name, check in checks.items()}
    }


def get_metrics() -> str:
    """Get metrics in Prometheus format."""
    return metrics_collector.export_metrics("prometheus")
