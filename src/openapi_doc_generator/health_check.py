"""
Comprehensive Health Check System

Provides deep health monitoring of all system components, dependencies,
and resources with configurable thresholds and automatic recovery.
"""

import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import psutil

from .circuit_breaker import get_all_circuit_breaker_metrics
from .enhanced_monitoring import get_monitor


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    component: str
    status: HealthStatus
    message: str
    metrics: Dict[str, Any]
    timestamp: float
    response_time_ms: float


@dataclass
class ResourceThresholds:
    """Configurable thresholds for resource monitoring."""
    cpu_warning: float = 70.0      # CPU usage warning threshold (%)
    cpu_critical: float = 90.0     # CPU usage critical threshold (%)
    memory_warning: float = 75.0   # Memory usage warning threshold (%)
    memory_critical: float = 90.0  # Memory usage critical threshold (%)
    disk_warning: float = 80.0     # Disk usage warning threshold (%)
    disk_critical: float = 95.0    # Disk usage critical threshold (%)


class HealthChecker:
    """Comprehensive health monitoring system."""

    def __init__(self, thresholds: ResourceThresholds = None):
        self.thresholds = thresholds or ResourceThresholds()
        self.monitor = get_monitor()
        self._custom_checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self._lock = threading.RLock()

    def register_custom_check(self, name: str, check_func: Callable[[], HealthCheckResult]) -> None:
        """Register a custom health check function."""
        with self._lock:
            self._custom_checks[name] = check_func

    def check_system_resources(self) -> HealthCheckResult:
        """Check system resource utilization."""
        start_time = time.time()

        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk usage for current working directory
            disk = psutil.disk_usage('.')
            disk_percent = (disk.used / disk.total) * 100

            # Determine overall status
            status = HealthStatus.HEALTHY
            issues = []

            if cpu_percent >= self.thresholds.cpu_critical:
                status = HealthStatus.CRITICAL
                issues.append(f"CPU usage critical: {cpu_percent:.1f}%")
            elif cpu_percent >= self.thresholds.cpu_warning:
                status = HealthStatus.DEGRADED
                issues.append(f"CPU usage high: {cpu_percent:.1f}%")

            if memory_percent >= self.thresholds.memory_critical:
                status = HealthStatus.CRITICAL
                issues.append(f"Memory usage critical: {memory_percent:.1f}%")
            elif memory_percent >= self.thresholds.memory_warning:
                status = HealthStatus.DEGRADED
                issues.append(f"Memory usage high: {memory_percent:.1f}%")

            if disk_percent >= self.thresholds.disk_critical:
                status = HealthStatus.CRITICAL
                issues.append(f"Disk usage critical: {disk_percent:.1f}%")
            elif disk_percent >= self.thresholds.disk_warning:
                status = HealthStatus.DEGRADED
                issues.append(f"Disk usage high: {disk_percent:.1f}%")

            message = "System resources normal" if not issues else "; ".join(issues)

            metrics = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk_percent,
                "disk_free_gb": disk.free / (1024**3)
            }

            response_time = (time.time() - start_time) * 1000

            return HealthCheckResult(
                component="system_resources",
                status=status,
                message=message,
                metrics=metrics,
                timestamp=time.time(),
                response_time_ms=response_time
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component="system_resources",
                status=HealthStatus.CRITICAL,
                message=f"Error checking system resources: {str(e)}",
                metrics={},
                timestamp=time.time(),
                response_time_ms=response_time
            )

    def check_application_components(self) -> HealthCheckResult:
        """Check application-specific components."""
        start_time = time.time()

        try:
            # Check if core modules can be imported
            issues = []
            components_status = {}

            core_modules = [
                'openapi_doc_generator.discovery',
                'openapi_doc_generator.documentator',
                'openapi_doc_generator.schema',
                'openapi_doc_generator.spec'
            ]

            for module in core_modules:
                try:
                    __import__(module)
                    components_status[module] = "healthy"
                except Exception as e:
                    components_status[module] = f"error: {str(e)}"
                    issues.append(f"Module {module} import failed")

            # Check circuit breaker status
            circuit_metrics = get_all_circuit_breaker_metrics()
            open_circuits = [name for name, metrics in circuit_metrics.items()
                           if metrics.get('state') == 'open']

            if open_circuits:
                issues.append(f"Circuit breakers open: {', '.join(open_circuits)}")

            # Determine overall status
            if len(issues) >= 3:
                status = HealthStatus.CRITICAL
            elif len(issues) >= 1:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY

            message = "Application components healthy" if not issues else "; ".join(issues)

            metrics = {
                "core_modules": components_status,
                "circuit_breakers": circuit_metrics,
                "issues_count": len(issues)
            }

            response_time = (time.time() - start_time) * 1000

            return HealthCheckResult(
                component="application_components",
                status=status,
                message=message,
                metrics=metrics,
                timestamp=time.time(),
                response_time_ms=response_time
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component="application_components",
                status=HealthStatus.CRITICAL,
                message=f"Error checking application components: {str(e)}",
                metrics={},
                timestamp=time.time(),
                response_time_ms=response_time
            )

    def check_dependencies(self) -> HealthCheckResult:
        """Check external dependencies and file system access."""
        start_time = time.time()

        try:
            issues = []
            dependencies_status = {}

            # Check required Python packages
            required_packages = ['jinja2', 'graphql-core', 'psutil']

            for package in required_packages:
                try:
                    __import__(package.replace('-', '_'))
                    dependencies_status[package] = "available"
                except ImportError:
                    dependencies_status[package] = "missing"
                    issues.append(f"Package {package} not available")

            # Check file system access
            try:
                test_path = Path('.') / 'health_check_test'
                test_path.write_text('test')
                test_path.unlink()
                dependencies_status['filesystem_write'] = "accessible"
            except Exception as e:
                dependencies_status['filesystem_write'] = f"error: {str(e)}"
                issues.append("File system write access failed")

            # Determine overall status
            if len(issues) >= 2:
                status = HealthStatus.CRITICAL
            elif len(issues) >= 1:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY

            message = "Dependencies healthy" if not issues else "; ".join(issues)

            metrics = {
                "dependencies": dependencies_status,
                "issues_count": len(issues)
            }

            response_time = (time.time() - start_time) * 1000

            return HealthCheckResult(
                component="dependencies",
                status=status,
                message=message,
                metrics=metrics,
                timestamp=time.time(),
                response_time_ms=response_time
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component="dependencies",
                status=HealthStatus.CRITICAL,
                message=f"Error checking dependencies: {str(e)}",
                metrics={},
                timestamp=time.time(),
                response_time_ms=response_time
            )

    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks and return comprehensive results."""
        results = {}

        # Run built-in checks
        results["system_resources"] = self.check_system_resources()
        results["application_components"] = self.check_application_components()
        results["dependencies"] = self.check_dependencies()

        # Run custom checks
        with self._lock:
            for name, check_func in self._custom_checks.items():
                try:
                    results[name] = check_func()
                except Exception as e:
                    results[name] = HealthCheckResult(
                        component=name,
                        status=HealthStatus.CRITICAL,
                        message=f"Custom check failed: {str(e)}",
                        metrics={},
                        timestamp=time.time(),
                        response_time_ms=0.0
                    )

        return results

    def get_overall_status(self, results: Dict[str, HealthCheckResult] = None) -> HealthStatus:
        """Get overall system health status."""
        if results is None:
            results = self.run_all_checks()

        # Determine worst status
        statuses = [result.status for result in results.values()]

        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary for monitoring."""
        results = self.run_all_checks()
        overall_status = self.get_overall_status(results)

        return {
            "overall_status": overall_status.value,
            "timestamp": time.time(),
            "checks": {name: {
                "status": result.status.value,
                "message": result.message,
                "response_time_ms": result.response_time_ms,
                "metrics": result.metrics
            } for name, result in results.items()}
        }


# Global health checker instance
_health_checker: Optional[HealthChecker] = None
_health_lock = threading.RLock()


def get_health_checker(thresholds: ResourceThresholds = None) -> HealthChecker:
    """Get global health checker instance."""
    global _health_checker
    with _health_lock:
        if _health_checker is None:
            _health_checker = HealthChecker(thresholds)
        return _health_checker


def quick_health_check() -> Dict[str, Any]:
    """Quick health check for API endpoints."""
    checker = get_health_checker()
    return checker.get_health_summary()
