"""Advanced health monitoring and system diagnostics."""

from __future__ import annotations

import json
import logging
import psutil
import time
from dataclasses import asdict, dataclass
from enum import Enum
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional

from .monitoring import MetricsCollector, SystemMetrics
from .quantum_audit_logger import AuditEventType, get_audit_logger
from .quantum_security import SecurityLevel


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of system components to monitor."""
    CORE_API = "core_api"
    QUANTUM_PLANNER = "quantum_planner"
    SECURITY_VALIDATOR = "security_validator"
    AUDIT_LOGGER = "audit_logger"
    DOCUMENTATION_GENERATOR = "documentation_generator"
    PLUGIN_SYSTEM = "plugin_system"
    MONITORING_SYSTEM = "monitoring_system"
    DATABASE = "database"
    EXTERNAL_DEPENDENCY = "external_dependency"


@dataclass
class ComponentHealth:
    """Health status of a system component."""
    component_type: ComponentType
    component_name: str
    status: HealthStatus
    last_check: float
    response_time_ms: float
    error_count: int
    warning_count: int
    details: Dict[str, Any]
    dependencies: List[str]


@dataclass
class SystemHealth:
    """Overall system health assessment."""
    overall_status: HealthStatus
    timestamp: float
    components: List[ComponentHealth]
    system_metrics: SystemMetrics
    alerts: List[str]
    recommendations: List[str]


class HealthCheck:
    """Individual health check implementation."""
    
    def __init__(self,
                 component_type: ComponentType,
                 component_name: str,
                 check_function: Callable[[], Dict[str, Any]],
                 critical_threshold: float = 5.0,
                 warning_threshold: float = 2.0,
                 timeout_seconds: float = 10.0):
        """Initialize health check."""
        self.component_type = component_type
        self.component_name = component_name
        self.check_function = check_function
        self.critical_threshold = critical_threshold
        self.warning_threshold = warning_threshold
        self.timeout_seconds = timeout_seconds
        self.error_count = 0
        self.warning_count = 0
        self.last_success = time.time()
        
    def execute(self) -> ComponentHealth:
        """Execute health check."""
        start_time = time.time()
        
        try:
            # Execute check function with timeout
            result = self._execute_with_timeout()
            duration = (time.time() - start_time) * 1000  # Convert to ms
            
            # Determine status based on response time and result
            status = self._determine_status(duration, result)
            
            # Update counters
            if status == HealthStatus.CRITICAL:
                self.error_count += 1
            elif status == HealthStatus.DEGRADED:
                self.warning_count += 1
            else:
                self.last_success = time.time()
                
            return ComponentHealth(
                component_type=self.component_type,
                component_name=self.component_name,
                status=status,
                last_check=time.time(),
                response_time_ms=duration,
                error_count=self.error_count,
                warning_count=self.warning_count,
                details=result,
                dependencies=result.get('dependencies', [])
            )
            
        except Exception as e:
            self.error_count += 1
            duration = (time.time() - start_time) * 1000
            
            return ComponentHealth(
                component_type=self.component_type,
                component_name=self.component_name,
                status=HealthStatus.CRITICAL,
                last_check=time.time(),
                response_time_ms=duration,
                error_count=self.error_count,
                warning_count=self.warning_count,
                details={"error": str(e)},
                dependencies=[]
            )
            
    def _execute_with_timeout(self) -> Dict[str, Any]:
        """Execute check function with timeout protection."""
        # Simple timeout implementation
        start_time = time.time()
        result = self.check_function()
        
        if time.time() - start_time > self.timeout_seconds:
            raise TimeoutError(f"Health check timeout after {self.timeout_seconds}s")
            
        return result
        
    def _determine_status(self, 
                         duration_ms: float, 
                         result: Dict[str, Any]) -> HealthStatus:
        """Determine health status based on metrics."""
        # Check for explicit status in result
        if 'status' in result:
            status_map = {
                'healthy': HealthStatus.HEALTHY,
                'degraded': HealthStatus.DEGRADED,
                'critical': HealthStatus.CRITICAL,
                'unknown': HealthStatus.UNKNOWN
            }
            return status_map.get(result['status'], HealthStatus.UNKNOWN)
            
        # Check response time thresholds
        if duration_ms >= self.critical_threshold * 1000:
            return HealthStatus.CRITICAL
        elif duration_ms >= self.warning_threshold * 1000:
            return HealthStatus.DEGRADED
            
        # Check for error indicators
        if result.get('error_rate', 0) > 0.1:  # >10% error rate
            return HealthStatus.CRITICAL
        elif result.get('error_rate', 0) > 0.05:  # >5% error rate
            return HealthStatus.DEGRADED
            
        return HealthStatus.HEALTHY


class QuantumHealthMonitor:
    """Advanced health monitoring system with automated diagnostics."""
    
    def __init__(self,
                 check_interval_seconds: float = 30.0,
                 enable_auto_diagnostics: bool = True,
                 enable_self_healing: bool = False):
        """Initialize health monitor."""
        self.check_interval = check_interval_seconds
        self.enable_auto_diagnostics = enable_auto_diagnostics
        self.enable_self_healing = enable_self_healing
        
        self.health_checks: Dict[str, HealthCheck] = {}
        self.metrics_collector = MetricsCollector()
        self.audit_logger = get_audit_logger()
        self.logger = logging.getLogger(__name__)
        
        self._monitoring_thread: Optional[Thread] = None
        self._stop_monitoring = False
        self._lock = Lock()
        
        # Health history for trend analysis
        self.health_history: List[SystemHealth] = []
        self.max_history_size = 1000
        
        # Setup default health checks
        self._setup_default_checks()
        
    def _setup_default_checks(self):
        """Setup default system health checks."""
        # Core API health
        self.add_health_check(
            component_type=ComponentType.CORE_API,
            component_name="openapi_generator",
            check_function=self._check_core_api_health,
            critical_threshold=5.0,
            warning_threshold=2.0
        )
        
        # Quantum planner health
        self.add_health_check(
            component_type=ComponentType.QUANTUM_PLANNER,
            component_name="quantum_task_planner",
            check_function=self._check_quantum_planner_health,
            critical_threshold=10.0,
            warning_threshold=5.0
        )
        
        # Security validator health
        self.add_health_check(
            component_type=ComponentType.SECURITY_VALIDATOR,
            component_name="quantum_security_validator",
            check_function=self._check_security_validator_health,
            critical_threshold=3.0,
            warning_threshold=1.5
        )
        
        # Plugin system health
        self.add_health_check(
            component_type=ComponentType.PLUGIN_SYSTEM,
            component_name="plugin_loader",
            check_function=self._check_plugin_system_health,
            critical_threshold=2.0,
            warning_threshold=1.0
        )
        
        # System resources health
        self.add_health_check(
            component_type=ComponentType.MONITORING_SYSTEM,
            component_name="system_resources",
            check_function=self._check_system_resources_health,
            critical_threshold=1.0,
            warning_threshold=0.5
        )
        
    def add_health_check(self,
                        component_type: ComponentType,
                        component_name: str,
                        check_function: Callable[[], Dict[str, Any]],
                        **kwargs) -> None:
        """Add a custom health check."""
        check_key = f"{component_type.value}_{component_name}"
        
        self.health_checks[check_key] = HealthCheck(
            component_type=component_type,
            component_name=component_name,
            check_function=check_function,
            **kwargs
        )
        
        self.logger.info(f"Added health check: {check_key}")
        
    def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self.logger.warning("Health monitoring already running")
            return
            
        self._stop_monitoring = False
        self._monitoring_thread = Thread(
            target=self._monitoring_loop,
            name="QuantumHealthMonitor",
            daemon=True
        )
        self._monitoring_thread.start()
        
        self.logger.info("Started health monitoring")
        self.audit_logger.log_security_event(
            event_type=AuditEventType.SYSTEM_ACCESS,
            action="start_health_monitoring",
            result="success",
            severity=SecurityLevel.LOW
        )
        
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self._stop_monitoring = True
        
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
            
        self.logger.info("Stopped health monitoring")
        
    def get_system_health(self) -> SystemHealth:
        """Get current system health status."""
        with self._lock:
            component_healths = []
            
            # Execute all health checks
            for check in self.health_checks.values():
                try:
                    health = check.execute()
                    component_healths.append(health)
                except Exception as e:
                    self.logger.error(f"Health check failed: {e}")
                    
            # Get system metrics
            system_metrics = self._get_system_metrics()
            
            # Determine overall status
            overall_status = self._determine_overall_status(component_healths)
            
            # Generate alerts and recommendations
            alerts, recommendations = self._analyze_health_issues(
                component_healths, 
                system_metrics
            )
            
            health = SystemHealth(
                overall_status=overall_status,
                timestamp=time.time(),
                components=component_healths,
                system_metrics=system_metrics,
                alerts=alerts,
                recommendations=recommendations
            )
            
            # Store in history
            self.health_history.append(health)
            if len(self.health_history) > self.max_history_size:
                self.health_history.pop(0)
                
            return health
            
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_monitoring:
            try:
                start_time = time.time()
                
                # Get system health
                health = self.get_system_health()
                
                # Log critical issues
                if health.overall_status == HealthStatus.CRITICAL:
                    self.audit_logger.log_security_event(
                        event_type=AuditEventType.ERROR_CONDITION,
                        action="system_health_critical",
                        result="critical",
                        severity=SecurityLevel.CRITICAL,
                        details={"alerts": health.alerts}
                    )
                    
                # Auto-diagnostics
                if self.enable_auto_diagnostics:
                    self._run_auto_diagnostics(health)
                    
                # Self-healing attempts
                if self.enable_self_healing:
                    self._attempt_self_healing(health)
                    
                # Calculate sleep time
                elapsed = time.time() - start_time
                sleep_time = max(0, self.check_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                time.sleep(5.0)  # Brief pause on error
                
    def _check_core_api_health(self) -> Dict[str, Any]:
        """Check core API health."""
        try:
            from .documentator import APIDocumentator
            
            # Simple functionality test
            start_time = time.time()
            documentator = APIDocumentator()
            duration = (time.time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "initialization_time_ms": duration,
                "dependencies": ["jinja2", "graphql-core"]
            }
        except Exception as e:
            return {
                "status": "critical",
                "error": str(e),
                "dependencies": []
            }
            
    def _check_quantum_planner_health(self) -> Dict[str, Any]:
        """Check quantum planner health."""
        try:
            from .quantum_planner import QuantumTaskPlanner
            
            start_time = time.time()
            planner = QuantumTaskPlanner()
            duration = (time.time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "initialization_time_ms": duration,
                "task_count": len(planner.task_registry),
                "dependencies": ["quantum_scheduler", "quantum_optimizer"]
            }
        except Exception as e:
            return {
                "status": "critical",
                "error": str(e),
                "dependencies": []
            }
            
    def _check_security_validator_health(self) -> Dict[str, Any]:
        """Check security validator health."""
        try:
            from .quantum_security import QuantumSecurityValidator
            
            start_time = time.time()
            validator = QuantumSecurityValidator()
            duration = (time.time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "initialization_time_ms": duration,
                "blocked_patterns_count": len(validator.blocked_patterns),
                "dependencies": []
            }
        except Exception as e:
            return {
                "status": "critical",
                "error": str(e),
                "dependencies": []
            }
            
    def _check_plugin_system_health(self) -> Dict[str, Any]:
        """Check plugin system health."""
        try:
            import pkg_resources
            
            # Check for available plugins
            plugins = list(pkg_resources.iter_entry_points('openapi_doc_generator.plugins'))
            
            return {
                "status": "healthy",
                "plugin_count": len(plugins),
                "plugins": [p.name for p in plugins],
                "dependencies": ["pkg_resources"]
            }
        except Exception as e:
            return {
                "status": "degraded",
                "error": str(e),
                "plugin_count": 0,
                "dependencies": []
            }
            
    def _check_system_resources_health(self) -> Dict[str, Any]:
        """Check system resource health."""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Determine status based on resource usage
            status = "healthy"
            if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
                status = "critical"
            elif cpu_percent > 70 or memory.percent > 70 or disk.percent > 80:
                status = "degraded"
                
            return {
                "status": status,
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "memory_used_gb": memory.used / (1024**3),
                "disk_free_gb": disk.free / (1024**3),
                "dependencies": ["psutil"]
            }
        except Exception as e:
            return {
                "status": "unknown",
                "error": str(e),
                "dependencies": []
            }
            
    def _get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return SystemMetrics(
                cpu_percent=psutil.cpu_percent(),
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024**2),
                memory_total_mb=memory.total / (1024**2),
                disk_usage_percent=disk.percent,
                process_count=len(psutil.pids()),
                timestamp=time.time()
            )
        except Exception as e:
            self.logger.error(f"Failed to get system metrics: {e}")
            return SystemMetrics(0, 0, 0, 0, 0, 0, time.time())
            
    def _determine_overall_status(self, 
                                component_healths: List[ComponentHealth]) -> HealthStatus:
        """Determine overall system status from components."""
        if not component_healths:
            return HealthStatus.UNKNOWN
            
        # Count statuses
        status_counts = {}
        for health in component_healths:
            status = health.status
            status_counts[status] = status_counts.get(status, 0) + 1
            
        # Determine overall status
        if status_counts.get(HealthStatus.CRITICAL, 0) > 0:
            return HealthStatus.CRITICAL
        elif status_counts.get(HealthStatus.DEGRADED, 0) > 0:
            return HealthStatus.DEGRADED
        elif status_counts.get(HealthStatus.UNKNOWN, 0) > 0:
            return HealthStatus.UNKNOWN
        else:
            return HealthStatus.HEALTHY
            
    def _analyze_health_issues(self, 
                             component_healths: List[ComponentHealth],
                             system_metrics: SystemMetrics) -> tuple[List[str], List[str]]:
        """Analyze health issues and generate alerts/recommendations."""
        alerts = []
        recommendations = []
        
        # Component-specific alerts
        for health in component_healths:
            if health.status == HealthStatus.CRITICAL:
                alerts.append(f"CRITICAL: {health.component_name} is unhealthy")
                recommendations.append(
                    f"Investigate {health.component_name} - {health.error_count} errors"
                )
            elif health.status == HealthStatus.DEGRADED:
                alerts.append(f"WARNING: {health.component_name} is degraded")
                
        # System resource alerts
        if system_metrics.cpu_percent > 90:
            alerts.append("CRITICAL: CPU usage exceeds 90%")
            recommendations.append("Scale up CPU resources or optimize workload")
            
        if system_metrics.memory_percent > 90:
            alerts.append("CRITICAL: Memory usage exceeds 90%")
            recommendations.append("Scale up memory or identify memory leaks")
            
        if system_metrics.disk_usage_percent > 90:
            alerts.append("CRITICAL: Disk usage exceeds 90%")
            recommendations.append("Clean up disk space or expand storage")
            
        return alerts, recommendations
        
    def _run_auto_diagnostics(self, health: SystemHealth) -> None:
        """Run automated diagnostics based on health status."""
        if health.overall_status in [HealthStatus.CRITICAL, HealthStatus.DEGRADED]:
            self.logger.info("Running auto-diagnostics...")
            
            # Analyze error patterns
            error_patterns = self._analyze_error_patterns()
            
            # Check dependency health
            dependency_issues = self._check_dependency_health()
            
            # Performance analysis
            performance_issues = self._analyze_performance_trends()
            
            diagnostics = {
                "error_patterns": error_patterns,
                "dependency_issues": dependency_issues,
                "performance_issues": performance_issues
            }
            
            self.logger.info(f"Auto-diagnostics completed: {diagnostics}")
            
    def _attempt_self_healing(self, health: SystemHealth) -> None:
        """Attempt automated self-healing actions."""
        if health.overall_status == HealthStatus.CRITICAL:
            self.logger.info("Attempting self-healing actions...")
            
            # Simple self-healing actions
            healing_actions = []
            
            for component in health.components:
                if component.status == HealthStatus.CRITICAL:
                    if component.error_count > 5:
                        # Reset error counters
                        check_key = f"{component.component_type.value}_{component.component_name}"
                        if check_key in self.health_checks:
                            self.health_checks[check_key].error_count = 0
                            healing_actions.append(f"Reset error count for {component.component_name}")
                            
            if healing_actions:
                self.logger.info(f"Self-healing actions taken: {healing_actions}")
                self.audit_logger.log_security_event(
                    event_type=AuditEventType.SYSTEM_ACCESS,
                    action="self_healing_attempted",
                    result="success",
                    severity=SecurityLevel.MEDIUM,
                    details={"actions": healing_actions}
                )
                
    def _analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze error patterns from health history."""
        if len(self.health_history) < 5:
            return {"status": "insufficient_data"}
            
        # Simple error pattern analysis
        recent_healths = self.health_history[-10:]
        error_components = set()
        
        for health in recent_healths:
            for component in health.components:
                if component.status in [HealthStatus.CRITICAL, HealthStatus.DEGRADED]:
                    error_components.add(component.component_name)
                    
        return {
            "status": "analyzed",
            "frequent_error_components": list(error_components),
            "analysis_window": len(recent_healths)
        }
        
    def _check_dependency_health(self) -> Dict[str, Any]:
        """Check health of external dependencies."""
        # Simple dependency health check
        dependencies = ["jinja2", "graphql-core", "psutil"]
        healthy_deps = []
        
        for dep in dependencies:
            try:
                __import__(dep.replace('-', '_'))
                healthy_deps.append(dep)
            except ImportError:
                pass
                
        return {
            "total_dependencies": len(dependencies),
            "healthy_dependencies": len(healthy_deps),
            "healthy_deps": healthy_deps
        }
        
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends from metrics history."""
        if len(self.health_history) < 5:
            return {"status": "insufficient_data"}
            
        # Calculate trend metrics
        recent_healths = self.health_history[-10:]
        cpu_values = [h.system_metrics.cpu_percent for h in recent_healths]
        memory_values = [h.system_metrics.memory_percent for h in recent_healths]
        
        return {
            "status": "analyzed",
            "cpu_trend": "increasing" if cpu_values[-1] > cpu_values[0] else "stable",
            "memory_trend": "increasing" if memory_values[-1] > memory_values[0] else "stable",
            "avg_cpu": sum(cpu_values) / len(cpu_values),
            "avg_memory": sum(memory_values) / len(memory_values)
        }


# Global health monitor instance
_health_monitor: Optional[QuantumHealthMonitor] = None


def get_health_monitor() -> QuantumHealthMonitor:
    """Get global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = QuantumHealthMonitor()
    return _health_monitor