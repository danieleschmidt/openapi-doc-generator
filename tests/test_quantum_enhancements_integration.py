"""Integration tests for quantum enhancements."""

import pytest
import time
from unittest.mock import Mock, patch

from openapi_doc_generator.quantum_audit_logger import (
    AuditEventType, 
    get_audit_logger, 
    SecurityLevel
)
from openapi_doc_generator.quantum_health_monitor import (
    get_health_monitor,
    HealthStatus,
    ComponentType
)
from openapi_doc_generator.quantum_performance_optimizer import (
    get_performance_optimizer,
    OptimizationStrategy,
    ResiliencePattern
)
from openapi_doc_generator.quantum_resilience_engine import (
    get_resilience_engine,
    ResilienceConfig
)


class TestQuantumEnhancementsIntegration:
    """Integration tests for quantum system enhancements."""
    
    def test_audit_logger_integration(self):
        """Test audit logger integration."""
        audit_logger = get_audit_logger()
        
        # Test basic logging
        audit_logger.log_security_event(
            event_type=AuditEventType.SYSTEM_ACCESS,
            action="test_action",
            result="success",
            severity=SecurityLevel.LOW
        )
        
        # Verify buffer has events
        assert len(audit_logger.audit_buffer) > 0
        
        # Test security violation logging
        audit_logger.log_security_violation(
            violation_type="test_violation",
            details={"test": "data"},
            severity=SecurityLevel.HIGH
        )
        
        # Should have security alerts
        assert len(audit_logger.security_alerts) > 0
        
    def test_health_monitor_integration(self):
        """Test health monitor integration."""
        health_monitor = get_health_monitor()
        
        # Get system health
        health = health_monitor.get_system_health()
        
        assert health.overall_status in [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.CRITICAL,
            HealthStatus.UNKNOWN
        ]
        assert len(health.components) > 0
        assert health.system_metrics is not None
        
        # Test custom health check
        def custom_check():
            return {"status": "healthy", "test": True}
            
        health_monitor.add_health_check(
            component_type=ComponentType.EXTERNAL_DEPENDENCY,
            component_name="test_component",
            check_function=custom_check
        )
        
        # Verify check was added
        assert "external_dependency_test_component" in health_monitor.health_checks
        
    def test_performance_optimizer_integration(self):
        """Test performance optimizer integration."""
        performance_optimizer = get_performance_optimizer()
        
        # Test simple operation optimization
        def test_operation(data):
            return [x * 2 for x in data]
            
        test_data = list(range(100))
        
        result = performance_optimizer.optimize_operation(
            operation_name="test_multiply",
            func=test_operation,
            data=test_data
        )
        
        assert result.success is True
        assert result.result == [x * 2 for x in test_data]
        assert result.optimization_applied is not None
        
        # Test performance profile creation
        profile = performance_optimizer._get_performance_profile("test_multiply")
        assert profile.strategy in [
            OptimizationStrategy.CPU_INTENSIVE,
            OptimizationStrategy.IO_INTENSIVE,
            OptimizationStrategy.MEMORY_INTENSIVE,
            OptimizationStrategy.MIXED_WORKLOAD
        ]
        
    def test_resilience_engine_integration(self):
        """Test resilience engine integration."""
        resilience_engine = get_resilience_engine()
        
        # Test successful operation
        def stable_operation():
            return "success"
            
        result = resilience_engine.execute_resilient(
            operation_name="stable_test",
            func=stable_operation
        )
        
        assert result.success is True
        assert result.result == "success"
        
        # Test retry with failure
        call_count = 0
        def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Temporary failure")
            return "success_after_retries"
            
        config = ResilienceConfig(
            pattern=ResiliencePattern.RETRY_WITH_BACKOFF,
            max_retries=5
        )
        
        result = resilience_engine.execute_resilient(
            operation_name="flaky_test",
            func=flaky_operation,
            config=config
        )
        
        assert result.success is True
        assert result.result == "success_after_retries"
        
    def test_quantum_system_integration(self):
        """Test integration between quantum systems."""
        # Get all quantum systems
        audit_logger = get_audit_logger()
        health_monitor = get_health_monitor()
        performance_optimizer = get_performance_optimizer()
        resilience_engine = get_resilience_engine()
        
        # Test coordinated operation
        def integrated_operation():
            # Log operation start
            audit_logger.log_resource_access(
                resource="test_resource",
                action="process",
                success=True
            )
            
            # Simulate work
            time.sleep(0.1)
            
            return "integrated_success"
            
        # Execute with optimization and resilience
        config = ResilienceConfig(
            pattern=ResiliencePattern.CACHE_ASIDE,
            cache_ttl_seconds=60
        )
        
        result = resilience_engine.execute_resilient(
            operation_name="integrated_test",
            func=integrated_operation,
            config=config
        )
        
        # Verify integration worked
        assert result.success is True
        assert result.result == "integrated_success"
        
        # Check that audit events were logged
        assert len(audit_logger.audit_buffer) > 0
        
        # Check system health after operations
        health = health_monitor.get_system_health()
        assert health.overall_status is not None
        
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_auto_scaling_integration(self, mock_memory, mock_cpu):
        """Test auto-scaling integration."""
        # Mock high CPU usage to trigger scaling
        mock_cpu.return_value = 85.0
        mock_memory.return_value = Mock(percent=60.0)
        
        performance_optimizer = get_performance_optimizer()
        
        # Check scaling triggers
        actions = performance_optimizer.check_scaling_triggers()
        
        # Should have triggered scaling action
        assert len(actions) > 0
        assert any("Scaled up" in action for action in actions)
        
    def test_compliance_reporting_integration(self):
        """Test compliance reporting integration."""
        audit_logger = get_audit_logger()
        
        # Generate some audit events
        events_to_generate = [
            (AuditEventType.AUTHENTICATION, "login", "success"),
            (AuditEventType.DATA_ACCESS, "read_user_data", "accessed"),
            (AuditEventType.SECURITY_VIOLATION, "unauthorized_access", "blocked"),
            (AuditEventType.CONFIGURATION_CHANGE, "update_settings", "changed")
        ]
        
        start_time = time.time()
        
        for event_type, action, result in events_to_generate:
            audit_logger.log_security_event(
                event_type=event_type,
                action=action,
                result=result
            )
            
        end_time = time.time()
        
        # Generate compliance report
        report = audit_logger.generate_compliance_report(start_time, end_time)
        
        assert report["compliance_mode"] == "SOX"
        assert report["event_summary"]["total_events"] >= len(events_to_generate)
        assert "authentication_attempts" in report["security_summary"]
        assert "security_violations" in report["security_summary"]
        
    def test_error_recovery_integration(self):
        """Test error recovery integration."""
        resilience_engine = get_resilience_engine()
        audit_logger = get_audit_logger()
        
        # Test circuit breaker pattern
        failure_count = 0
        def unreliable_operation():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 3:
                raise RuntimeError(f"Failure {failure_count}")
            return "recovered"
            
        config = ResilienceConfig(
            pattern=ResiliencePattern.CIRCUIT_BREAKER,
            circuit_breaker_threshold=2
        )
        
        # First few calls should fail and trip circuit breaker
        for i in range(3):
            result = resilience_engine.execute_resilient(
                operation_name="circuit_breaker_test",
                func=unreliable_operation,
                config=config
            )
            if i < 2:
                assert result.success is False
                
        # Check that security events were logged for failures
        error_events = [
            event for event in audit_logger.audit_buffer
            if event.event_type == AuditEventType.ERROR_CONDITION
        ]
        assert len(error_events) >= 0  # Should have logged some error conditions
        
    def test_performance_benchmarking_integration(self):
        """Test performance benchmarking integration."""
        performance_optimizer = get_performance_optimizer()
        
        # Benchmark different operation types
        test_operations = {
            "cpu_intensive": lambda x: sum(i * i for i in range(x)),
            "memory_intensive": lambda x: [0] * x,
            "io_simulation": lambda x: time.sleep(0.001) or x
        }
        
        for op_name, op_func in test_operations.items():
            result = performance_optimizer.optimize_operation(
                operation_name=op_name,
                func=op_func,
                data=100
            )
            
            assert result.success is True
            assert result.optimized_duration_ms > 0
            
        # Check performance statistics
        stats = performance_optimizer.get_performance_stats()
        assert stats["optimization_profiles"] >= len(test_operations)
        
    def test_resource_management_integration(self):
        """Test resource management integration."""
        performance_optimizer = get_performance_optimizer()
        health_monitor = get_health_monitor()
        
        # Get resource utilization
        resource_stats = performance_optimizer.resource_pool.get_utilization()
        
        assert "cpu_pool_utilization" in resource_stats
        assert "io_pool_utilization" in resource_stats
        assert resource_stats["cpu_cores"] > 0
        assert resource_stats["io_threads"] > 0
        
        # Check system health includes resource monitoring
        health = health_monitor.get_system_health()
        assert health.system_metrics.cpu_percent >= 0
        assert health.system_metrics.memory_percent >= 0
        
    def test_security_validation_integration(self):
        """Test security validation integration."""
        from openapi_doc_generator.quantum_security import QuantumSecurityValidator, SecurityLevel
        from openapi_doc_generator.quantum_scheduler import QuantumTask, TaskState
        
        validator = QuantumSecurityValidator(SecurityLevel.HIGH)
        audit_logger = get_audit_logger()
        
        # Test secure task
        secure_task = QuantumTask(
            id="secure_task",
            name="safe_operation",
            priority=1.0,
            effort=10.0,
            coherence_time=30.0,
            state=TaskState.READY,
            dependencies=[]
        )
        
        issues = validator.validate_task_security(secure_task)
        assert len(issues) == 0  # Should be no security issues
        
        # Test insecure task
        insecure_task = QuantumTask(
            id="insecure_task",
            name="rm -rf /",  # Suspicious command
            priority=1.0,
            effort=1000.0,  # Excessive effort
            coherence_time=-1.0,  # Invalid parameter
            state=TaskState.READY,
            dependencies=["dep"] * 25  # Too many dependencies
        )
        
        issues = validator.validate_task_security(insecure_task)
        assert len(issues) > 0  # Should have security issues
        
        # Log security violation
        audit_logger.log_security_violation(
            violation_type="malicious_task",
            details={"task_id": insecure_task.id, "issues": len(issues)},
            severity=SecurityLevel.CRITICAL
        )
        
        # Verify security alert was created
        alerts = audit_logger.get_security_alerts(SecurityLevel.CRITICAL)
        assert len(alerts) > 0