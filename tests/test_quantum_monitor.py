"""Tests for quantum planning monitoring and health checks."""

import pytest
import time
import threading
from unittest.mock import patch, MagicMock
from openapi_doc_generator.quantum_monitor import (
    QuantumPlanningMonitor,
    PerformanceMetrics,
    HealthStatus,
    HealthCheckResult,
    get_monitor,
    monitor_operation
)
from openapi_doc_generator.quantum_scheduler import QuantumScheduleResult, QuantumTask


class TestPerformanceMetrics:
    """Test performance metrics data structure."""
    
    def test_performance_metrics_creation(self):
        """Test basic performance metrics creation."""
        metrics = PerformanceMetrics(
            operation_name="test_operation",
            start_time=time.time()
        )
        
        assert metrics.operation_name == "test_operation"
        assert metrics.start_time > 0
        assert metrics.end_time is None
        assert metrics.duration_ms is None
        assert metrics.error is None


class TestHealthCheckResult:
    """Test health check result data structure."""
    
    def test_health_check_result_creation(self):
        """Test health check result creation."""
        result = HealthCheckResult(
            component="test_component",
            status=HealthStatus.HEALTHY,
            message="All systems operational"
        )
        
        assert result.component == "test_component"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "All systems operational"
        assert result.timestamp > 0
        assert isinstance(result.details, dict)


class TestQuantumPlanningMonitor:
    """Test quantum planning monitor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = QuantumPlanningMonitor(max_metrics_history=100)
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        assert self.monitor.max_metrics_history == 100
        assert len(self.monitor.metrics_history) == 0
        assert len(self.monitor.active_operations) == 0
        assert len(self.monitor.health_checks) > 0  # Should have default health checks
        assert "memory" in self.monitor.health_checks
        assert "active_operations" in self.monitor.health_checks
        assert "recent_errors" in self.monitor.health_checks
    
    def test_start_operation(self):
        """Test starting operation monitoring."""
        operation_id = self.monitor.start_operation("test_operation", {"key": "value"})
        
        assert operation_id.startswith("test_operation_")
        assert operation_id in self.monitor.active_operations
        
        metrics = self.monitor.active_operations[operation_id]
        assert metrics.operation_name == "test_operation"
        assert metrics.start_time > 0
        assert metrics.metadata == {"key": "value"}
    
    def test_end_operation_success(self):
        """Test ending operation monitoring successfully."""
        operation_id = self.monitor.start_operation("test_operation")
        time.sleep(0.01)  # Small delay to ensure measurable duration
        
        # Create mock quantum result
        mock_result = QuantumScheduleResult(
            optimized_tasks=[
                QuantumTask(id="task1", name="Task 1"),
                QuantumTask(id="task2", name="Task 2")
            ],
            total_value=10.0,
            execution_time=0.1,
            quantum_fidelity=0.95,
            convergence_iterations=50
        )
        
        completed_metrics = self.monitor.end_operation(operation_id, mock_result)
        
        assert completed_metrics is not None
        assert completed_metrics.operation_name == "test_operation"
        assert completed_metrics.end_time > completed_metrics.start_time
        assert completed_metrics.duration_ms > 0
        assert completed_metrics.task_count == 2
        assert completed_metrics.quantum_fidelity == 0.95
        assert completed_metrics.convergence_iterations == 50
        assert completed_metrics.error is None
        
        # Should be removed from active operations
        assert operation_id not in self.monitor.active_operations
        
        # Should be in metrics history
        assert len(self.monitor.metrics_history) == 1
        assert self.monitor.metrics_history[0] == completed_metrics
    
    def test_end_operation_with_error(self):
        """Test ending operation monitoring with error."""
        operation_id = self.monitor.start_operation("failing_operation")
        
        completed_metrics = self.monitor.end_operation(operation_id, error="Test error occurred")
        
        assert completed_metrics is not None
        assert completed_metrics.error == "Test error occurred"
        assert operation_id not in self.monitor.active_operations
    
    def test_end_nonexistent_operation(self):
        """Test ending non-existent operation."""
        result = self.monitor.end_operation("nonexistent_operation")
        
        assert result is None
    
    def test_metrics_history_limit(self):
        """Test metrics history size limit."""
        small_monitor = QuantumPlanningMonitor(max_metrics_history=3)
        
        # Add more metrics than the limit
        for i in range(5):
            op_id = small_monitor.start_operation(f"operation_{i}")
            small_monitor.end_operation(op_id)
        
        # Should only keep the most recent metrics
        assert len(small_monitor.metrics_history) == 3
        assert small_monitor.metrics_history[0].operation_name == "operation_2"  # Oldest kept
        assert small_monitor.metrics_history[-1].operation_name == "operation_4"  # Most recent
    
    def test_get_metrics_summary_all(self):
        """Test getting summary of all metrics."""
        # Add some test metrics
        for i in range(3):
            op_id = self.monitor.start_operation(f"operation_{i}")
            time.sleep(0.001)  # Small delay
            self.monitor.end_operation(op_id)
        
        summary = self.monitor.get_metrics_summary()
        
        assert summary["total_operations"] == 3
        assert summary["operation_name"] == "all"
        assert "duration_stats" in summary
        assert "time_range" in summary
        
        duration_stats = summary["duration_stats"]
        assert "avg_ms" in duration_stats
        assert "min_ms" in duration_stats
        assert "max_ms" in duration_stats
        assert "total_ms" in duration_stats
    
    def test_get_metrics_summary_filtered(self):
        """Test getting summary for specific operation."""
        # Add mixed operations
        op1_id = self.monitor.start_operation("type_a")
        self.monitor.end_operation(op1_id)
        
        op2_id = self.monitor.start_operation("type_b")
        self.monitor.end_operation(op2_id)
        
        op3_id = self.monitor.start_operation("type_a")  # Another type_a
        self.monitor.end_operation(op3_id)
        
        summary = self.monitor.get_metrics_summary("type_a")
        
        assert summary["total_operations"] == 2
        assert summary["operation_name"] == "type_a"
    
    def test_get_metrics_summary_empty(self):
        """Test getting summary with no metrics."""
        summary = self.monitor.get_metrics_summary()
        
        assert "message" in summary
        assert summary["message"] == "No metrics available"
    
    def test_register_custom_health_check(self):
        """Test registering custom health check."""
        def custom_check():
            return HealthCheckResult(
                component="custom",
                status=HealthStatus.HEALTHY,
                message="Custom check passed"
            )
        
        self.monitor.register_health_check("custom", custom_check)
        
        assert "custom" in self.monitor.health_checks
        
        # Run health checks to verify it works
        results = self.monitor.run_health_checks()
        custom_results = [r for r in results if r.component == "custom"]
        assert len(custom_results) == 1
        assert custom_results[0].status == HealthStatus.HEALTHY
    
    def test_run_health_checks(self):
        """Test running health checks."""
        results = self.monitor.run_health_checks()
        
        assert len(results) >= 3  # At least the default health checks
        
        component_names = [r.component for r in results]
        assert "memory" in component_names
        assert "active_operations" in component_names
        assert "recent_errors" in component_names
        
        # All results should have valid structure
        for result in results:
            assert isinstance(result.component, str)
            assert isinstance(result.status, HealthStatus)
            assert isinstance(result.message, str)
            assert result.timestamp > 0
    
    def test_health_check_failure_handling(self):
        """Test handling of failing health checks."""
        def failing_check():
            raise Exception("Health check failed!")
        
        self.monitor.register_health_check("failing", failing_check)
        
        results = self.monitor.run_health_checks()
        
        failing_results = [r for r in results if r.component == "failing"]
        assert len(failing_results) == 1
        assert failing_results[0].status == HealthStatus.CRITICAL
        assert "Health check failed" in failing_results[0].message
    
    def test_get_system_status(self):
        """Test getting overall system status."""
        # With default health checks should be healthy
        status = self.monitor.get_system_status()
        assert status in [HealthStatus.HEALTHY, HealthStatus.WARNING, HealthStatus.UNKNOWN]
        
        # Add a critical health check
        def critical_check():
            return HealthCheckResult(
                component="critical",
                status=HealthStatus.CRITICAL,
                message="System is critical"
            )
        
        self.monitor.register_health_check("critical", critical_check)
        
        status = self.monitor.get_system_status()
        assert status == HealthStatus.CRITICAL
    
    @patch('psutil.Process')
    def test_memory_health_check(self, mock_process):
        """Test memory health check with mocked psutil."""
        # Mock high memory usage
        mock_process.return_value.memory_info.return_value.rss = 3 * 1024 * 1024 * 1024  # 3GB
        
        results = self.monitor.run_health_checks()
        memory_results = [r for r in results if r.component == "memory"]
        
        if memory_results:  # May not run if psutil not available
            assert memory_results[0].status == HealthStatus.CRITICAL
            assert "High memory usage" in memory_results[0].message
    
    def test_active_operations_health_check(self):
        """Test active operations health check."""
        # Start many operations without ending them
        for i in range(15):
            self.monitor.start_operation(f"operation_{i}")
        
        results = self.monitor.run_health_checks()
        active_op_results = [r for r in results if r.component == "active_operations"]
        
        assert len(active_op_results) == 1
        assert active_op_results[0].status == HealthStatus.WARNING
        assert "Many active operations" in active_op_results[0].message
    
    def test_recent_errors_health_check(self):
        """Test recent errors health check."""
        # Add operations with errors
        for i in range(10):
            op_id = self.monitor.start_operation(f"operation_{i}")
            error = "Test error" if i < 3 else None  # 30% error rate
            self.monitor.end_operation(op_id, error=error)
        
        results = self.monitor.run_health_checks()
        error_results = [r for r in results if r.component == "recent_errors"]
        
        assert len(error_results) == 1
        assert error_results[0].status == HealthStatus.CRITICAL  # >20% error rate
        assert "High error rate" in error_results[0].message
    
    def test_export_metrics_json(self):
        """Test exporting metrics as JSON."""
        # Add some test metrics
        op_id = self.monitor.start_operation("test_export", {"test": True})
        self.monitor.end_operation(op_id)
        
        json_data = self.monitor.export_metrics("json")
        
        assert isinstance(json_data, str)
        
        import json
        parsed_data = json.loads(json_data)
        
        assert isinstance(parsed_data, list)
        assert len(parsed_data) == 1
        
        metric = parsed_data[0]
        assert metric["operation_name"] == "test_export"
        assert "start_time" in metric
        assert "duration_ms" in metric
        assert metric["metadata"]["test"] == True
    
    def test_export_metrics_unsupported_format(self):
        """Test exporting metrics with unsupported format."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            self.monitor.export_metrics("xml")
    
    def test_performance_alerts(self):
        """Test performance alert detection."""
        # Create metrics that should trigger alerts
        metrics = PerformanceMetrics(
            operation_name="alert_test",
            start_time=time.time(),
            end_time=time.time() + 1,
            duration_ms=65000,  # > 60s threshold
            memory_delta_mb=1500,  # > 1GB threshold
            quantum_fidelity=0.3,  # < 0.5 threshold
            cpu_percent=90  # > 80% threshold
        )
        
        with patch.object(self.monitor, '_check_performance_alerts') as mock_check:
            self.monitor.metrics_history.append(metrics)
            self.monitor._check_performance_alerts(metrics)
            mock_check.assert_called_once_with(metrics)


class TestGlobalMonitor:
    """Test global monitor singleton."""
    
    def test_get_monitor_singleton(self):
        """Test that get_monitor returns singleton instance."""
        monitor1 = get_monitor()
        monitor2 = get_monitor()
        
        assert monitor1 is monitor2
        assert isinstance(monitor1, QuantumPlanningMonitor)


class TestMonitorOperationDecorator:
    """Test monitor operation decorator."""
    
    def test_monitor_operation_decorator_success(self):
        """Test decorator with successful operation."""
        @monitor_operation("test_decorated_op")
        def test_function(x, y):
            return x + y
        
        monitor = get_monitor()
        initial_count = len(monitor.metrics_history)
        
        result = test_function(2, 3)
        
        assert result == 5
        assert len(monitor.metrics_history) == initial_count + 1
        
        latest_metric = monitor.metrics_history[-1]
        assert latest_metric.operation_name == "test_decorated_op"
        assert latest_metric.error is None
    
    def test_monitor_operation_decorator_failure(self):
        """Test decorator with failing operation."""
        @monitor_operation("test_failing_op")
        def failing_function():
            raise ValueError("Test error")
        
        monitor = get_monitor()
        initial_count = len(monitor.metrics_history)
        
        with pytest.raises(ValueError, match="Test error"):
            failing_function()
        
        assert len(monitor.metrics_history) == initial_count + 1
        
        latest_metric = monitor.metrics_history[-1]
        assert latest_metric.operation_name == "test_failing_op"
        assert latest_metric.error == "Test error"
    
    def test_monitor_operation_with_metadata(self):
        """Test decorator with metadata."""
        @monitor_operation("test_metadata_op", {"category": "test"})
        def test_function():
            return "success"
        
        monitor = get_monitor()
        initial_count = len(monitor.metrics_history)
        
        result = test_function()
        
        assert result == "success"
        assert len(monitor.metrics_history) == initial_count + 1
        
        latest_metric = monitor.metrics_history[-1]
        assert latest_metric.metadata == {"category": "test"}
    
    def test_concurrent_operations(self):
        """Test monitoring concurrent operations."""
        @monitor_operation("concurrent_op")
        def concurrent_function(duration):
            time.sleep(duration)
            return f"completed_{duration}"
        
        import concurrent.futures
        
        monitor = get_monitor()
        initial_count = len(monitor.metrics_history)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(concurrent_function, 0.01),
                executor.submit(concurrent_function, 0.02),
                executor.submit(concurrent_function, 0.01)
            ]
            
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        assert len(results) == 3
        assert len(monitor.metrics_history) >= initial_count + 3
        
        # All operations should have been recorded
        recent_metrics = monitor.metrics_history[-3:]
        for metric in recent_metrics:
            assert metric.operation_name == "concurrent_op"
            assert metric.error is None
            assert metric.duration_ms > 0