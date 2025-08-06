"""Comprehensive tests for quantum task planning system."""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

# Import quantum components
from src.openapi_doc_generator.quantum_planner import QuantumTaskPlanner, integrate_with_existing_sdlc
from src.openapi_doc_generator.quantum_api import QuantumPlannerAPI, get_quantum_api
from src.openapi_doc_generator.quantum_security import QuantumSecurityValidator, SecurityLevel, get_security_validator
from src.openapi_doc_generator.quantum_recovery import QuantumRecoveryManager, RecoveryStrategy, get_recovery_manager
from src.openapi_doc_generator.quantum_scaler import QuantumTaskScaler, ScalingStrategy, ScalingConfig, get_quantum_scaler
from src.openapi_doc_generator.quantum_scheduler import QuantumTask, TaskState
from src.openapi_doc_generator.quantum_validator import ValidationLevel


class TestQuantumTaskPlanner:
    """Test quantum task planner functionality."""
    
    @pytest.fixture
    def planner(self):
        """Create a quantum task planner for testing."""
        return QuantumTaskPlanner(
            temperature=2.0,
            num_resources=4,
            validation_level=ValidationLevel.MODERATE,
            enable_monitoring=False,  # Disable for testing
            enable_optimization=True
        )
    
    def test_planner_initialization(self, planner):
        """Test planner initialization."""
        assert planner.scheduler.temperature == 2.0
        assert planner.allocator.num_resources == 4
        assert planner.validation_level == ValidationLevel.MODERATE
        assert len(planner.task_registry) == 0
    
    def test_add_task_basic(self, planner):
        """Test adding a basic task."""
        task = planner.add_task(
            task_id="test_task",
            name="Test Task",
            priority=3.0,
            effort=2.0,
            value=5.0
        )
        
        assert task.id == "test_task"
        assert task.name == "Test Task"
        assert task.priority == 3.0
        assert task.effort == 2.0
        assert task.value == 5.0
        assert len(planner.task_registry) == 1
    
    def test_add_task_with_dependencies(self, planner):
        """Test adding a task with dependencies."""
        # Add parent task first
        planner.add_task(task_id="parent", name="Parent Task")
        
        # Add child task with dependency
        child_task = planner.add_task(
            task_id="child",
            name="Child Task", 
            dependencies=["parent"]
        )
        
        assert "parent" in child_task.dependencies
    
    def test_add_task_security_validation(self, planner):
        """Test security validation during task addition."""
        # Test malicious task name
        with pytest.raises(ValueError, match="Security validation failed"):
            planner.add_task(
                task_id="evil_task",
                name="<script>alert('xss')</script>",
                priority=1.0
            )
    
    def test_create_quantum_plan_empty(self, planner):
        """Test creating plan with no tasks."""
        result = planner.create_quantum_plan()
        assert len(result.optimized_tasks) == 0
        assert result.total_value == 0.0
    
    def test_create_quantum_plan_with_tasks(self, planner):
        """Test creating plan with tasks."""
        # Add some test tasks
        planner.add_task("task1", "Task 1", priority=3.0, value=10.0)
        planner.add_task("task2", "Task 2", priority=2.0, value=8.0)
        planner.add_task("task3", "Task 3", priority=4.0, value=12.0, dependencies=["task1"])
        
        result = planner.create_quantum_plan()
        
        assert len(result.optimized_tasks) == 3
        assert result.total_value > 0
        assert result.quantum_fidelity > 0
        assert result.convergence_iterations >= 0
    
    def test_integrate_sdlc_tasks(self, planner):
        """Test SDLC task integration."""
        initial_count = len(planner.task_registry)
        integrate_with_existing_sdlc(planner)
        
        assert len(planner.task_registry) > initial_count
        
        # Check for specific SDLC tasks
        task_ids = list(planner.task_registry.keys())
        assert "requirements_analysis" in task_ids
        assert "architecture_design" in task_ids
        assert "core_implementation" in task_ids
    
    def test_export_plan_to_json(self, planner):
        """Test exporting plan to JSON."""
        # Add tasks and create plan
        planner.add_task("task1", "Task 1", priority=2.0, value=5.0)
        result = planner.create_quantum_plan()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = Path(f.name)
        
        try:
            planner.export_plan_to_json(result, output_path)
            
            # Verify file was created and contains valid JSON
            assert output_path.exists()
            
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            assert "quantum_schedule" in data
            assert "total_value" in data["quantum_schedule"]
            assert "tasks" in data["quantum_schedule"]
            
        finally:
            if output_path.exists():
                output_path.unlink()
    
    def test_simulate_execution(self, planner):
        """Test execution simulation."""
        planner.add_task("task1", "Task 1", effort=2.0)
        planner.add_task("task2", "Task 2", effort=3.0)
        
        result = planner.create_quantum_plan()
        simulation = planner.simulate_execution(result)
        
        assert "total_tasks" in simulation
        assert "estimated_completion_time" in simulation
        assert "resource_utilization" in simulation
        assert "quantum_effects" in simulation
        
        assert simulation["total_tasks"] == 2
        assert simulation["estimated_completion_time"] > 0
    
    def test_get_performance_statistics(self, planner):
        """Test performance statistics collection."""
        planner.add_task("task1", "Task 1")
        stats = planner.get_performance_statistics()
        
        assert "configuration" in stats
        assert "task_registry" in stats
        assert "security" in stats
        
        assert stats["task_registry"]["total_tasks"] == 1
        assert stats["configuration"]["validation_level"] == "moderate"
    
    def test_get_security_report(self, planner):
        """Test security report generation."""
        planner.add_task("task1", "Normal Task")
        planner.add_task("task2", "High Effort Task", effort=50.0)  # Should trigger warning
        
        report = planner.get_security_report()
        
        assert "security_score" in report
        assert "compliance_status" in report
        assert "detailed_issues" in report
        assert "recommendations" in report
        
        assert isinstance(report["security_score"], (int, float))
        assert report["compliance_status"] in ["PASS", "FAIL"]


class TestQuantumAPI:
    """Test quantum planner API."""
    
    @pytest.fixture
    def api(self):
        """Create a quantum API instance for testing."""
        return QuantumPlannerAPI()
    
    def test_create_session(self, api):
        """Test session creation."""
        response = api.create_session(
            session_id="test_session",
            temperature=2.5,
            num_resources=6
        )
        
        assert response["status"] == "success"
        assert response["session_id"] == "test_session"
        assert response["configuration"]["temperature"] == 2.5
        assert response["configuration"]["num_resources"] == 6
    
    def test_add_task_to_session(self, api):
        """Test adding task to session."""
        # Create session first
        api.create_session("test_session")
        
        # Add task
        task_data = {
            "id": "api_task",
            "name": "API Task",
            "priority": 3.0,
            "effort": 2.0,
            "value": 8.0
        }
        
        response = api.add_task("test_session", task_data)
        
        assert response["status"] == "success"
        assert response["task_id"] == "api_task"
        assert response["task"]["name"] == "API Task"
    
    def test_add_sdlc_tasks_to_session(self, api):
        """Test adding SDLC tasks to session."""
        api.create_session("test_session")
        
        response = api.add_sdlc_tasks("test_session")
        
        assert response["status"] == "success"
        assert response["tasks_added"] > 0
        assert response["total_tasks"] > 0
    
    def test_create_plan_through_api(self, api):
        """Test creating plan through API."""
        # Create session and add tasks
        api.create_session("test_session")
        api.add_sdlc_tasks("test_session")
        
        response = api.create_plan("test_session")
        
        assert response["status"] == "success"
        assert "quantum_plan" in response
        assert "simulation" in response
        assert "performance" in response
        
        plan = response["quantum_plan"]
        assert "total_value" in plan
        assert "quantum_fidelity" in plan
        assert "optimized_tasks" in plan
        assert len(plan["optimized_tasks"]) > 0
    
    def test_export_plan_json(self, api):
        """Test exporting plan as JSON."""
        # Create session with tasks
        api.create_session("test_session")
        api.add_task("test_session", {"id": "task1", "name": "Task 1"})
        
        response = api.export_plan("test_session", format="json")
        
        assert response["status"] == "success"
        assert response["format"] == "json"
        assert "data" in response
    
    def test_export_plan_markdown(self, api):
        """Test exporting plan as markdown."""
        # Create session with tasks
        api.create_session("test_session")
        api.add_task("test_session", {"id": "task1", "name": "Task 1"})
        
        response = api.export_plan("test_session", format="markdown")
        
        assert response["status"] == "success"
        assert response["format"] == "markdown"
        assert "data" in response
        assert "# Quantum Task Plan" in response["data"]
    
    def test_list_sessions(self, api):
        """Test listing sessions."""
        # Create multiple sessions
        api.create_session("session1")
        api.create_session("session2")
        
        response = api.list_sessions()
        
        assert response["status"] == "success"
        assert len(response["sessions"]) == 2
        assert response["total_sessions"] == 2
        
        session_ids = [s["session_id"] for s in response["sessions"]]
        assert "session1" in session_ids
        assert "session2" in session_ids
    
    def test_delete_session(self, api):
        """Test deleting session."""
        api.create_session("temp_session")
        
        response = api.delete_session("temp_session")
        
        assert response["status"] == "success"
        assert "temp_session" not in api.planners


class TestQuantumSecurity:
    """Test quantum security validation."""
    
    @pytest.fixture
    def validator(self):
        """Create security validator for testing."""
        return get_security_validator(SecurityLevel.MEDIUM)
    
    def test_validate_safe_task(self, validator):
        """Test validation of safe task."""
        task = QuantumTask(
            id="safe_task",
            name="Safe Task",
            priority=2.0,
            effort=1.0,
            value=3.0
        )
        
        issues = validator.validate_task_security(task)
        # Should have no critical or high severity issues
        critical_issues = [i for i in issues if i.severity.name in ["CRITICAL", "HIGH"]]
        assert len(critical_issues) == 0
    
    def test_validate_suspicious_task_name(self, validator):
        """Test validation of task with suspicious name."""
        task = QuantumTask(
            id="malicious_task",
            name="<script>alert('xss')</script>",
            priority=1.0,
            effort=1.0,
            value=1.0
        )
        
        issues = validator.validate_task_security(task)
        # Should detect injection risk
        injection_issues = [i for i in issues if i.issue_type == "injection_risk"]
        assert len(injection_issues) > 0
        assert injection_issues[0].severity == SecurityLevel.HIGH
    
    def test_validate_excessive_effort(self, validator):
        """Test validation of task with excessive effort."""
        task = QuantumTask(
            id="heavy_task",
            name="Heavy Task",
            priority=1.0,
            effort=200.0,  # Excessive effort
            value=1.0
        )
        
        issues = validator.validate_task_security(task)
        resource_issues = [i for i in issues if i.issue_type == "resource_abuse"]
        assert len(resource_issues) > 0
    
    def test_validate_plan_security(self, validator):
        """Test plan-level security validation."""
        tasks = [
            QuantumTask(id=f"task_{i}", name=f"Task {i}", priority=1.0, effort=1.0, value=1.0)
            for i in range(5)
        ]
        
        issues = validator.validate_plan_security(tasks)
        # Normal plan should have few issues
        critical_issues = [i for i in issues if i.severity == SecurityLevel.CRITICAL]
        assert len(critical_issues) == 0
    
    def test_generate_security_report(self, validator):
        """Test security report generation."""
        tasks = [
            QuantumTask(id="normal_task", name="Normal Task", priority=1.0, effort=1.0, value=1.0),
            QuantumTask(id="risky_task", name="<script>", priority=1.0, effort=150.0, value=1.0)  # Has issues
        ]
        
        report = validator.generate_security_report(tasks)
        
        assert "security_score" in report
        assert "total_issues" in report
        assert "issues_by_severity" in report
        assert "detailed_issues" in report
        assert "compliance_status" in report
        assert "recommendations" in report
        
        assert isinstance(report["security_score"], (int, float))
        assert report["total_issues"] > 0  # Should find issues in risky task
    
    def test_sanitize_input(self, validator):
        """Test input sanitization."""
        malicious_input = "<script>alert('xss')</script>../../../etc/passwd"
        sanitized = validator.sanitize_input(malicious_input)
        
        # Should remove dangerous patterns
        assert "<script>" not in sanitized
        assert "../" not in sanitized
        assert "/etc/" not in sanitized
    
    def test_rate_limiting(self, validator):
        """Test rate limiting functionality."""
        client_id = "test_client"
        
        # Should allow initial requests
        assert validator.check_rate_limit(client_id, max_requests=5, window_seconds=60)
        assert validator.check_rate_limit(client_id, max_requests=5, window_seconds=60)
        
        # Fill up the rate limit
        for _ in range(3):
            validator.check_rate_limit(client_id, max_requests=5, window_seconds=60)
        
        # Should now be rate limited
        assert not validator.check_rate_limit(client_id, max_requests=5, window_seconds=60)


class TestQuantumRecovery:
    """Test quantum recovery mechanisms."""
    
    @pytest.fixture
    def recovery_manager(self):
        """Create recovery manager for testing."""
        return get_recovery_manager()
    
    def test_retry_with_backoff_success(self, recovery_manager):
        """Test successful retry with backoff."""
        call_count = 0
        
        def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("Temporary failure")
            return "success"
        
        result = recovery_manager.retry_with_backoff(flaky_operation, "test_operation")
        
        assert result == "success"
        assert call_count == 2  # Failed once, succeeded on second try
    
    def test_retry_with_backoff_failure(self, recovery_manager):
        """Test retry with backoff when all attempts fail."""
        def always_fails():
            raise RuntimeError("Always fails")
        
        with pytest.raises(RuntimeError, match="Always fails"):
            recovery_manager.retry_with_backoff(always_fails, "failing_operation")
    
    def test_resilient_execution_context(self, recovery_manager):
        """Test resilient execution context manager."""
        def fallback_operation():
            return "fallback_result"
        
        with recovery_manager.resilient_execution(
            "test_operation",
            RecoveryStrategy.FALLBACK,
            fallback=fallback_operation
        ) as context:
            # Simulate operation failure
            raise RuntimeError("Operation failed")
    
    def test_circuit_breaker_functionality(self, recovery_manager):
        """Test circuit breaker functionality."""
        from src.openapi_doc_generator.quantum_recovery import CircuitBreaker
        
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        
        # Initially closed
        assert cb.can_execute()
        
        # Record failures to open circuit
        for _ in range(3):
            cb.record_failure()
            if cb.can_execute():  # May still be closed after first few failures
                continue
        
        # Should be open now
        assert not cb.can_execute()
    
    def test_get_recovery_statistics(self, recovery_manager):
        """Test recovery statistics collection."""
        stats = recovery_manager.get_recovery_statistics()
        
        assert "recovery_stats" in stats
        assert "circuit_breaker_states" in stats
        assert "active_policies" in stats
        
        assert isinstance(stats["recovery_stats"], dict)
        assert isinstance(stats["circuit_breaker_states"], dict)
        assert isinstance(stats["active_policies"], dict)


class TestQuantumScaler:
    """Test quantum scaling functionality."""
    
    @pytest.fixture
    def scaler(self):
        """Create quantum scaler for testing."""
        config = ScalingConfig(min_workers=2, max_workers=8)
        return QuantumTaskScaler(config)
    
    def test_scaler_initialization(self, scaler):
        """Test scaler initialization."""
        assert scaler.current_workers >= scaler.config.min_workers
        assert scaler.current_workers <= scaler.config.max_workers
        assert scaler.thread_pool is not None
    
    def test_get_performance_stats(self, scaler):
        """Test performance statistics collection."""
        stats = scaler.get_performance_stats()
        
        assert "scaling" in stats
        assert "performance" in stats
        assert "caching" in stats
        assert "history" in stats
        
        scaling_stats = stats["scaling"]
        assert "strategy" in scaling_stats
        assert "current_workers" in scaling_stats
        assert "min_workers" in scaling_stats
        assert "max_workers" in scaling_stats
    
    @pytest.mark.asyncio
    async def test_process_tasks_concurrent(self, scaler):
        """Test concurrent task processing."""
        # Create mock tasks
        tasks = [
            QuantumTask(id=f"task_{i}", name=f"Task {i}", priority=1.0, effort=1.0, value=1.0)
            for i in range(10)
        ]
        
        def mock_operation(task_batch):
            # Simulate processing
            return [f"processed_{task.id}" for task in task_batch]
        
        results = await scaler.process_tasks_concurrent(tasks, mock_operation)
        
        assert len(results) == 10
        assert all("processed_task_" in str(result) for result in results)
    
    def test_cache_functionality(self, scaler):
        """Test result caching."""
        # Test cache miss
        key = scaler._generate_cache_key("test_op", "arg1", "arg2")
        assert key not in scaler.result_cache
        
        # Cache a result
        scaler._cache_result(key, "test_result")
        assert key in scaler.result_cache
        assert scaler.result_cache[key] == "test_result"
        
        # Verify cache stats
        stats = scaler.get_performance_stats()
        assert stats["caching"]["result_cache_size"] > 0
    
    @pytest.mark.skip(reason="Requires psutil and may be flaky in test environment")
    def test_auto_scaling(self, scaler):
        """Test auto-scaling functionality."""
        initial_workers = scaler.current_workers
        
        # Force high CPU scenario (mocked)
        with patch('psutil.cpu_percent', return_value=90.0):
            scaler.auto_scale_workers()
            
            # Should scale up
            assert scaler.current_workers >= initial_workers
    
    def teardown_method(self):
        """Clean up after each test."""
        try:
            # Get the global scaler and shut it down
            scaler = get_quantum_scaler()
            scaler.shutdown()
        except:
            pass  # Ignore shutdown errors in tests


class TestIntegration:
    """Integration tests for quantum system components."""
    
    def test_end_to_end_quantum_planning(self):
        """Test complete quantum planning workflow."""
        # Create planner with all features enabled
        planner = QuantumTaskPlanner(
            temperature=2.0,
            num_resources=4,
            enable_monitoring=False,  # Disable for testing
            enable_optimization=True
        )
        
        # Add SDLC tasks
        integrate_with_existing_sdlc(planner)
        
        # Create quantum plan
        result = planner.create_quantum_plan()
        
        # Verify result
        assert len(result.optimized_tasks) > 0
        assert result.total_value > 0
        assert result.quantum_fidelity > 0
        
        # Simulate execution
        simulation = planner.simulate_execution(result)
        assert simulation["estimated_completion_time"] > 0
        
        # Get performance statistics
        stats = planner.get_performance_statistics()
        assert stats["security"]["compliance_status"] in ["PASS", "FAIL"]
    
    def test_api_to_planner_integration(self):
        """Test API integration with planner."""
        api = QuantumPlannerAPI()
        
        # Create session
        response = api.create_session("integration_test", temperature=2.0)
        assert response["status"] == "success"
        
        # Add custom task
        task_response = api.add_task("integration_test", {
            "id": "custom_task",
            "name": "Custom Integration Task", 
            "priority": 3.0,
            "effort": 2.0,
            "value": 5.0
        })
        assert task_response["status"] == "success"
        
        # Add SDLC tasks
        sdlc_response = api.add_sdlc_tasks("integration_test")
        assert sdlc_response["status"] == "success"
        
        # Create plan
        plan_response = api.create_plan("integration_test")
        assert plan_response["status"] == "success"
        assert len(plan_response["quantum_plan"]["optimized_tasks"]) > 1
        
        # Export plan
        export_response = api.export_plan("integration_test", format="markdown")
        assert export_response["status"] == "success"
        assert "# Quantum Task Plan" in export_response["data"]
    
    def test_security_integration(self):
        """Test security integration with planning."""
        planner = QuantumTaskPlanner(enable_monitoring=False)
        
        # Try to add malicious task (should be rejected)
        with pytest.raises(ValueError, match="Security validation failed"):
            planner.add_task(
                task_id="malicious",
                name="<script>alert('hack')</script>",
                priority=1.0
            )
        
        # Add normal task (should succeed)
        task = planner.add_task(
            task_id="normal",
            name="Normal Task",
            priority=2.0,
            effort=1.0,
            value=3.0
        )
        assert task.id == "normal"
        
        # Get security report
        report = planner.get_security_report()
        assert report["compliance_status"] in ["PASS", "FAIL"]
    
    @pytest.mark.asyncio
    async def test_scaling_integration(self):
        """Test scaling integration with task processing."""
        scaler = get_quantum_scaler()
        
        # Create mock tasks
        tasks = [
            QuantumTask(id=f"scale_task_{i}", name=f"Scale Task {i}", 
                       priority=1.0, effort=1.0, value=1.0)
            for i in range(20)
        ]
        
        def process_batch(batch):
            return [task.id for task in batch]
        
        # Process tasks concurrently
        results = await scaler.process_tasks_concurrent(tasks, process_batch, batch_size=5)
        
        # Verify all tasks were processed
        processed_ids = []
        for result in results:
            if isinstance(result, list):
                processed_ids.extend(result)
            else:
                processed_ids.append(result)
        
        assert len(processed_ids) == 20
        assert all(f"scale_task_{i}" in processed_ids for i in range(20))


# Performance benchmarks
@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmarks for quantum system."""
    
    def test_plan_creation_performance(self):
        """Benchmark plan creation performance."""
        planner = QuantumTaskPlanner(enable_monitoring=False, enable_optimization=True)
        
        # Add many tasks
        num_tasks = 100
        for i in range(num_tasks):
            planner.add_task(
                task_id=f"perf_task_{i}",
                name=f"Performance Task {i}",
                priority=float(i % 5 + 1),
                effort=float(i % 10 + 1),
                value=float(i % 15 + 1)
            )
        
        # Measure plan creation time
        import time
        start_time = time.time()
        result = planner.create_quantum_plan()
        end_time = time.time()
        
        planning_time = end_time - start_time
        
        # Performance assertions
        assert len(result.optimized_tasks) == num_tasks
        assert planning_time < 10.0  # Should complete within 10 seconds
        assert result.quantum_fidelity > 0.5  # Should achieve reasonable fidelity
        
        print(f"Planning {num_tasks} tasks took {planning_time:.2f} seconds")
        print(f"Quantum fidelity: {result.quantum_fidelity:.3f}")
        print(f"Total business value: {result.total_value:.2f}")
    
    @pytest.mark.asyncio
    async def test_concurrent_processing_performance(self):
        """Benchmark concurrent task processing."""
        scaler = get_quantum_scaler()
        
        # Create large number of tasks
        num_tasks = 1000
        tasks = [
            QuantumTask(id=f"concurrent_task_{i}", name=f"Concurrent Task {i}",
                       priority=1.0, effort=0.1, value=1.0)
            for i in range(num_tasks)
        ]
        
        def lightweight_operation(batch):
            return len(batch)  # Just return batch size
        
        # Measure concurrent processing time
        import time
        start_time = time.time()
        results = await scaler.process_tasks_concurrent(tasks, lightweight_operation, batch_size=50)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Performance assertions
        assert sum(results) == num_tasks  # All tasks processed
        assert processing_time < 5.0  # Should complete within 5 seconds
        
        print(f"Concurrent processing of {num_tasks} tasks took {processing_time:.2f} seconds")
        print(f"Throughput: {num_tasks / processing_time:.0f} tasks/second")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])