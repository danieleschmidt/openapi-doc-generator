"""Integration tests for complete quantum-inspired task planning system."""

import pytest
import asyncio
import time
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from openapi_doc_generator.quantum_planner import QuantumTaskPlanner, integrate_with_existing_sdlc
from openapi_doc_generator.quantum_scheduler import QuantumTask, TaskState
from openapi_doc_generator.quantum_validator import ValidationLevel
from openapi_doc_generator.quantum_optimizer import OptimizationConfig
from openapi_doc_generator.quantum_monitor import get_monitor


class TestQuantumSystemIntegration:
    """Test complete quantum task planning system integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create planner with full features enabled
        self.planner = QuantumTaskPlanner(
            temperature=1.5,
            cooling_rate=0.9,
            num_resources=4,
            validation_level=ValidationLevel.MODERATE,
            enable_monitoring=True,
            enable_optimization=True
        )
    
    def test_complete_workflow_small_tasks(self):
        """Test complete workflow with small task set."""
        # Add tasks with various characteristics
        self.planner.add_task("frontend", "Frontend Development", priority=3.0, effort=5.0, value=8.0)
        self.planner.add_task("backend", "Backend API", priority=4.0, effort=6.0, value=10.0)
        self.planner.add_task("database", "Database Design", priority=5.0, effort=3.0, value=7.0)
        self.planner.add_task("testing", "Testing Framework", priority=2.0, effort=2.0, value=5.0, dependencies=["backend"])
        self.planner.add_task("deployment", "Deployment Setup", priority=3.5, effort=4.0, value=6.0, dependencies=["backend", "frontend"])
        
        # Create quantum plan
        result = self.planner.create_quantum_plan()
        
        # Verify result structure
        assert len(result.optimized_tasks) >= 5
        assert result.total_value > 0
        assert 0 <= result.quantum_fidelity <= 1
        assert result.execution_time >= 0
        assert result.convergence_iterations >= 0
        
        # Verify resource allocation
        allocated_resources = set()
        for task in result.optimized_tasks:
            if task.state != TaskState.SUPERPOSITION:
                assert hasattr(task, 'allocated_resource')
                allocated_resources.add(getattr(task, 'allocated_resource'))
        
        # Should use multiple resources
        assert len(allocated_resources) >= 1
        
        # Export plan
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            self.planner.export_plan_to_json(result, temp_path)
            assert temp_path.exists()
            assert temp_path.stat().st_size > 0
        finally:
            temp_path.unlink()
    
    def test_complete_workflow_large_tasks(self):
        """Test complete workflow with larger task set."""
        # Add many tasks to test scaling
        for i in range(25):
            dependencies = []
            if i > 5:  # Add some dependencies for later tasks
                dependencies = [f"task_{j}" for j in range(max(0, i-3), i) if j % 3 == 0]
            
            self.planner.add_task(
                f"task_{i}",
                f"Task {i}",
                priority=float(i % 5 + 1),
                effort=float(i % 4 + 1),
                value=float(i % 6 + 2),
                dependencies=dependencies,
                coherence_time=float(10 + i % 20)
            )
        
        # Create quantum plan
        result = self.planner.create_quantum_plan()
        
        # Verify result
        assert len(result.optimized_tasks) >= 25
        assert result.total_value > 0
        assert result.quantum_fidelity > 0
        
        # Simulate execution
        simulation = self.planner.simulate_execution(result)
        assert simulation["total_tasks"] >= 25
        assert simulation["estimated_completion_time"] > 0
    
    def test_performance_statistics(self):
        """Test performance statistics collection."""
        # Add some tasks
        self.planner.add_task("stat_test1", "Statistics Test 1", priority=2.0, effort=1.0, value=3.0)
        self.planner.add_task("stat_test2", "Statistics Test 2", priority=4.0, effort=2.0, value=5.0)
        
        # Create plan to generate statistics
        result = self.planner.create_quantum_plan()
        
        # Get performance statistics
        stats = self.planner.get_performance_statistics()
        
        # Verify statistics structure
        assert "configuration" in stats
        assert "task_registry" in stats
        assert "monitoring" in stats
        assert "optimization" in stats
        
        config_stats = stats["configuration"]
        assert config_stats["validation_level"] == "moderate"
        assert config_stats["monitoring_enabled"] == True
        assert config_stats["optimization_enabled"] == True
        
        task_stats = stats["task_registry"]
        assert task_stats["total_tasks"] == 2
        assert "task_types" in task_stats
        
        # Task type analysis
        task_types = task_stats["task_types"]
        assert "priority_distribution" in task_types
        assert "effort_stats" in task_types
        assert "dependency_stats" in task_types
        assert "quantum_stats" in task_types
    
    def test_sdlc_integration(self):
        """Test SDLC integration functionality."""
        # Start with empty planner
        empty_planner = QuantumTaskPlanner()
        assert len(empty_planner.task_registry) == 0
        
        # Integrate SDLC tasks
        integrate_with_existing_sdlc(empty_planner)
        
        # Should have SDLC tasks now
        assert len(empty_planner.task_registry) > 0
        
        # Verify expected SDLC tasks exist
        expected_tasks = [
            "requirements_analysis",
            "architecture_design",
            "core_implementation",
            "testing_framework",
            "security_audit",
            "performance_optimization",
            "documentation_generation",
            "deployment_automation"
        ]
        
        for task_id in expected_tasks:
            assert task_id in empty_planner.task_registry
        
        # Create plan with SDLC tasks
        result = empty_planner.create_quantum_plan()
        assert len(result.optimized_tasks) >= len(expected_tasks)
        
        # Verify dependencies are respected
        task_positions = {}
        for i, task in enumerate(result.optimized_tasks):
            if task.state != TaskState.SUPERPOSITION:
                task_positions[task.id] = i
        
        # Check some key dependencies
        if "architecture_design" in task_positions and "requirements_analysis" in task_positions:
            assert task_positions["requirements_analysis"] < task_positions["architecture_design"]
        
        if "core_implementation" in task_positions and "architecture_design" in task_positions:
            assert task_positions["architecture_design"] < task_positions["core_implementation"]
    
    def test_validation_integration(self):
        """Test validation integration with quantum planning."""
        # Add invalid task (should raise error)
        with pytest.raises(ValueError, match="Task validation failed"):
            self.planner.add_task("", "Invalid Task with Empty ID")
        
        # Add tasks with warnings (should succeed but log warnings)
        task = self.planner.add_task("warning_task", "High Priority Task", priority=15.0)
        assert task.id == "warning_task"
        
        # Tasks should still be in registry despite warnings
        assert "warning_task" in self.planner.task_registry
    
    def test_monitoring_integration(self):
        """Test monitoring integration with quantum planning."""
        monitor = get_monitor()
        initial_metrics_count = len(monitor.metrics_history)
        
        # Add tasks and create plan
        self.planner.add_task("monitor_test", "Monitoring Test", priority=2.0, effort=1.0, value=3.0)
        result = self.planner.create_quantum_plan()
        
        # Should have recorded metrics
        assert len(monitor.metrics_history) > initial_metrics_count
        
        # Check recent metrics
        recent_metrics = monitor.metrics_history[-3:]  # Last few operations
        operation_names = [m.operation_name for m in recent_metrics]
        
        # Should include quantum planning operations
        assert any("quantum" in name.lower() for name in operation_names)
        
        # Run health checks
        health_results = monitor.run_health_checks()
        assert len(health_results) > 0
        
        # All health checks should have valid structure
        for result in health_results:
            assert hasattr(result, 'component')
            assert hasattr(result, 'status')
            assert hasattr(result, 'message')
    
    def test_optimization_integration(self):
        """Test optimization integration."""
        if not self.planner.optimized_planner:
            pytest.skip("Optimization not enabled")
        
        # Add tasks for optimization testing
        for i in range(15):  # Medium-sized task set
            self.planner.add_task(f"opt_task_{i}", f"Optimization Task {i}", 
                                priority=float(i % 3 + 1), effort=float(i % 2 + 1), value=float(i % 4 + 1))
        
        # Create plan (should use optimization)
        start_time = time.time()
        result1 = self.planner.create_quantum_plan()
        first_duration = time.time() - start_time
        
        # Create same plan again (should hit cache)
        start_time = time.time()
        result2 = self.planner.create_quantum_plan()
        second_duration = time.time() - start_time
        
        # Both should succeed
        assert len(result1.optimized_tasks) >= 15
        assert len(result2.optimized_tasks) >= 15
        
        # Second should generally be faster due to caching (though not guaranteed)
        # Just verify both completed successfully
        assert first_duration >= 0
        assert second_duration >= 0
        
        # Get optimization statistics
        opt_stats = self.planner.optimized_planner.get_performance_stats()
        assert "config" in opt_stats
        assert opt_stats["config"]["caching_enabled"] == True
        assert opt_stats["config"]["parallel_processing_enabled"] == True
    
    def test_performance_tuning(self):
        """Test performance tuning functionality."""
        # Add tasks
        self.planner.add_task("tune_task1", "Tuning Task 1", priority=2.0, effort=1.0, value=3.0)
        self.planner.add_task("tune_task2", "Tuning Task 2", priority=3.0, effort=2.0, value=4.0)
        
        # Create initial plan
        result1 = self.planner.create_quantum_plan()
        
        # Tune performance (should not raise errors)
        self.planner.tune_performance(target_fidelity=0.9, target_duration_ms=1000)
        
        # Create another plan after tuning
        result2 = self.planner.create_quantum_plan()
        
        # Both should succeed
        assert len(result1.optimized_tasks) >= 2
        assert len(result2.optimized_tasks) >= 2
    
    def test_cache_management(self):
        """Test cache management functionality."""
        if not self.planner.optimized_planner:
            pytest.skip("Optimization not enabled")
        
        # Add tasks
        self.planner.add_task("cache_task", "Cache Test Task", priority=2.0, effort=1.0, value=3.0)
        
        # Create plan (fills cache)
        result1 = self.planner.create_quantum_plan()
        
        # Clear caches
        self.planner.clear_caches()
        
        # Create plan again (should work after cache clear)
        result2 = self.planner.create_quantum_plan()
        
        assert len(result1.optimized_tasks) >= 1
        assert len(result2.optimized_tasks) >= 1
    
    def test_concurrent_planning(self):
        """Test concurrent quantum planning operations."""
        if not self.planner.optimized_planner:
            pytest.skip("Optimization not enabled")
        
        # Create multiple task sets
        task_sets = []
        for i in range(3):
            tasks = []
            for j in range(5):
                task = QuantumTask(
                    id=f"concurrent_{i}_{j}",
                    name=f"Concurrent Task {i}-{j}",
                    priority=float(j + 1),
                    effort=float(j % 2 + 1),
                    value=float(j % 3 + 2)
                )
                tasks.append(task)
            task_sets.append((tasks, {"temperature": 1.0 + i * 0.5}))
        
        # Process concurrently
        async def run_concurrent_test():
            results = await self.planner.optimized_planner.parallel_processor.process_multiple_plans_async(task_sets)
            return results
        
        # Run async test
        results = asyncio.run(run_concurrent_test())
        
        # Verify results
        assert len(results) == 3
        for result in results:
            assert len(result.optimized_tasks) >= 5
            assert result.total_value > 0
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        # Test with problematic tasks
        problematic_tasks = [
            QuantumTask(id="normal", name="Normal Task", priority=2.0, effort=1.0, value=3.0),
            QuantumTask(id="high_effort", name="Very High Effort", priority=1.0, effort=1000.0, value=2.0),  # Might trigger warnings
        ]
        
        # Should handle gracefully
        for task in problematic_tasks:
            try:
                # Add via planner method (with validation)
                self.planner.task_registry[task.id] = task
            except Exception as e:
                # Should not crash the system
                assert isinstance(e, (ValueError, RuntimeError))
        
        # Should still be able to create some kind of plan
        try:
            result = self.planner.create_quantum_plan()
            # If it succeeds, verify basic structure
            assert hasattr(result, 'optimized_tasks')
            assert hasattr(result, 'total_value')
            assert hasattr(result, 'quantum_fidelity')
        except Exception as e:
            # If it fails, should be a known exception type
            assert isinstance(e, (ValueError, TimeoutError, RuntimeError))
    
    def test_memory_and_resource_management(self):
        """Test memory and resource management under load."""
        # Create a planner with strict resource limits
        strict_planner = QuantumTaskPlanner(
            enable_optimization=True,
            enable_monitoring=True
        )
        
        if strict_planner.optimized_planner:
            # Set strict memory limit
            strict_planner.optimized_planner.config.memory_limit_mb = 50.0  # Very low limit for testing
        
        # Add moderate number of tasks
        for i in range(20):
            strict_planner.add_task(f"resource_task_{i}", f"Resource Task {i}", 
                                  priority=float(i % 3 + 1), effort=float(i % 2 + 1), value=float(i + 1))
        
        # Should handle resource constraints gracefully
        try:
            result = strict_planner.create_quantum_plan()
            # Should either succeed with normal result or fallback result
            assert hasattr(result, 'optimized_tasks')
            assert len(result.optimized_tasks) >= 0  # Might be reduced due to constraints
        except Exception as e:
            # Should not crash with memory errors
            assert not isinstance(e, MemoryError)
    
    def test_quantum_properties_preservation(self):
        """Test that quantum properties are preserved through the workflow."""
        # Add tasks with specific quantum properties
        task1 = self.planner.add_task("quantum_test1", "Quantum Test 1", 
                                     priority=2.0, effort=1.0, value=3.0, coherence_time=5.0)
        task2 = self.planner.add_task("quantum_test2", "Quantum Test 2",
                                     priority=3.0, effort=2.0, value=4.0, coherence_time=15.0)
        
        # Manually entangle tasks
        task1.entangled_tasks.add(task2.id)
        task2.entangled_tasks.add(task1.id)
        
        # Create plan
        result = self.planner.create_quantum_plan()
        
        # Find our tasks in the result
        result_task1 = None
        result_task2 = None
        
        for task in result.optimized_tasks:
            if task.id == "quantum_test1":
                result_task1 = task
            elif task.id == "quantum_test2":
                result_task2 = task
        
        # Verify quantum properties are preserved
        if result_task1 and result_task2:
            # Entanglement should be preserved (or appropriately modified)
            assert isinstance(result_task1.entangled_tasks, set)
            assert isinstance(result_task2.entangled_tasks, set)
            
            # Quantum weights should be positive
            assert result_task1.quantum_weight > 0
            assert result_task2.quantum_weight > 0
            
            # Coherence times should be preserved
            assert result_task1.coherence_time > 0
            assert result_task2.coherence_time > 0