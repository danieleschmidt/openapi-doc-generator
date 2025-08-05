"""Tests for quantum-inspired task planning integration."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from openapi_doc_generator.quantum_planner import (
    QuantumTaskPlanner,
    integrate_with_existing_sdlc
)
from openapi_doc_generator.quantum_scheduler import (
    QuantumTask,
    TaskState,
    QuantumScheduleResult
)


class TestQuantumTaskPlanner:
    """Test quantum task planner integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.planner = QuantumTaskPlanner(
            temperature=1.5,
            cooling_rate=0.9,
            num_resources=3
        )
    
    def test_planner_initialization(self):
        """Test quantum planner initialization."""
        assert self.planner.scheduler.temperature == 1.5
        assert self.planner.scheduler.cooling_rate == 0.9
        assert self.planner.allocator.num_resources == 3
        assert len(self.planner.task_registry) == 0
    
    def test_add_task(self):
        """Test adding tasks to quantum planner."""
        task = self.planner.add_task(
            task_id="test_task",
            name="Test Task",
            priority=3.0,
            effort=2.5,
            value=7.0,
            dependencies=["dep1"],
            coherence_time=20.0
        )
        
        assert isinstance(task, QuantumTask)
        assert task.id == "test_task"
        assert task.name == "Test Task"
        assert task.priority == 3.0
        assert task.effort == 2.5
        assert task.value == 7.0
        assert task.dependencies == ["dep1"]
        assert task.coherence_time == 20.0
        
        # Check task is registered
        assert "test_task" in self.planner.task_registry
        assert self.planner.task_registry["test_task"] == task
    
    def test_add_task_with_defaults(self):
        """Test adding task with default parameters."""
        task = self.planner.add_task("simple", "Simple Task")
        
        assert task.priority == 1.0
        assert task.effort == 1.0
        assert task.value == 1.0
        assert task.dependencies == []
        assert task.coherence_time == 10.0
    
    def test_create_quantum_plan_empty(self):
        """Test creating quantum plan with no tasks."""
        result = self.planner.create_quantum_plan()
        
        assert isinstance(result, QuantumScheduleResult)
        assert len(result.optimized_tasks) == 0
        assert result.total_value == 0.0
        assert result.execution_time == 0.0
        assert result.quantum_fidelity == 1.0
        assert result.convergence_iterations == 0
    
    def test_create_quantum_plan_with_tasks(self):
        """Test creating quantum plan with tasks."""
        # Add some tasks
        self.planner.add_task("task1", "First Task", priority=2.0, value=5.0)
        self.planner.add_task("task2", "Second Task", priority=3.0, value=8.0, dependencies=["task1"])
        self.planner.add_task("task3", "Third Task", priority=1.0, value=3.0)
        
        result = self.planner.create_quantum_plan()
        
        assert isinstance(result, QuantumScheduleResult)
        assert len(result.optimized_tasks) >= 3  # At least our tasks
        assert result.total_value > 0
        assert result.execution_time >= 0
        assert 0 <= result.quantum_fidelity <= 1
        assert result.convergence_iterations >= 0
        
        # Check that tasks have resource allocation
        for task in result.optimized_tasks:
            if task.state != TaskState.SUPERPOSITION:
                assert hasattr(task, 'allocated_resource')
                assert isinstance(getattr(task, 'allocated_resource'), int)
    
    def test_export_plan_to_json(self):
        """Test exporting quantum plan to JSON."""
        # Add tasks and create plan
        self.planner.add_task("task1", "Task One", priority=2.0, effort=1.5, value=4.0)
        self.planner.add_task("task2", "Task Two", priority=3.0, effort=2.0, value=6.0)
        
        result = self.planner.create_quantum_plan()
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            self.planner.export_plan_to_json(result, temp_path)
            
            # Verify file was created and contains valid JSON
            assert temp_path.exists()
            
            with open(temp_path, 'r') as f:
                plan_data = json.load(f)
            
            assert "quantum_schedule" in plan_data
            schedule = plan_data["quantum_schedule"]
            
            assert "total_value" in schedule
            assert "execution_time" in schedule
            assert "quantum_fidelity" in schedule
            assert "convergence_iterations" in schedule
            assert "tasks" in schedule
            
            assert isinstance(schedule["tasks"], list)
            assert len(schedule["tasks"]) >= 2
            
            # Check task structure
            for task_data in schedule["tasks"]:
                assert "id" in task_data
                assert "name" in task_data
                assert "execution_order" in task_data
                assert "allocated_resource" in task_data
                assert isinstance(task_data["entangled_tasks"], list)
                assert isinstance(task_data["state"], str)
        
        finally:
            temp_path.unlink()  # Clean up
    
    def test_import_classical_tasks(self):
        """Test importing classical tasks to quantum format."""
        classical_tasks = [
            {
                "id": "classic1",
                "name": "Classical Task 1",
                "priority": 3.0,
                "effort": 2.0,
                "business_value": 6.0,
                "dependencies": ["dep1"],
                "urgency": 2.0
            },
            {
                "title": "Classical Task 2",  # Using title instead of name
                "story_points": 5.0,  # Using story_points instead of effort
                "value": 8.0,
                "blockers": ["blocker1", "blocker2"],  # Using blockers
                "urgency": 4.0  # High urgency should reduce coherence time
            }
        ]
        
        quantum_tasks = self.planner.import_classical_tasks(classical_tasks)
        
        assert len(quantum_tasks) == 2
        
        # Check first task
        task1 = quantum_tasks[0]
        assert task1.id == "classic1"
        assert task1.name == "Classical Task 1"
        assert task1.priority == 3.0
        assert task1.effort == 2.0
        assert task1.value == 6.0
        assert task1.dependencies == ["dep1"]
        assert task1.coherence_time == 10.0  # 20/2.0
        
        # Check second task
        task2 = quantum_tasks[1]
        assert task2.name == "Classical Task 2"
        assert task2.effort == 5.0  # From story_points
        assert task2.value == 8.0
        assert "blocker1" in task2.dependencies
        assert "blocker2" in task2.dependencies
        assert task2.coherence_time == 5.0  # 20/4.0 (high urgency)
        
        # Check tasks are registered
        assert "classic1" in self.planner.task_registry
        assert len(self.planner.task_registry) == 2
    
    def test_get_task_quantum_metrics(self):
        """Test getting quantum metrics for tasks."""
        # Test non-existent task
        metrics = self.planner.get_task_quantum_metrics("nonexistent")
        assert metrics is None
        
        # Add task and get metrics
        task = self.planner.add_task("test", "Test Task", coherence_time=15.0)
        task.quantum_weight = 0.8
        task.measurement_count = 2
        task.entangled_tasks.add("other_task")
        
        metrics = self.planner.get_task_quantum_metrics("test")
        
        assert isinstance(metrics, dict)
        assert metrics["quantum_weight"] == 0.8
        assert metrics["coherence_time"] == 15.0
        assert metrics["measurement_count"] == 2
        assert metrics["entanglement_degree"] == 1
        assert "quantum_priority" in metrics
        assert isinstance(metrics["quantum_priority"], float)
    
    def test_simulate_execution(self):
        """Test execution simulation."""
        # Create quantum plan
        self.planner.add_task("task1", "Task 1", effort=2.0)
        self.planner.add_task("task2", "Task 2", effort=3.0)
        result = self.planner.create_quantum_plan()
        
        simulation = self.planner.simulate_execution(result)
        
        assert isinstance(simulation, dict)
        assert "total_tasks" in simulation
        assert "estimated_completion_time" in simulation
        assert "resource_utilization" in simulation
        assert "quantum_effects" in simulation
        
        assert simulation["total_tasks"] >= 2
        assert simulation["estimated_completion_time"] >= 0
        assert isinstance(simulation["resource_utilization"], dict)
        
        quantum_effects = simulation["quantum_effects"]
        assert "superposition_collapses" in quantum_effects
        assert "entanglement_breaks" in quantum_effects
        assert "coherence_loss" in quantum_effects


class TestSDLCIntegration:
    """Test SDLC integration functionality."""
    
    def test_integrate_with_existing_sdlc(self):
        """Test integration with existing SDLC systems."""
        planner = QuantumTaskPlanner()
        
        # Should start empty
        assert len(planner.task_registry) == 0
        
        # Integrate SDLC tasks
        integrate_with_existing_sdlc(planner)
        
        # Should have added SDLC tasks
        assert len(planner.task_registry) > 0
        
        # Check for expected SDLC tasks
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
            assert task_id in planner.task_registry
            task = planner.task_registry[task_id]
            assert isinstance(task, QuantumTask)
            assert task.priority > 0
            assert task.effort > 0
            assert task.value > 0
            assert task.coherence_time > 0
        
        # Check dependencies are set correctly
        design_task = planner.task_registry["architecture_design"]
        assert "requirements_analysis" in design_task.dependencies
        
        impl_task = planner.task_registry["core_implementation"]
        assert "architecture_design" in impl_task.dependencies
        
        security_task = planner.task_registry["security_audit"]
        assert "core_implementation" in security_task.dependencies
        
        deployment_task = planner.task_registry["deployment_automation"]
        assert "testing_framework" in deployment_task.dependencies
        assert "security_audit" in deployment_task.dependencies
    
    def test_sdlc_task_properties(self):
        """Test that SDLC tasks have appropriate quantum properties."""
        planner = QuantumTaskPlanner()
        integrate_with_existing_sdlc(planner)
        
        # Security tasks should have shorter coherence time (urgent)
        security_task = planner.task_registry["security_audit"]
        assert security_task.coherence_time == 8.0  # Should be relatively short
        
        # Documentation can wait longer
        doc_task = planner.task_registry["documentation_generation"] 
        assert doc_task.coherence_time == 30.0  # Should be longer
        
        # Core implementation should have high value
        impl_task = planner.task_registry["core_implementation"]
        assert impl_task.value == 10.0  # High value
        assert impl_task.priority == 5.0  # High priority
        
        # Check that tasks have reasonable effort estimates
        for task in planner.task_registry.values():
            assert 1.0 <= task.effort <= 10.0  # Reasonable range
            assert 0.5 <= task.priority <= 5.0  # Reasonable priority range
            assert 1.0 <= task.value <= 15.0  # Reasonable value range


class TestQuantumPlannerEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_duplicate_task_ids(self):
        """Test handling of duplicate task IDs."""
        planner = QuantumTaskPlanner()
        
        # Add first task
        task1 = planner.add_task("duplicate", "First Task", value=5.0)
        assert planner.task_registry["duplicate"].value == 5.0
        
        # Add second task with same ID (should overwrite)
        task2 = planner.add_task("duplicate", "Second Task", value=10.0)
        assert planner.task_registry["duplicate"].value == 10.0
        assert planner.task_registry["duplicate"].name == "Second Task"
        
        # Should still only have one task with that ID
        assert len(planner.task_registry) == 1
    
    def test_large_number_of_tasks(self):
        """Test performance with larger number of tasks."""
        planner = QuantumTaskPlanner()
        
        # Add many tasks
        num_tasks = 50
        for i in range(num_tasks):
            planner.add_task(
                f"task_{i}",
                f"Task {i}",
                priority=float(i % 5 + 1),
                effort=float(i % 3 + 1),
                value=float(i % 10 + 1)
            )
        
        assert len(planner.task_registry) == num_tasks
        
        # Should still be able to create plan (though may take longer)
        result = planner.create_quantum_plan()
        assert isinstance(result, QuantumScheduleResult)
        assert len(result.optimized_tasks) >= num_tasks
    
    def test_circular_dependencies(self):
        """Test handling of circular dependencies."""
        planner = QuantumTaskPlanner()
        
        # Create circular dependencies: A -> B -> C -> A
        planner.add_task("taskA", "Task A", dependencies=["taskC"])
        planner.add_task("taskB", "Task B", dependencies=["taskA"])
        planner.add_task("taskC", "Task C", dependencies=["taskB"])
        
        # Should still be able to create a plan (scheduler should handle gracefully)
        result = planner.create_quantum_plan()
        assert isinstance(result, QuantumScheduleResult)
        assert len(result.optimized_tasks) >= 3
    
    def test_missing_dependencies(self):
        """Test tasks with dependencies that don't exist."""
        planner = QuantumTaskPlanner()
        
        # Add task with non-existent dependency
        planner.add_task("task1", "Task 1", dependencies=["nonexistent_task"])
        planner.add_task("task2", "Task 2")
        
        # Should still create plan without errors
        result = planner.create_quantum_plan()
        assert isinstance(result, QuantumScheduleResult)
        assert len(result.optimized_tasks) >= 2
    
    def test_zero_values(self):
        """Test tasks with zero or very small values."""
        planner = QuantumTaskPlanner()
        
        planner.add_task("zero_effort", "Zero Effort", effort=0.0, value=5.0)
        planner.add_task("zero_value", "Zero Value", effort=2.0, value=0.0)
        planner.add_task("tiny_values", "Tiny Values", effort=0.001, value=0.001)
        
        # Should handle gracefully without division by zero
        result = planner.create_quantum_plan()
        assert isinstance(result, QuantumScheduleResult)
        assert len(result.optimized_tasks) >= 3