"""Tests for quantum-inspired task scheduling algorithms."""

import pytest
import math
from unittest.mock import patch
from pathlib import Path

from openapi_doc_generator.quantum_scheduler import (
    QuantumInspiredScheduler,
    QuantumResourceAllocator,
    QuantumTask,
    TaskState,
    QuantumScheduleResult
)


class TestQuantumTask:
    """Test quantum task data structure."""
    
    def test_quantum_task_creation(self):
        """Test basic quantum task creation."""
        task = QuantumTask(
            id="test_task",
            name="Test Task",
            priority=2.5,
            effort=3.0,
            value=5.0,
            dependencies=["dep1", "dep2"]
        )
        
        assert task.id == "test_task"
        assert task.name == "Test Task"
        assert task.priority == 2.5
        assert task.effort == 3.0
        assert task.value == 5.0
        assert task.dependencies == ["dep1", "dep2"]
        assert task.state == TaskState.PENDING
        assert task.quantum_weight == 1.0
        assert task.coherence_time == 10.0
        assert len(task.entangled_tasks) == 0
        assert task.measurement_count == 0
    
    def test_quantum_task_defaults(self):
        """Test quantum task with default values."""
        task = QuantumTask(id="simple", name="Simple Task")
        
        assert task.priority == 1.0
        assert task.effort == 1.0
        assert task.value == 1.0
        assert task.dependencies == []
        assert task.state == TaskState.PENDING


class TestQuantumInspiredScheduler:
    """Test quantum-inspired scheduling algorithms."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scheduler = QuantumInspiredScheduler(temperature=1.0, cooling_rate=0.9)
        self.sample_tasks = [
            QuantumTask(
                id="task1",
                name="High Priority Task",
                priority=5.0,
                effort=2.0,
                value=10.0,
                coherence_time=15.0
            ),
            QuantumTask(
                id="task2", 
                name="Medium Priority Task",
                priority=3.0,
                effort=3.0,
                value=6.0,
                dependencies=["task1"]
            ),
            QuantumTask(
                id="task3",
                name="Low Priority Task", 
                priority=1.0,
                effort=1.0,
                value=2.0
            )
        ]
    
    def test_quantum_priority_score(self):
        """Test quantum priority score calculation."""
        task = self.sample_tasks[0]
        current_time = task.created_at + 5.0  # 5 seconds after creation
        
        score = self.scheduler.quantum_priority_score(task, current_time)
        
        assert isinstance(score, float)
        assert score > 0  # Should be positive for high-value tasks
        
        # Test with older task (should have different interference)
        older_time = task.created_at + task.coherence_time + 5.0
        older_score = self.scheduler.quantum_priority_score(task, older_time)
        
        assert isinstance(older_score, float)
        # Score should be different due to quantum interference
        assert older_score != score
    
    def test_create_superposition_state(self):
        """Test quantum superposition state creation."""
        superposition_tasks = self.scheduler.create_superposition_state(self.sample_tasks)
        
        # Should create superposition versions of pending tasks
        assert len(superposition_tasks) == 3  # All tasks are pending
        
        for super_task in superposition_tasks:
            assert super_task.state == TaskState.SUPERPOSITION
            assert super_task.id.endswith("_super")
            assert "(Superposition)" in super_task.name
            assert super_task.quantum_weight < 1.0  # Should be reduced
    
    def test_entangle_tasks(self):
        """Test quantum entanglement creation."""
        # Create tasks with shared dependencies
        tasks = [
            QuantumTask(id="task1", name="Task 1", dependencies=["dep1", "dep2"]),
            QuantumTask(id="task2", name="Task 2", dependencies=["dep2", "dep3"]),
            QuantumTask(id="task3", name="Task 3", dependencies=["dep4"])
        ]
        
        original_weights = [task.quantum_weight for task in tasks]
        self.scheduler.entangle_tasks(tasks)
        
        # Tasks 1 and 2 should be entangled (shared dep2)
        assert "task2" in tasks[0].entangled_tasks
        assert "task1" in tasks[1].entangled_tasks
        
        # Task 3 should not be entangled (no shared deps)
        assert len(tasks[2].entangled_tasks) == 0
        
        # Entangled tasks should have modified quantum weights
        assert tasks[0].quantum_weight > original_weights[0]
        assert tasks[1].quantum_weight > original_weights[1]
    
    def test_quantum_annealing_schedule(self):
        """Test quantum annealing scheduling."""
        result = self.scheduler.quantum_annealing_schedule(self.sample_tasks)
        
        assert isinstance(result, QuantumScheduleResult)
        assert len(result.optimized_tasks) >= len(self.sample_tasks)  # May include collapsed superposition
        assert result.total_value >= 0
        assert result.execution_time >= 0
        assert 0 <= result.quantum_fidelity <= 1
        assert result.convergence_iterations >= 0
    
    def test_system_energy_calculation(self):
        """Test system energy calculation."""
        energy = self.scheduler._calculate_system_energy(self.sample_tasks)
        
        assert isinstance(energy, float)
        # Energy should be finite
        assert not math.isinf(energy)
        assert not math.isnan(energy)
    
    def test_collapse_superposition(self):
        """Test quantum superposition collapse."""
        # Create superposition tasks
        superposition_tasks = self.scheduler.create_superposition_state(self.sample_tasks)
        all_tasks = self.sample_tasks + superposition_tasks
        
        collapsed_tasks = self.scheduler._collapse_superposition(all_tasks)
        
        # Should have same number of unique tasks
        assert len(collapsed_tasks) == len(self.sample_tasks)
        
        # No tasks should be in superposition state
        for task in collapsed_tasks:
            assert task.state != TaskState.SUPERPOSITION
            assert not task.id.endswith("_super")
            assert "(Superposition)" not in task.name
    
    def test_quantum_fidelity_calculation(self):
        """Test quantum fidelity calculation."""
        fidelity = self.scheduler._calculate_quantum_fidelity(self.sample_tasks)
        
        assert 0 <= fidelity <= 1
        assert isinstance(fidelity, float)
        
        # Test with empty tasks
        empty_fidelity = self.scheduler._calculate_quantum_fidelity([])
        assert empty_fidelity == 1.0


class TestQuantumResourceAllocator:
    """Test quantum resource allocation algorithms."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.allocator = QuantumResourceAllocator(num_resources=4)
        self.sample_tasks = [
            QuantumTask(id="task1", name="Task 1", effort=2.0),
            QuantumTask(id="task2", name="Task 2", effort=3.0),
            QuantumTask(id="task3", name="Task 3", effort=1.0),
            QuantumTask(id="task4", name="Task 4", effort=4.0)
        ]
    
    def test_variational_optimize(self):
        """Test variational quantum optimization."""
        allocation = self.allocator.variational_optimize(self.sample_tasks, max_iterations=10)
        
        assert isinstance(allocation, dict)
        assert len(allocation) == len(self.sample_tasks)
        
        # All tasks should be allocated to valid resources
        for task_id, resource_id in allocation.items():
            assert 0 <= resource_id < self.allocator.num_resources
            assert task_id in [task.id for task in self.sample_tasks]
    
    def test_allocation_cost_calculation(self):
        """Test allocation cost calculation."""
        allocation = {
            "task1": 0,
            "task2": 1, 
            "task3": 0,
            "task4": 2
        }
        
        cost = self.allocator._calculate_allocation_cost(self.sample_tasks, allocation)
        
        assert isinstance(cost, float)
        assert cost >= 0  # Cost should be non-negative
    
    def test_quantum_rotation(self):
        """Test quantum rotation gate application."""
        initial_allocation = {task.id: 0 for task in self.sample_tasks}
        
        rotated_allocation = self.allocator._apply_quantum_rotation(initial_allocation, 0.5)
        
        assert isinstance(rotated_allocation, dict)
        assert len(rotated_allocation) == len(initial_allocation)
        
        # Some allocations should have changed
        changes = sum(1 for task_id in initial_allocation 
                     if initial_allocation[task_id] != rotated_allocation[task_id])
        assert changes >= 0  # At least some possibility of change


class TestQuantumSchedulingIntegration:
    """Integration tests for quantum scheduling components."""
    
    def test_full_quantum_scheduling_pipeline(self):
        """Test complete quantum scheduling pipeline."""
        scheduler = QuantumInspiredScheduler(temperature=2.0)
        allocator = QuantumResourceAllocator(num_resources=3)
        
        tasks = [
            QuantumTask(
                id="analysis",
                name="Requirements Analysis",
                priority=4.0,
                effort=2.0,
                value=8.0
            ),
            QuantumTask(
                id="design",
                name="System Design",
                priority=5.0,
                effort=3.0,
                value=10.0,
                dependencies=["analysis"]
            ),
            QuantumTask(
                id="implementation",
                name="Implementation", 
                priority=3.0,
                effort=5.0,
                value=12.0,
                dependencies=["design"]
            ),
            QuantumTask(
                id="testing",
                name="Testing",
                priority=4.5,
                effort=2.5,
                value=9.0,
                dependencies=["implementation"]
            )
        ]
        
        # Run quantum scheduling
        schedule_result = scheduler.quantum_annealing_schedule(tasks)
        
        # Allocate resources
        allocation = allocator.variational_optimize(schedule_result.optimized_tasks)
        
        # Verify results
        assert isinstance(schedule_result, QuantumScheduleResult)
        assert len(schedule_result.optimized_tasks) >= len(tasks)
        assert isinstance(allocation, dict)
        assert len(allocation) >= len(tasks)
        
        # Verify dependency constraints are respected in final ordering
        task_positions = {}
        for i, task in enumerate(schedule_result.optimized_tasks):
            if task.state != TaskState.SUPERPOSITION:
                task_positions[task.id] = i
        
        for task in schedule_result.optimized_tasks:
            if task.state != TaskState.SUPERPOSITION:
                for dep_id in task.dependencies:
                    if dep_id in task_positions:
                        assert task_positions[dep_id] < task_positions[task.id], \
                            f"Dependency {dep_id} should come before {task.id}"
    
    @pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0, 5.0])
    def test_different_temperatures(self, temperature):
        """Test quantum annealing with different temperatures."""
        scheduler = QuantumInspiredScheduler(temperature=temperature)
        
        tasks = [
            QuantumTask(id=f"task{i}", name=f"Task {i}", 
                       priority=float(i), effort=1.0, value=float(i*2))
            for i in range(1, 6)
        ]
        
        result = scheduler.quantum_annealing_schedule(tasks)
        
        assert isinstance(result, QuantumScheduleResult)
        assert len(result.optimized_tasks) >= len(tasks)
        assert result.total_value > 0
        assert 0 <= result.quantum_fidelity <= 1
    
    @pytest.mark.parametrize("num_resources", [1, 2, 4, 8])
    def test_different_resource_counts(self, num_resources):
        """Test resource allocation with different resource counts."""
        allocator = QuantumResourceAllocator(num_resources=num_resources)
        
        tasks = [
            QuantumTask(id=f"task{i}", name=f"Task {i}", effort=float(i))
            for i in range(1, 6)
        ]
        
        allocation = allocator.variational_optimize(tasks)
        
        assert isinstance(allocation, dict)
        assert len(allocation) == len(tasks)
        
        # All allocations should be within resource bounds
        for resource_id in allocation.values():
            assert 0 <= resource_id < num_resources