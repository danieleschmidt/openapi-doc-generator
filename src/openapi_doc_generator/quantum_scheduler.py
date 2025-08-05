"""Quantum-inspired task scheduling and optimization algorithms."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class TaskState(Enum):
    """Quantum-inspired task states using superposition concepts."""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    BLOCKED = "blocked"
    SUPERPOSITION = "superposition"  # Task in multiple states simultaneously


@dataclass
class QuantumTask:
    """Task with quantum-inspired properties."""
    id: str
    name: str
    priority: float = 1.0
    effort: float = 1.0
    value: float = 1.0
    dependencies: List[str] = field(default_factory=list)
    state: TaskState = TaskState.PENDING
    quantum_weight: float = 1.0  # Quantum probability amplitude
    coherence_time: float = 10.0  # How long task maintains quantum properties
    entangled_tasks: Set[str] = field(default_factory=set)
    measurement_count: int = 0
    created_at: float = field(default_factory=time.time)


@dataclass 
class QuantumScheduleResult:
    """Result of quantum-inspired scheduling optimization."""
    optimized_tasks: List[QuantumTask]
    total_value: float
    execution_time: float
    quantum_fidelity: float  # How well quantum properties were preserved
    convergence_iterations: int


class QuantumInspiredScheduler:
    """Quantum-inspired task scheduler using superposition and entanglement concepts."""
    
    def __init__(self, temperature: float = 1.0, cooling_rate: float = 0.95):
        """Initialize quantum scheduler with annealing parameters."""
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.random = random.Random(42)  # Deterministic for testing
        
    def quantum_priority_score(self, task: QuantumTask, current_time: float) -> float:
        """Calculate quantum-inspired priority score using interference patterns."""
        # Base quantum amplitude
        amplitude = math.sqrt(task.quantum_weight)
        
        # Quantum interference based on task age
        age = current_time - task.created_at
        phase = 2 * math.pi * age / task.coherence_time
        interference = math.cos(phase) ** 2
        
        # Value-effort ratio with quantum uncertainty
        uncertainty_factor = 1 + 0.1 * math.sin(phase)
        base_score = (task.value / max(task.effort, 0.1)) * uncertainty_factor
        
        # Apply quantum amplitude and interference
        quantum_score = amplitude * base_score * interference
        
        # Add measurement collapse penalty (tasks lose quantum properties when measured)
        measurement_penalty = 1 / (1 + 0.1 * task.measurement_count)
        
        return quantum_score * measurement_penalty
    
    def create_superposition_state(self, tasks: List[QuantumTask]) -> List[QuantumTask]:
        """Create quantum superposition of task states for parallel exploration."""
        superposition_tasks = []
        
        for task in tasks:
            if task.state == TaskState.PENDING:
                # Create superposition copy
                super_task = QuantumTask(
                    id=f"{task.id}_super",
                    name=f"{task.name} (Superposition)",
                    priority=task.priority,
                    effort=task.effort,
                    value=task.value,
                    dependencies=task.dependencies.copy(),
                    state=TaskState.SUPERPOSITION,
                    quantum_weight=task.quantum_weight * 0.707,  # √2/2 for equal superposition
                    coherence_time=task.coherence_time,
                    entangled_tasks=task.entangled_tasks.copy(),
                    created_at=task.created_at
                )
                superposition_tasks.append(super_task)
        
        return superposition_tasks
    
    def entangle_tasks(self, tasks: List[QuantumTask]) -> None:
        """Create quantum entanglement between related tasks."""
        for i, task1 in enumerate(tasks):
            for j, task2 in enumerate(tasks[i+1:], i+1):
                # Entangle if tasks have shared dependencies or similar domains
                shared_deps = set(task1.dependencies) & set(task2.dependencies)
                similarity = len(shared_deps) / max(len(task1.dependencies) + len(task2.dependencies), 1)
                
                if similarity > 0.3:  # Entanglement threshold
                    task1.entangled_tasks.add(task2.id)
                    task2.entangled_tasks.add(task1.id)
                    
                    # Adjust quantum weights based on entanglement
                    entanglement_factor = math.sqrt(similarity)
                    task1.quantum_weight *= (1 + entanglement_factor)
                    task2.quantum_weight *= (1 + entanglement_factor)
    
    def quantum_annealing_schedule(self, tasks: List[QuantumTask]) -> QuantumScheduleResult:
        """Use quantum annealing to find optimal task schedule."""
        start_time = time.time()
        current_temp = self.temperature
        current_tasks = tasks.copy()
        best_tasks = tasks.copy()
        best_energy = self._calculate_system_energy(tasks)
        iterations = 0
        
        # Create quantum superposition states
        superposition_tasks = self.create_superposition_state(tasks)
        current_tasks.extend(superposition_tasks)
        
        # Establish quantum entanglements
        self.entangle_tasks(current_tasks)
        
        while current_temp > 0.01 and iterations < 1000:
            iterations += 1
            
            # Generate new configuration by swapping tasks
            new_tasks = self._quantum_mutation(current_tasks.copy(), current_temp)
            
            # Calculate energy difference
            current_energy = self._calculate_system_energy(current_tasks)
            new_energy = self._calculate_system_energy(new_tasks)
            energy_diff = new_energy - current_energy
            
            # Quantum acceptance probability (includes tunneling)
            if energy_diff < 0 or self.random.random() < math.exp(-energy_diff / current_temp):
                current_tasks = new_tasks
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_tasks = current_tasks.copy()
                    best_energy = current_energy
            
            # Cool down (annealing)
            current_temp *= self.cooling_rate
        
        # Collapse superposition states (quantum measurement)
        final_tasks = self._collapse_superposition(best_tasks)
        
        # Calculate quantum fidelity
        fidelity = self._calculate_quantum_fidelity(final_tasks)
        
        execution_time = time.time() - start_time
        total_value = sum(task.value for task in final_tasks if task.state != TaskState.SUPERPOSITION)
        
        return QuantumScheduleResult(
            optimized_tasks=final_tasks,
            total_value=total_value,
            execution_time=execution_time,
            quantum_fidelity=fidelity,
            convergence_iterations=iterations
        )
    
    def _quantum_mutation(self, tasks: List[QuantumTask], temperature: float) -> List[QuantumTask]:
        """Apply quantum-inspired mutations to task ordering."""
        mutated_tasks = tasks.copy()
        
        # Number of mutations based on temperature (quantum tunneling)
        num_mutations = max(1, int(temperature * len(tasks) * 0.1))
        
        for _ in range(num_mutations):
            if len(mutated_tasks) > 1:
                # Quantum swap: respect entanglement
                i, j = self.random.sample(range(len(mutated_tasks)), 2)
                task_i, task_j = mutated_tasks[i], mutated_tasks[j]
                
                # Higher probability of swapping entangled tasks
                if task_j.id in task_i.entangled_tasks:
                    swap_prob = 0.8
                else:
                    swap_prob = 0.3
                
                if self.random.random() < swap_prob:
                    mutated_tasks[i], mutated_tasks[j] = mutated_tasks[j], mutated_tasks[i]
        
        return mutated_tasks
    
    def _calculate_system_energy(self, tasks: List[QuantumTask]) -> float:
        """Calculate total system energy (lower is better)."""
        energy = 0.0
        current_time = time.time()
        
        for i, task in enumerate(tasks):
            if task.state == TaskState.SUPERPOSITION:
                continue
                
            # Priority-based energy (negative because higher priority = lower energy)
            priority_energy = -self.quantum_priority_score(task, current_time)
            
            # Dependency violation penalty
            dependency_penalty = 0.0
            for dep_id in task.dependencies:
                dep_found = False
                for j, dep_task in enumerate(tasks[:i]):  # Only check previous tasks
                    if dep_task.id == dep_id and dep_task.state == TaskState.COMPLETED:
                        dep_found = True
                        break
                if not dep_found:
                    dependency_penalty += 10.0  # High penalty for unmet dependencies
            
            # Quantum decoherence penalty
            age = current_time - task.created_at
            if age > task.coherence_time:
                decoherence_penalty = (age - task.coherence_time) * 0.1
            else:
                decoherence_penalty = 0.0
            
            energy += priority_energy + dependency_penalty + decoherence_penalty
        
        return energy
    
    def _collapse_superposition(self, tasks: List[QuantumTask]) -> List[QuantumTask]:
        """Collapse quantum superposition states through measurement."""
        collapsed_tasks = []
        
        for task in tasks:
            if task.state == TaskState.SUPERPOSITION:
                # Measurement collapses superposition
                task.measurement_count += 1
                task.state = TaskState.PENDING
                task.quantum_weight *= 0.9  # Lose some quantum properties
                
                # Remove superposition suffix from ID and name
                if task.id.endswith("_super"):
                    task.id = task.id[:-6]
                if task.name.endswith(" (Superposition)"):
                    task.name = task.name[:-15]
            
            collapsed_tasks.append(task)
        
        # Remove duplicates (keep the collapsed version)
        unique_tasks = {}
        for task in collapsed_tasks:
            if task.id not in unique_tasks:
                unique_tasks[task.id] = task
        
        return list(unique_tasks.values())
    
    def _calculate_quantum_fidelity(self, tasks: List[QuantumTask]) -> float:
        """Calculate how well quantum properties were preserved during scheduling."""
        if not tasks:
            return 1.0
            
        total_weight = sum(task.quantum_weight for task in tasks)
        avg_weight = total_weight / len(tasks)
        
        # Fidelity based on quantum weight preservation and measurement count
        weight_fidelity = min(avg_weight, 1.0)
        measurement_fidelity = 1.0 / (1.0 + sum(task.measurement_count for task in tasks) * 0.1)
        
        return (weight_fidelity + measurement_fidelity) / 2.0


class QuantumResourceAllocator:
    """Variational quantum-inspired resource allocation."""
    
    def __init__(self, num_resources: int = 4):
        """Initialize with quantum circuit depth (resource count)."""
        self.num_resources = num_resources
        self.random = random.Random(42)
    
    def variational_optimize(self, tasks: List[QuantumTask], max_iterations: int = 100) -> Dict[str, int]:
        """Use variational quantum eigensolvers concept for resource allocation."""
        allocation = {task.id: self.random.randint(0, self.num_resources - 1) for task in tasks}
        best_allocation = allocation.copy()
        best_cost = self._calculate_allocation_cost(tasks, allocation)
        
        # Variational parameter optimization
        for iteration in range(max_iterations):
            # Generate new allocation by rotating "quantum gates" (changing assignments)
            new_allocation = self._apply_quantum_rotation(allocation, iteration / max_iterations)
            cost = self._calculate_allocation_cost(tasks, new_allocation)
            
            if cost < best_cost:
                best_cost = cost
                best_allocation = new_allocation.copy()
                allocation = new_allocation
            
            # Adaptive step size (quantum circuit parameter update)
            if iteration % 10 == 0:
                logger.debug(f"Variational iteration {iteration}: cost = {cost:.3f}")
        
        return best_allocation
    
    def _apply_quantum_rotation(self, allocation: Dict[str, int], progress: float) -> Dict[str, int]:
        """Apply quantum rotation gates to change resource allocation."""
        new_allocation = allocation.copy()
        
        # Rotation angle decreases as we progress (like cooling in annealing)
        rotation_angle = math.pi * (1 - progress)
        num_rotations = max(1, int(len(allocation) * math.sin(rotation_angle)))
        
        tasks_to_rotate = self.random.sample(list(allocation.keys()), num_rotations)
        
        for task_id in tasks_to_rotate:
            # Quantum rotation: probabilistic resource change
            current_resource = allocation[task_id]
            
            # Create superposition of possible resources
            probabilities = [1.0] * self.num_resources
            probabilities[current_resource] *= 2.0  # Bias towards current
            
            # Normalize probabilities
            total_prob = sum(probabilities)
            probabilities = [p / total_prob for p in probabilities]
            
            # Sample from quantum distribution
            new_resource = self.random.choices(range(self.num_resources), weights=probabilities)[0]
            new_allocation[task_id] = new_resource
        
        return new_allocation
    
    def _calculate_allocation_cost(self, tasks: List[QuantumTask], allocation: Dict[str, int]) -> float:
        """Calculate cost of resource allocation (load balancing + affinity)."""
        # Resource load
        resource_loads = [0.0] * self.num_resources
        for task in tasks:
            resource_id = allocation.get(task.id, 0)
            resource_loads[resource_id] += task.effort
        
        # Load balancing cost (variance)
        avg_load = sum(resource_loads) / len(resource_loads)
        load_variance = sum((load - avg_load) ** 2 for load in resource_loads) / len(resource_loads)
        
        # Entanglement affinity cost (entangled tasks prefer same resource)
        affinity_cost = 0.0
        for task in tasks:
            task_resource = allocation.get(task.id, 0)
            for entangled_id in task.entangled_tasks:
                entangled_resource = allocation.get(entangled_id, 0)
                if task_resource != entangled_resource:
                    affinity_cost += 1.0
        
        return load_variance + 0.5 * affinity_cost