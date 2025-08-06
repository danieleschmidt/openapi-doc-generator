"""Quantum-inspired task planning integration with existing SDLC systems."""

from __future__ import annotations

import logging
import json
from typing import List, Dict, Optional, Any
from dataclasses import asdict
from pathlib import Path

from .quantum_scheduler import (
    QuantumInspiredScheduler,
    QuantumResourceAllocator, 
    QuantumTask,
    TaskState,
    QuantumScheduleResult
)
from .quantum_validator import (
    QuantumTaskValidator,
    ValidationLevel,
    ValidationIssue,
    validate_quantum_plan
)
from .quantum_monitor import (
    QuantumPlanningMonitor,
    monitor_operation,
    get_monitor
)
from .quantum_optimizer import (
    OptimizedQuantumPlanner,
    OptimizationConfig,
    AdaptiveQuantumScheduler,
    ParallelQuantumProcessor
)
from .quantum_security import (
    QuantumSecurityValidator,
    SecurityLevel,
    get_security_validator
)

logger = logging.getLogger(__name__)


class QuantumTaskPlanner:
    """Main interface for quantum-inspired task planning."""
    
    def __init__(self, 
                 temperature: float = 2.0,
                 cooling_rate: float = 0.95,
                 num_resources: int = 4,
                 validation_level: ValidationLevel = ValidationLevel.MODERATE,
                 enable_monitoring: bool = True,
                 enable_optimization: bool = True,
                 security_level: SecurityLevel = SecurityLevel.MEDIUM):
        """Initialize quantum task planner with security validation."""
        self.scheduler = QuantumInspiredScheduler(temperature, cooling_rate)
        self.allocator = QuantumResourceAllocator(num_resources)
        self.task_registry: Dict[str, QuantumTask] = {}
        self.validation_level = validation_level
        self.enable_monitoring = enable_monitoring
        self.validator = QuantumTaskValidator(validation_level)
        
        # Security validation
        self.security_validator = get_security_validator(security_level)
        self.security_level = security_level
        
        # Optimization configuration
        if enable_optimization:
            self.optimization_config = OptimizationConfig(
                enable_caching=True,
                enable_parallel_processing=True,
                enable_adaptive_scaling=True
            )
            self.optimized_planner = OptimizedQuantumPlanner(self.optimization_config)
        else:
            self.optimization_config = None
            self.optimized_planner = None
        
        if enable_monitoring:
            self.monitor = get_monitor()
        else:
            self.monitor = None
        
    def add_task(self, 
                 task_id: str,
                 name: str,
                 priority: float = 1.0,
                 effort: float = 1.0, 
                 value: float = 1.0,
                 dependencies: Optional[List[str]] = None,
                 coherence_time: float = 10.0) -> QuantumTask:
        """Add a task to the quantum planning system with security validation."""
        dependencies = dependencies or []
        
        # Sanitize inputs for security
        task_id = self.security_validator.sanitize_input(task_id)
        name = self.security_validator.sanitize_input(name)
        dependencies = self.security_validator.sanitize_input(dependencies)
        
        task = QuantumTask(
            id=task_id,
            name=name,
            priority=priority,
            effort=effort,
            value=value,
            dependencies=dependencies,
            coherence_time=coherence_time
        )
        
        # Security validation
        security_issues = self.security_validator.validate_task_security(task)
        critical_security = [i for i in security_issues if i.severity.name in ["CRITICAL", "HIGH"]]
        if critical_security:
            error_msg = f"Security validation failed: {'; '.join(i.message for i in critical_security)}"
            logger.error(error_msg)
            self.security_validator.audit_log_security_event("task_rejected", {
                "task_id": task_id,
                "reason": "security_validation_failed",
                "issues": [i.message for i in critical_security]
            })
            raise ValueError(error_msg)
        
        # Standard validation
        validation_issues = self.validator.validate_tasks([task])
        if validation_issues:
            error_issues = [i for i in validation_issues if i.issue_type.value == "error"]
            if error_issues:
                error_msg = f"Task validation failed: {'; '.join(i.message for i in error_issues)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                # Log warnings but continue
                for issue in validation_issues:
                    logger.warning(f"Task validation warning: {issue.message}")
        
        # Log security warnings
        for issue in security_issues:
            if issue.severity.name not in ["CRITICAL", "HIGH"]:
                logger.warning(f"Security warning for task {task_id}: {issue.message}")
        
        self.task_registry[task_id] = task
        logger.info(f"Added quantum task: {task_id} - {name}")
        
        # Audit log for task addition
        self.security_validator.audit_log_security_event("task_added", {
            "task_id": task_id,
            "name": name,
            "security_issues": len(security_issues)
        })
        
        return task
    
    @monitor_operation("create_quantum_plan")
    def create_quantum_plan(self) -> QuantumScheduleResult:
        """Create optimized quantum-inspired execution plan with security validation."""
        if not self.task_registry:
            logger.warning("No tasks registered for quantum planning")
            return QuantumScheduleResult([], 0.0, 0.0, 1.0, 0)
        
        tasks = list(self.task_registry.values())
        logger.info(f"Creating quantum plan for {len(tasks)} tasks")
        
        # Comprehensive security validation
        security_issues = self.security_validator.validate_plan_security(tasks)
        critical_security = [i for i in security_issues if i.severity.name in ["CRITICAL", "HIGH"]]
        if critical_security:
            error_msg = f"Security validation failed: {'; '.join(i.message for i in critical_security)}"
            logger.error(error_msg)
            self.security_validator.audit_log_security_event("plan_rejected", {
                "task_count": len(tasks),
                "reason": "security_validation_failed",
                "critical_issues": len(critical_security)
            })
            raise ValueError(error_msg)
        
        # Log security warnings
        for issue in security_issues:
            if issue.severity.name not in ["CRITICAL", "HIGH"]:
                logger.warning(f"Security warning: {issue.message}")
        
        # Standard task plan validation
        validation_issues, is_valid = validate_quantum_plan(
            tasks, 
            self.validation_level, 
            include_security=True
        )
        
        if not is_valid:
            error_messages = [i.message for i in validation_issues if i.issue_type.value == "error"]
            error_msg = f"Plan validation failed: {'; '.join(error_messages)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Log validation warnings
        warnings = [i for i in validation_issues if i.issue_type.value == "warning"]
        for warning in warnings:
            logger.warning(f"Plan validation warning: {warning.message}")
        
        # Run quantum annealing optimization (optimized if available)
        try:
            if self.monitor:
                optimization_op = self.monitor.start_operation("quantum_annealing", {"task_count": len(tasks)})
            
            # Use optimized planner if available
            if self.optimized_planner:
                result = self.optimized_planner.create_optimized_plan(tasks)
            else:
                result = self.scheduler.quantum_annealing_schedule(tasks)
            
            if self.monitor:
                self.monitor.end_operation(optimization_op, result)
        except Exception as e:
            if self.monitor:
                self.monitor.end_operation(optimization_op, error=str(e))
            raise
        
        # Allocate resources using variational optimization
        try:
            if self.monitor:
                allocation_op = self.monitor.start_operation("resource_allocation", {"task_count": len(result.optimized_tasks)})
            
            resource_allocation = self.allocator.variational_optimize(result.optimized_tasks)
            
            if self.monitor:
                self.monitor.end_operation(allocation_op)
        except Exception as e:
            if self.monitor:
                self.monitor.end_operation(allocation_op, error=str(e))
            raise
        
        # Update tasks with resource allocation
        for task in result.optimized_tasks:
            if task.id in resource_allocation:
                # Store resource allocation as custom attribute
                setattr(task, 'allocated_resource', resource_allocation[task.id])
        
        logger.info(f"Quantum plan created: {len(result.optimized_tasks)} tasks, "
                   f"fidelity={result.quantum_fidelity:.3f}, "
                   f"value={result.total_value:.2f}")
        
        return result
    
    def export_plan_to_json(self, result: QuantumScheduleResult, output_path: Path) -> None:
        """Export quantum plan to JSON format."""
        plan_data = {
            "quantum_schedule": {
                "total_value": result.total_value,
                "execution_time": result.execution_time,
                "quantum_fidelity": result.quantum_fidelity,
                "convergence_iterations": result.convergence_iterations,
                "tasks": []
            }
        }
        
        for i, task in enumerate(result.optimized_tasks):
            task_data = asdict(task)
            task_data["execution_order"] = i + 1
            task_data["allocated_resource"] = getattr(task, 'allocated_resource', 0)
            # Convert sets to lists for JSON serialization
            task_data["entangled_tasks"] = list(task_data["entangled_tasks"])
            task_data["state"] = task_data["state"].value if hasattr(task_data["state"], 'value') else str(task_data["state"])
            plan_data["quantum_schedule"]["tasks"].append(task_data)
        
        with open(output_path, 'w') as f:
            json.dump(plan_data, f, indent=2)
        
        logger.info(f"Quantum plan exported to {output_path}")
    
    def import_classical_tasks(self, classical_tasks: List[Dict[str, Any]]) -> List[QuantumTask]:
        """Convert classical task format to quantum tasks."""
        quantum_tasks = []
        
        for task_data in classical_tasks:
            # Map classical task fields to quantum task fields
            task_id = task_data.get("id", f"task_{len(quantum_tasks)}")
            name = task_data.get("name", task_data.get("title", "Unnamed Task"))
            
            # Convert classical priority/scoring to quantum properties
            priority = task_data.get("priority", 1.0)
            effort = task_data.get("effort", task_data.get("story_points", 1.0))
            value = task_data.get("value", task_data.get("business_value", 1.0))
            
            # Extract dependencies
            dependencies = task_data.get("dependencies", [])
            if "blockers" in task_data:
                dependencies.extend(task_data["blockers"])
            
            # Calculate coherence time based on task urgency
            urgency = task_data.get("urgency", 1.0)
            coherence_time = max(1.0, 20.0 / urgency)  # More urgent = shorter coherence
            
            quantum_task = self.add_task(
                task_id=task_id,
                name=name,
                priority=priority,
                effort=effort,
                value=value,
                dependencies=dependencies,
                coherence_time=coherence_time
            )
            
            quantum_tasks.append(quantum_task)
        
        logger.info(f"Imported {len(quantum_tasks)} classical tasks to quantum format")
        return quantum_tasks
    
    def get_task_quantum_metrics(self, task_id: str) -> Optional[Dict[str, float]]:
        """Get quantum-specific metrics for a task."""
        if task_id not in self.task_registry:
            return None
        
        task = self.task_registry[task_id]
        return {
            "quantum_weight": task.quantum_weight,
            "coherence_time": task.coherence_time,
            "measurement_count": task.measurement_count,
            "entanglement_degree": len(task.entangled_tasks),
            "quantum_priority": self.scheduler.quantum_priority_score(task, task.created_at)
        }
    
    def simulate_execution(self, result: QuantumScheduleResult) -> Dict[str, Any]:
        """Simulate quantum plan execution and measure performance."""
        simulation_results = {
            "total_tasks": len(result.optimized_tasks),
            "estimated_completion_time": 0.0,
            "resource_utilization": {},
            "quantum_effects": {
                "superposition_collapses": 0,
                "entanglement_breaks": 0,
                "coherence_loss": 0.0
            }
        }
        
        current_time = 0.0
        resource_busy_until = {}
        
        for task in result.optimized_tasks:
            if task.state == TaskState.SUPERPOSITION:
                simulation_results["quantum_effects"]["superposition_collapses"] += 1
                continue
            
            # Get allocated resource
            resource_id = getattr(task, 'allocated_resource', 0)
            
            # Wait for resource availability
            resource_key = f"resource_{resource_id}"
            if resource_key in resource_busy_until:
                current_time = max(current_time, resource_busy_until[resource_key])
            
            # Execute task
            execution_time = task.effort
            task_end_time = current_time + execution_time
            resource_busy_until[resource_key] = task_end_time
            
            # Track resource utilization
            if resource_key not in simulation_results["resource_utilization"]:
                simulation_results["resource_utilization"][resource_key] = 0.0
            simulation_results["resource_utilization"][resource_key] += execution_time
            
            # Check for quantum effects
            if current_time > task.coherence_time:
                simulation_results["quantum_effects"]["coherence_loss"] += 1.0
            
            if task.measurement_count > 0:
                simulation_results["quantum_effects"]["entanglement_breaks"] += task.measurement_count
        
        simulation_results["estimated_completion_time"] = max(resource_busy_until.values()) if resource_busy_until else 0.0
        
        logger.info(f"Simulation completed: {simulation_results['estimated_completion_time']:.2f} time units")
        return simulation_results
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics including security metrics."""
        stats = {
            "configuration": {
                "validation_level": self.validation_level.value,
                "security_level": self.security_level.value,
                "monitoring_enabled": self.enable_monitoring,
                "optimization_enabled": self.optimized_planner is not None
            },
            "task_registry": {
                "total_tasks": len(self.task_registry),
                "task_types": self._analyze_task_types()
            }
        }
        
        # Security statistics
        if self.task_registry:
            security_report = self.security_validator.generate_security_report(list(self.task_registry.values()))
            stats["security"] = {
                "security_score": security_report["security_score"],
                "compliance_status": security_report["compliance_status"],
                "total_security_issues": security_report["total_issues"],
                "issues_by_severity": security_report["issues_by_severity"],
                "recommendations": security_report["recommendations"][:3]  # Top 3 recommendations
            }
        
        # Monitoring statistics
        if self.monitor:
            stats["monitoring"] = self.monitor.get_metrics_summary()
            stats["health"] = {
                "system_status": self.monitor.get_system_status().value,
                "health_checks": [
                    {"component": r.component, "status": r.status.value, "message": r.message}
                    for r in self.monitor.run_health_checks()
                ]
            }
        
        # Optimization statistics
        if self.optimized_planner:
            stats["optimization"] = self.optimized_planner.get_performance_stats()
        
        return stats
    
    def get_security_report(self) -> Dict[str, Any]:
        """Get detailed security validation report."""
        if not self.task_registry:
            return {
                "status": "no_tasks",
                "message": "No tasks to validate"
            }
        
        tasks = list(self.task_registry.values())
        return self.security_validator.generate_security_report(tasks)
    
    def _analyze_task_types(self) -> Dict[str, Any]:
        """Analyze task registry for insights."""
        if not self.task_registry:
            return {"message": "No tasks in registry"}
        
        tasks = list(self.task_registry.values())
        
        return {
            "priority_distribution": {
                "high": len([t for t in tasks if t.priority >= 4.0]),
                "medium": len([t for t in tasks if 2.0 <= t.priority < 4.0]),
                "low": len([t for t in tasks if t.priority < 2.0])
            },
            "effort_stats": {
                "avg_effort": sum(t.effort for t in tasks) / len(tasks),
                "max_effort": max(t.effort for t in tasks),
                "min_effort": min(t.effort for t in tasks)
            },
            "dependency_stats": {
                "tasks_with_dependencies": len([t for t in tasks if t.dependencies]),
                "avg_dependencies": sum(len(t.dependencies) for t in tasks) / len(tasks),
                "max_dependencies": max(len(t.dependencies) for t in tasks)
            },
            "quantum_stats": {
                "avg_quantum_weight": sum(t.quantum_weight for t in tasks) / len(tasks),
                "avg_coherence_time": sum(t.coherence_time for t in tasks) / len(tasks),
                "total_entanglements": sum(len(t.entangled_tasks) for t in tasks) // 2  # Divide by 2 since each entanglement is counted twice
            }
        }
    
    def tune_performance(self, target_fidelity: float = 0.8, target_duration_ms: float = 5000) -> None:
        """Auto-tune performance parameters."""
        if self.optimized_planner:
            self.optimized_planner.tune_performance(target_fidelity, target_duration_ms)
        else:
            logger.warning("Performance tuning requires optimization to be enabled")
    
    def clear_caches(self) -> None:
        """Clear all performance caches."""
        if self.optimized_planner:
            self.optimized_planner.clear_cache()
            logger.info("Performance caches cleared")
        else:
            logger.info("No caches to clear (optimization not enabled)")


def integrate_with_existing_sdlc(planner: QuantumTaskPlanner) -> None:
    """Integrate quantum planner with existing SDLC automation."""
    logger.info("Integrating quantum planner with existing SDLC systems")
    
    # Add common SDLC tasks with quantum properties
    sdlc_tasks = [
        {
            "id": "requirements_analysis",
            "name": "Requirements Analysis & Documentation", 
            "priority": 3.0,
            "effort": 2.0,
            "value": 5.0,
            "coherence_time": 15.0
        },
        {
            "id": "architecture_design",
            "name": "System Architecture Design",
            "priority": 4.0,
            "effort": 3.0, 
            "value": 8.0,
            "dependencies": ["requirements_analysis"],
            "coherence_time": 20.0
        },
        {
            "id": "core_implementation",
            "name": "Core Feature Implementation",
            "priority": 5.0,
            "effort": 8.0,
            "value": 10.0,
            "dependencies": ["architecture_design"],
            "coherence_time": 25.0
        },
        {
            "id": "testing_framework",
            "name": "Testing Framework Setup",
            "priority": 3.5,
            "effort": 2.5,
            "value": 6.0,
            "dependencies": ["architecture_design"],
            "coherence_time": 12.0
        },
        {
            "id": "security_audit",
            "name": "Security Audit & Vulnerability Assessment",
            "priority": 4.5,
            "effort": 3.0,
            "value": 9.0,
            "dependencies": ["core_implementation"],
            "coherence_time": 8.0  # Security issues need quick attention
        },
        {
            "id": "performance_optimization",
            "name": "Performance Optimization & Benchmarking",
            "priority": 3.0,
            "effort": 4.0,
            "value": 7.0,
            "dependencies": ["core_implementation", "testing_framework"],
            "coherence_time": 18.0
        },
        {
            "id": "documentation_generation",
            "name": "Automated Documentation Generation",
            "priority": 2.5,
            "effort": 2.0,
            "value": 4.0,
            "dependencies": ["core_implementation"],
            "coherence_time": 30.0  # Documentation can wait longer
        },
        {
            "id": "deployment_automation",
            "name": "CI/CD Pipeline & Deployment Automation",
            "priority": 4.0,
            "effort": 3.5,
            "value": 8.5,
            "dependencies": ["testing_framework", "security_audit"],
            "coherence_time": 10.0
        }
    ]
    
    # Add tasks to quantum planner
    for task_data in sdlc_tasks:
        planner.add_task(**task_data)
    
    logger.info(f"Added {len(sdlc_tasks)} SDLC tasks to quantum planner")