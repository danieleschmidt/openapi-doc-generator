"""REST API endpoints for quantum task planning integration."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from .quantum_planner import QuantumTaskPlanner, integrate_with_existing_sdlc
from .quantum_validator import ValidationLevel

logger = logging.getLogger(__name__)


class QuantumPlannerAPI:
    """REST API interface for quantum task planning."""

    def __init__(self):
        """Initialize the quantum planner API."""
        self.planners: Dict[str, QuantumTaskPlanner] = {}
        self.active_session = None

    def create_session(self,
                      session_id: str,
                      temperature: float = 2.0,
                      cooling_rate: float = 0.95,
                      num_resources: int = 4,
                      validation_level: str = "moderate",
                      enable_monitoring: bool = True) -> Dict[str, Any]:
        """Create a new quantum planning session."""
        try:
            # Map validation level
            validation_map = {
                "strict": ValidationLevel.STRICT,
                "moderate": ValidationLevel.MODERATE,
                "lenient": ValidationLevel.LENIENT
            }

            planner = QuantumTaskPlanner(
                temperature=temperature,
                cooling_rate=cooling_rate,
                num_resources=num_resources,
                validation_level=validation_map.get(validation_level, ValidationLevel.MODERATE),
                enable_monitoring=enable_monitoring,
                enable_optimization=True
            )

            self.planners[session_id] = planner
            self.active_session = session_id

            logger.info(f"Created quantum planning session: {session_id}")

            return {
                "status": "success",
                "session_id": session_id,
                "configuration": {
                    "temperature": temperature,
                    "cooling_rate": cooling_rate,
                    "num_resources": num_resources,
                    "validation_level": validation_level,
                    "monitoring_enabled": enable_monitoring
                },
                "message": "Quantum planning session created successfully"
            }

        except Exception as e:
            logger.error(f"Failed to create session {session_id}: {str(e)}")
            return {
                "status": "error",
                "message": f"Session creation failed: {str(e)}"
            }

    def add_task(self,
                session_id: str,
                task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add a task to a quantum planning session."""
        try:
            if session_id not in self.planners:
                return {
                    "status": "error",
                    "message": f"Session {session_id} not found"
                }

            planner = self.planners[session_id]

            # Extract task parameters with defaults
            task_id = task_data.get("id", f"task_{len(planner.task_registry)}")
            name = task_data.get("name", "Unnamed Task")
            priority = float(task_data.get("priority", 1.0))
            effort = float(task_data.get("effort", 1.0))
            value = float(task_data.get("value", 1.0))
            dependencies = task_data.get("dependencies", [])
            coherence_time = float(task_data.get("coherence_time", 10.0))

            # Add task to planner
            task = planner.add_task(
                task_id=task_id,
                name=name,
                priority=priority,
                effort=effort,
                value=value,
                dependencies=dependencies,
                coherence_time=coherence_time
            )

            return {
                "status": "success",
                "task_id": task_id,
                "message": f"Task '{name}' added successfully",
                "task": {
                    "id": task.id,
                    "name": task.name,
                    "priority": task.priority,
                    "effort": task.effort,
                    "value": task.value,
                    "dependencies": task.dependencies,
                    "quantum_weight": task.quantum_weight,
                    "coherence_time": task.coherence_time
                }
            }

        except Exception as e:
            logger.error(f"Failed to add task to session {session_id}: {str(e)}")
            return {
                "status": "error",
                "message": f"Task addition failed: {str(e)}"
            }

    def add_sdlc_tasks(self, session_id: str) -> Dict[str, Any]:
        """Add standard SDLC tasks to a session."""
        try:
            if session_id not in self.planners:
                return {
                    "status": "error",
                    "message": f"Session {session_id} not found"
                }

            planner = self.planners[session_id]
            initial_count = len(planner.task_registry)

            # Integrate standard SDLC tasks
            integrate_with_existing_sdlc(planner)

            final_count = len(planner.task_registry)
            added_count = final_count - initial_count

            return {
                "status": "success",
                "message": f"Added {added_count} SDLC tasks to session",
                "tasks_added": added_count,
                "total_tasks": final_count
            }

        except Exception as e:
            logger.error(f"Failed to add SDLC tasks to session {session_id}: {str(e)}")
            return {
                "status": "error",
                "message": f"SDLC task integration failed: {str(e)}"
            }

    def create_plan(self, session_id: str) -> Dict[str, Any]:
        """Create a quantum-optimized execution plan."""
        try:
            if session_id not in self.planners:
                return {
                    "status": "error",
                    "message": f"Session {session_id} not found"
                }

            planner = self.planners[session_id]

            if not planner.task_registry:
                return {
                    "status": "error",
                    "message": "No tasks found in session. Add tasks before creating a plan."
                }

            # Create quantum plan
            result = planner.create_quantum_plan()

            # Convert tasks to serializable format
            optimized_tasks = []
            for i, task in enumerate(result.optimized_tasks):
                task_dict = asdict(task)
                task_dict["execution_order"] = i + 1
                task_dict["allocated_resource"] = getattr(task, 'allocated_resource', 0)
                task_dict["entangled_tasks"] = list(task_dict["entangled_tasks"])
                task_dict["state"] = task_dict["state"].value
                optimized_tasks.append(task_dict)

            # Run simulation
            simulation = planner.simulate_execution(result)

            return {
                "status": "success",
                "session_id": session_id,
                "quantum_plan": {
                    "total_value": result.total_value,
                    "execution_time": result.execution_time,
                    "quantum_fidelity": result.quantum_fidelity,
                    "convergence_iterations": result.convergence_iterations,
                    "optimized_tasks": optimized_tasks
                },
                "simulation": simulation,
                "performance": planner.get_performance_statistics()
            }

        except Exception as e:
            logger.error(f"Failed to create plan for session {session_id}: {str(e)}")
            return {
                "status": "error",
                "message": f"Plan creation failed: {str(e)}"
            }

    def export_plan(self,
                   session_id: str,
                   format: str = "json",
                   output_path: Optional[str] = None) -> Dict[str, Any]:
        """Export quantum plan in specified format."""
        try:
            if session_id not in self.planners:
                return {
                    "status": "error",
                    "message": f"Session {session_id} not found"
                }

            planner = self.planners[session_id]
            plan_result = self.create_plan(session_id)

            if plan_result["status"] != "success":
                return plan_result

            if format == "json":
                if output_path:
                    export_path = Path(output_path)
                    with open(export_path, 'w') as f:
                        json.dump(plan_result, f, indent=2)

                    return {
                        "status": "success",
                        "message": f"Plan exported to {export_path}",
                        "export_path": str(export_path),
                        "format": "json"
                    }
                else:
                    return {
                        "status": "success",
                        "data": plan_result,
                        "format": "json"
                    }

            elif format == "markdown":
                # Generate markdown report
                quantum_plan = plan_result["quantum_plan"]
                simulation = plan_result["simulation"]

                markdown_lines = [
                    f"# Quantum Task Plan - Session {session_id}",
                    "",
                    f"**Quantum Fidelity**: {quantum_plan['quantum_fidelity']:.3f}",
                    f"**Total Business Value**: {quantum_plan['total_value']:.2f}",
                    f"**Estimated Completion**: {simulation['estimated_completion_time']:.2f} time units",
                    "",
                    "## Optimized Task Schedule",
                    ""
                ]

                for task in quantum_plan["optimized_tasks"]:
                    markdown_lines.extend([
                        f"### {task['execution_order']}. {task['name']}",
                        f"- **Priority**: {task['priority']:.2f}",
                        f"- **Effort**: {task['effort']:.1f}",
                        f"- **Value**: {task['value']:.1f}",
                        f"- **Resource**: Resource-{task['allocated_resource']}",
                        f"- **Dependencies**: {', '.join(task['dependencies']) if task['dependencies'] else 'None'}",
                        ""
                    ])

                markdown_content = "\n".join(markdown_lines)

                if output_path:
                    export_path = Path(output_path)
                    export_path.write_text(markdown_content, encoding='utf-8')

                    return {
                        "status": "success",
                        "message": f"Markdown plan exported to {export_path}",
                        "export_path": str(export_path),
                        "format": "markdown"
                    }
                else:
                    return {
                        "status": "success",
                        "data": markdown_content,
                        "format": "markdown"
                    }

            else:
                return {
                    "status": "error",
                    "message": f"Unsupported export format: {format}"
                }

        except Exception as e:
            logger.error(f"Failed to export plan for session {session_id}: {str(e)}")
            return {
                "status": "error",
                "message": f"Plan export failed: {str(e)}"
            }

    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get status and metrics for a quantum planning session."""
        try:
            if session_id not in self.planners:
                return {
                    "status": "error",
                    "message": f"Session {session_id} not found"
                }

            planner = self.planners[session_id]
            stats = planner.get_performance_statistics()

            return {
                "status": "success",
                "session_id": session_id,
                "task_count": len(planner.task_registry),
                "configuration": stats["configuration"],
                "performance": stats
            }

        except Exception as e:
            logger.error(f"Failed to get status for session {session_id}: {str(e)}")
            return {
                "status": "error",
                "message": f"Status retrieval failed: {str(e)}"
            }

    def list_sessions(self) -> Dict[str, Any]:
        """List all active quantum planning sessions."""
        try:
            sessions = []

            for session_id, planner in self.planners.items():
                sessions.append({
                    "session_id": session_id,
                    "task_count": len(planner.task_registry),
                    "is_active": session_id == self.active_session
                })

            return {
                "status": "success",
                "sessions": sessions,
                "total_sessions": len(sessions),
                "active_session": self.active_session
            }

        except Exception as e:
            logger.error(f"Failed to list sessions: {str(e)}")
            return {
                "status": "error",
                "message": f"Session listing failed: {str(e)}"
            }

    def delete_session(self, session_id: str) -> Dict[str, Any]:
        """Delete a quantum planning session."""
        try:
            if session_id not in self.planners:
                return {
                    "status": "error",
                    "message": f"Session {session_id} not found"
                }

            del self.planners[session_id]

            if self.active_session == session_id:
                self.active_session = None
                if self.planners:
                    # Set another session as active
                    self.active_session = next(iter(self.planners.keys()))

            logger.info(f"Deleted quantum planning session: {session_id}")

            return {
                "status": "success",
                "message": f"Session {session_id} deleted successfully"
            }

        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {str(e)}")
            return {
                "status": "error",
                "message": f"Session deletion failed: {str(e)}"
            }


# Global API instance
quantum_api = QuantumPlannerAPI()


def get_quantum_api() -> QuantumPlannerAPI:
    """Get the global quantum planner API instance."""
    return quantum_api
