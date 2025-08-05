"""Validation and error handling for quantum-inspired task planning."""

from __future__ import annotations

import logging
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from .quantum_scheduler import QuantumTask, TaskState

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"


class ValidationIssueType(Enum):
    """Types of validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Represents a validation issue found during quantum task validation."""
    issue_type: ValidationIssueType
    code: str
    message: str
    task_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class QuantumTaskValidator:
    """Validator for quantum task planning configurations."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        """Initialize validator with specified strictness level."""
        self.validation_level = validation_level
        self.issues: List[ValidationIssue] = []
    
    def validate_tasks(self, tasks: List[QuantumTask]) -> List[ValidationIssue]:
        """Validate a list of quantum tasks and return issues found."""
        self.issues = []
        
        if not tasks:
            self._add_issue(
                ValidationIssueType.WARNING,
                "EMPTY_TASK_LIST",
                "No tasks provided for validation"
            )
            return self.issues
        
        # Validate individual tasks
        for task in tasks:
            self._validate_task(task)
        
        # Validate task relationships
        self._validate_task_dependencies(tasks)
        self._validate_task_uniqueness(tasks)
        self._validate_quantum_properties(tasks)
        
        logger.info(f"Validation completed: {len(self.issues)} issues found")
        return self.issues
    
    def _validate_task(self, task: QuantumTask) -> None:
        """Validate individual task properties."""
        # Validate required fields
        if not task.id:
            self._add_issue(
                ValidationIssueType.ERROR,
                "MISSING_TASK_ID",
                "Task ID is required",
                task_id=task.id
            )
        
        if not task.name:
            self._add_issue(
                ValidationIssueType.ERROR,
                "MISSING_TASK_NAME", 
                "Task name is required",
                task_id=task.id
            )
        
        # Validate numeric fields
        if task.priority < 0:
            self._add_issue(
                ValidationIssueType.ERROR,
                "INVALID_PRIORITY",
                f"Task priority must be non-negative, got {task.priority}",
                task_id=task.id
            )
        
        if task.effort <= 0:
            self._add_issue(
                ValidationIssueType.ERROR,
                "INVALID_EFFORT",
                f"Task effort must be positive, got {task.effort}",
                task_id=task.id
            )
        
        if task.value < 0:
            self._add_issue(
                ValidationIssueType.ERROR,
                "INVALID_VALUE",
                f"Task value must be non-negative, got {task.value}",
                task_id=task.id
            )
        
        # Validate quantum properties
        if task.quantum_weight <= 0:
            self._add_issue(
                ValidationIssueType.ERROR,
                "INVALID_QUANTUM_WEIGHT",
                f"Quantum weight must be positive, got {task.quantum_weight}",
                task_id=task.id
            )
        
        if task.coherence_time <= 0:
            self._add_issue(
                ValidationIssueType.ERROR,
                "INVALID_COHERENCE_TIME",
                f"Coherence time must be positive, got {task.coherence_time}",
                task_id=task.id
            )
        
        # Validate reasonable ranges (warnings)
        if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.MODERATE]:
            if task.priority > 10:
                self._add_issue(
                    ValidationIssueType.WARNING,
                    "HIGH_PRIORITY",
                    f"Task priority {task.priority} is unusually high",
                    task_id=task.id
                )
            
            if task.effort > 100:
                self._add_issue(
                    ValidationIssueType.WARNING,
                    "HIGH_EFFORT",
                    f"Task effort {task.effort} is unusually high",
                    task_id=task.id
                )
            
            if task.quantum_weight > 10:
                self._add_issue(
                    ValidationIssueType.WARNING,
                    "HIGH_QUANTUM_WEIGHT",
                    f"Quantum weight {task.quantum_weight} is unusually high",
                    task_id=task.id
                )
    
    def _validate_task_dependencies(self, tasks: List[QuantumTask]) -> None:
        """Validate task dependency relationships."""
        task_ids = {task.id for task in tasks}
        
        for task in tasks:
            # Check for non-existent dependencies
            for dep_id in task.dependencies:
                if dep_id not in task_ids:
                    self._add_issue(
                        ValidationIssueType.ERROR,
                        "MISSING_DEPENDENCY",
                        f"Task depends on non-existent task '{dep_id}'",
                        task_id=task.id,
                        details={"missing_dependency": dep_id}
                    )
            
            # Check for self-dependencies
            if task.id in task.dependencies:
                self._add_issue(
                    ValidationIssueType.ERROR,
                    "SELF_DEPENDENCY",
                    "Task cannot depend on itself",
                    task_id=task.id
                )
        
        # Check for circular dependencies
        cycles = self._detect_dependency_cycles(tasks)
        for cycle in cycles:
            self._add_issue(
                ValidationIssueType.ERROR,
                "CIRCULAR_DEPENDENCY",
                f"Circular dependency detected: {' -> '.join(cycle + [cycle[0]])}",
                details={"cycle": cycle}
            )
    
    def _validate_task_uniqueness(self, tasks: List[QuantumTask]) -> None:
        """Validate task ID uniqueness."""
        seen_ids = set()
        duplicates = set()
        
        for task in tasks:
            if task.id in seen_ids:
                duplicates.add(task.id)
            seen_ids.add(task.id)
        
        for duplicate_id in duplicates:
            self._add_issue(
                ValidationIssueType.ERROR,
                "DUPLICATE_TASK_ID",
                f"Duplicate task ID found: '{duplicate_id}'",
                task_id=duplicate_id
            )
    
    def _validate_quantum_properties(self, tasks: List[QuantumTask]) -> None:
        """Validate quantum-specific properties and relationships."""
        for task in tasks:
            # Validate entangled task references
            for entangled_id in task.entangled_tasks:
                if not any(t.id == entangled_id for t in tasks):
                    self._add_issue(
                        ValidationIssueType.WARNING,
                        "INVALID_ENTANGLEMENT",
                        f"Task is entangled with non-existent task '{entangled_id}'",
                        task_id=task.id,
                        details={"entangled_task": entangled_id}
                    )
            
            # Check for excessive entanglement
            if len(task.entangled_tasks) > len(tasks) * 0.5:
                self._add_issue(
                    ValidationIssueType.WARNING,
                    "EXCESSIVE_ENTANGLEMENT",
                    f"Task has excessive entanglements ({len(task.entangled_tasks)})",
                    task_id=task.id
                )
            
            # Validate quantum state consistency
            if task.state == TaskState.SUPERPOSITION and task.measurement_count > 0:
                self._add_issue(
                    ValidationIssueType.WARNING,
                    "MEASURED_SUPERPOSITION",
                    "Task in superposition state has been measured",
                    task_id=task.id
                )
    
    def _detect_dependency_cycles(self, tasks: List[QuantumTask]) -> List[List[str]]:
        """Detect circular dependencies using DFS."""
        graph = {task.id: task.dependencies for task in tasks}
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node: str, path: List[str]) -> None:
            if node in rec_stack:
                # Found cycle
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:])
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                dfs(neighbor, path + [neighbor])
            
            rec_stack.remove(node)
        
        for task_id in graph:
            if task_id not in visited:
                dfs(task_id, [task_id])
        
        return cycles
    
    def _add_issue(self, 
                   issue_type: ValidationIssueType, 
                   code: str, 
                   message: str, 
                   task_id: Optional[str] = None,
                   details: Optional[Dict[str, Any]] = None) -> None:
        """Add a validation issue to the list."""
        issue = ValidationIssue(
            issue_type=issue_type,
            code=code,
            message=message,
            task_id=task_id,
            details=details or {}
        )
        self.issues.append(issue)
        
        # Log based on issue type
        log_message = f"[{code}] {message}"
        if task_id:
            log_message += f" (Task: {task_id})"
        
        if issue_type == ValidationIssueType.ERROR:
            logger.error(log_message)
        elif issue_type == ValidationIssueType.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def get_error_count(self) -> int:
        """Get number of error-level issues."""
        return sum(1 for issue in self.issues if issue.issue_type == ValidationIssueType.ERROR)
    
    def get_warning_count(self) -> int:
        """Get number of warning-level issues."""
        return sum(1 for issue in self.issues if issue.issue_type == ValidationIssueType.WARNING)
    
    def has_errors(self) -> bool:
        """Check if any error-level issues were found."""
        return self.get_error_count() > 0
    
    def format_issues(self) -> str:
        """Format validation issues as human-readable string."""
        if not self.issues:
            return "✅ No validation issues found."
        
        lines = [f"Found {len(self.issues)} validation issues:"]
        
        errors = [i for i in self.issues if i.issue_type == ValidationIssueType.ERROR]
        warnings = [i for i in self.issues if i.issue_type == ValidationIssueType.WARNING]
        infos = [i for i in self.issues if i.issue_type == ValidationIssueType.INFO]
        
        if errors:
            lines.append(f"\n❌ ERRORS ({len(errors)}):")
            for issue in errors:
                lines.append(f"  • [{issue.code}] {issue.message}")
                if issue.task_id:
                    lines.append(f"    Task: {issue.task_id}")
        
        if warnings:
            lines.append(f"\n⚠️  WARNINGS ({len(warnings)}):")
            for issue in warnings:
                lines.append(f"  • [{issue.code}] {issue.message}")
                if issue.task_id:
                    lines.append(f"    Task: {issue.task_id}")
        
        if infos:
            lines.append(f"\nℹ️  INFO ({len(infos)}):")
            for issue in infos:
                lines.append(f"  • [{issue.code}] {issue.message}")
                if issue.task_id:
                    lines.append(f"    Task: {issue.task_id}")
        
        return "\n".join(lines)


class QuantumSecurityValidator:
    """Security validator for quantum task planning."""
    
    def __init__(self):
        """Initialize security validator."""
        self.security_issues: List[ValidationIssue] = []
    
    def validate_security(self, tasks: List[QuantumTask]) -> List[ValidationIssue]:
        """Validate security aspects of quantum tasks."""
        self.security_issues = []
        
        for task in tasks:
            self._check_task_security(task)
        
        logger.info(f"Security validation completed: {len(self.security_issues)} issues found")
        return self.security_issues
    
    def _check_task_security(self, task: QuantumTask) -> None:
        """Check security aspects of individual task."""
        # Check for potentially dangerous task names/IDs
        dangerous_patterns = [
            "../", "./", "~", "/etc/", "/proc/", "/sys/", 
            "<script>", "javascript:", "eval(", "exec(",
            "DROP TABLE", "DELETE FROM", "INSERT INTO"
        ]
        
        for pattern in dangerous_patterns:
            if pattern.lower() in task.id.lower() or pattern.lower() in task.name.lower():
                self._add_security_issue(
                    ValidationIssueType.ERROR,
                    "DANGEROUS_PATTERN",
                    f"Potentially dangerous pattern '{pattern}' found in task",
                    task_id=task.id
                )
        
        # Check for excessive resource consumption potential
        if task.effort > 1000:
            self._add_security_issue(
                ValidationIssueType.WARNING,
                "RESOURCE_EXHAUSTION",
                f"Task has very high effort ({task.effort}) - potential DoS risk",
                task_id=task.id
            )
        
        # Check quantum properties for security implications
        if task.quantum_weight > 100:
            self._add_security_issue(
                ValidationIssueType.WARNING,
                "EXCESSIVE_QUANTUM_WEIGHT",
                f"Unusually high quantum weight ({task.quantum_weight}) may indicate manipulation",
                task_id=task.id
            )
        
        # Check for suspicious entanglement patterns
        if len(task.entangled_tasks) > 50:
            self._add_security_issue(
                ValidationIssueType.WARNING,
                "EXCESSIVE_ENTANGLEMENT",
                f"Suspiciously high number of entanglements ({len(task.entangled_tasks)})",
                task_id=task.id
            )
    
    def _add_security_issue(self, 
                           issue_type: ValidationIssueType, 
                           code: str, 
                           message: str,
                           task_id: Optional[str] = None) -> None:
        """Add a security validation issue."""
        issue = ValidationIssue(
            issue_type=issue_type,
            code=f"SEC_{code}",
            message=f"Security: {message}",
            task_id=task_id
        )
        self.security_issues.append(issue)
        
        log_message = f"[SECURITY_{code}] {message}"
        if task_id:
            log_message += f" (Task: {task_id})"
        
        if issue_type == ValidationIssueType.ERROR:
            logger.error(log_message)
        else:
            logger.warning(log_message)


def validate_quantum_plan(tasks: List[QuantumTask], 
                         validation_level: ValidationLevel = ValidationLevel.MODERATE,
                         include_security: bool = True) -> Tuple[List[ValidationIssue], bool]:
    """Validate a complete quantum task plan.
    
    Returns:
        Tuple of (validation_issues, is_valid)
    """
    all_issues = []
    
    # Standard validation
    validator = QuantumTaskValidator(validation_level)
    issues = validator.validate_tasks(tasks)
    all_issues.extend(issues)
    
    # Security validation
    if include_security:
        security_validator = QuantumSecurityValidator()
        security_issues = security_validator.validate_security(tasks)
        all_issues.extend(security_issues)
    
    # Plan is valid if no errors found
    error_count = sum(1 for issue in all_issues if issue.issue_type == ValidationIssueType.ERROR)
    is_valid = error_count == 0
    
    logger.info(f"Plan validation complete: {len(all_issues)} issues, valid={is_valid}")
    
    return all_issues, is_valid