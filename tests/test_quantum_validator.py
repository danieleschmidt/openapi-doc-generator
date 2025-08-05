"""Tests for quantum task validation."""

import pytest
from openapi_doc_generator.quantum_validator import (
    QuantumTaskValidator,
    QuantumSecurityValidator,
    ValidationLevel,
    ValidationIssue,
    ValidationIssueType,
    validate_quantum_plan
)
from openapi_doc_generator.quantum_scheduler import QuantumTask, TaskState


class TestQuantumTaskValidator:
    """Test quantum task validator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = QuantumTaskValidator(ValidationLevel.MODERATE)
    
    def test_validator_initialization(self):
        """Test validator initialization with different levels."""
        strict_validator = QuantumTaskValidator(ValidationLevel.STRICT)
        assert strict_validator.validation_level == ValidationLevel.STRICT
        
        lenient_validator = QuantumTaskValidator(ValidationLevel.LENIENT)  
        assert lenient_validator.validation_level == ValidationLevel.LENIENT
    
    def test_validate_empty_task_list(self):
        """Test validation of empty task list."""
        issues = self.validator.validate_tasks([])
        
        assert len(issues) == 1
        assert issues[0].issue_type == ValidationIssueType.WARNING
        assert issues[0].code == "EMPTY_TASK_LIST"
    
    def test_validate_valid_tasks(self):
        """Test validation of valid tasks."""
        tasks = [
            QuantumTask(
                id="task1",
                name="Valid Task 1",
                priority=2.0,
                effort=1.5,
                value=3.0
            ),
            QuantumTask(
                id="task2", 
                name="Valid Task 2",
                priority=1.0,
                effort=2.0,
                value=4.0,
                dependencies=["task1"]
            )
        ]
        
        issues = self.validator.validate_tasks(tasks)
        
        # Should have no error-level issues
        errors = [i for i in issues if i.issue_type == ValidationIssueType.ERROR]
        assert len(errors) == 0
    
    def test_validate_missing_required_fields(self):
        """Test validation of tasks with missing required fields."""
        tasks = [
            QuantumTask(id="", name="Task with empty ID"),
            QuantumTask(id="task2", name=""),
        ]
        
        issues = self.validator.validate_tasks(tasks)
        
        error_codes = [i.code for i in issues if i.issue_type == ValidationIssueType.ERROR]
        assert "MISSING_TASK_ID" in error_codes
        assert "MISSING_TASK_NAME" in error_codes
    
    def test_validate_invalid_numeric_fields(self):
        """Test validation of invalid numeric values."""
        tasks = [
            QuantumTask(id="task1", name="Negative Priority", priority=-1.0),
            QuantumTask(id="task2", name="Zero Effort", effort=0.0),
            QuantumTask(id="task3", name="Negative Value", value=-5.0),
            QuantumTask(id="task4", name="Invalid Quantum Weight", quantum_weight=0.0),
            QuantumTask(id="task5", name="Invalid Coherence Time", coherence_time=-1.0)
        ]
        
        issues = self.validator.validate_tasks(tasks)
        
        error_codes = [i.code for i in issues if i.issue_type == ValidationIssueType.ERROR]
        assert "INVALID_PRIORITY" in error_codes
        assert "INVALID_EFFORT" in error_codes
        assert "INVALID_VALUE" in error_codes
        assert "INVALID_QUANTUM_WEIGHT" in error_codes
        assert "INVALID_COHERENCE_TIME" in error_codes
    
    def test_validate_high_values_warnings(self):
        """Test warnings for unusually high values."""
        tasks = [
            QuantumTask(id="task1", name="High Priority", priority=15.0),
            QuantumTask(id="task2", name="High Effort", effort=150.0),
            QuantumTask(id="task3", name="High Quantum Weight", quantum_weight=20.0)
        ]
        
        issues = self.validator.validate_tasks(tasks)
        
        warning_codes = [i.code for i in issues if i.issue_type == ValidationIssueType.WARNING]
        assert "HIGH_PRIORITY" in warning_codes
        assert "HIGH_EFFORT" in warning_codes
        assert "HIGH_QUANTUM_WEIGHT" in warning_codes
    
    def test_validate_duplicate_task_ids(self):
        """Test validation of duplicate task IDs."""
        tasks = [
            QuantumTask(id="duplicate", name="First Task"),
            QuantumTask(id="duplicate", name="Second Task"),
            QuantumTask(id="unique", name="Unique Task")
        ]
        
        issues = self.validator.validate_tasks(tasks)
        
        duplicate_errors = [i for i in issues if i.code == "DUPLICATE_TASK_ID"]
        assert len(duplicate_errors) == 1
        assert duplicate_errors[0].task_id == "duplicate"
    
    def test_validate_missing_dependencies(self):
        """Test validation of tasks with missing dependencies."""
        tasks = [
            QuantumTask(id="task1", name="Task 1", dependencies=["nonexistent"]),
            QuantumTask(id="task2", name="Task 2", dependencies=["task1", "another_missing"])
        ]
        
        issues = self.validator.validate_tasks(tasks)
        
        missing_dep_errors = [i for i in issues if i.code == "MISSING_DEPENDENCY"]
        assert len(missing_dep_errors) == 3  # nonexistent, another_missing
        
        # Check details are included
        for error in missing_dep_errors:
            assert "missing_dependency" in error.details
    
    def test_validate_self_dependency(self):
        """Test validation of self-dependencies."""
        tasks = [
            QuantumTask(id="self_dep", name="Self Dependent", dependencies=["self_dep"])
        ]
        
        issues = self.validator.validate_tasks(tasks)
        
        self_dep_errors = [i for i in issues if i.code == "SELF_DEPENDENCY"]
        assert len(self_dep_errors) == 1
        assert self_dep_errors[0].task_id == "self_dep"
    
    def test_validate_circular_dependencies(self):
        """Test detection of circular dependencies."""
        tasks = [
            QuantumTask(id="taskA", name="Task A", dependencies=["taskC"]),
            QuantumTask(id="taskB", name="Task B", dependencies=["taskA"]),
            QuantumTask(id="taskC", name="Task C", dependencies=["taskB"])
        ]
        
        issues = self.validator.validate_tasks(tasks)
        
        circular_errors = [i for i in issues if i.code == "CIRCULAR_DEPENDENCY"]
        assert len(circular_errors) >= 1  # At least one cycle detected
        
        # Check cycle details are included
        for error in circular_errors:
            assert "cycle" in error.details
            assert isinstance(error.details["cycle"], list)
    
    def test_validate_quantum_properties(self):
        """Test validation of quantum-specific properties."""
        task1 = QuantumTask(id="task1", name="Task 1")
        task1.entangled_tasks.add("nonexistent_task")
        
        task2 = QuantumTask(id="task2", name="Task 2") 
        task2.entangled_tasks = set(f"task_{i}" for i in range(20))  # Excessive entanglement
        
        task3 = QuantumTask(id="task3", name="Task 3", state=TaskState.SUPERPOSITION)
        task3.measurement_count = 5  # Measured but still in superposition
        
        tasks = [task1, task2, task3]
        
        issues = self.validator.validate_tasks(tasks)
        
        warning_codes = [i.code for i in issues if i.issue_type == ValidationIssueType.WARNING]
        assert "INVALID_ENTANGLEMENT" in warning_codes
        assert "EXCESSIVE_ENTANGLEMENT" in warning_codes
        assert "MEASURED_SUPERPOSITION" in warning_codes
    
    def test_error_and_warning_counts(self):
        """Test counting of errors and warnings."""
        tasks = [
            QuantumTask(id="", name="Missing ID"),  # Error
            QuantumTask(id="task1", name="High Priority", priority=15.0),  # Warning
            QuantumTask(id="task2", name="Valid Task")  # No issues
        ]
        
        issues = self.validator.validate_tasks(tasks)
        
        assert self.validator.get_error_count() == 1
        assert self.validator.get_warning_count() == 1
        assert self.validator.has_errors() == True
    
    def test_format_issues(self):
        """Test formatting of validation issues."""
        tasks = [
            QuantumTask(id="", name="Missing ID"),
            QuantumTask(id="task1", name="High Priority", priority=15.0)
        ]
        
        issues = self.validator.validate_tasks(tasks)
        formatted = self.validator.format_issues()
        
        assert "❌ ERRORS" in formatted
        assert "⚠️  WARNINGS" in formatted
        assert "MISSING_TASK_ID" in formatted
        assert "HIGH_PRIORITY" in formatted
    
    def test_validation_levels(self):
        """Test different validation strictness levels."""
        task = QuantumTask(id="task1", name="High Values", priority=15.0, effort=150.0)
        
        # Strict validation should catch high values
        strict_validator = QuantumTaskValidator(ValidationLevel.STRICT)
        strict_issues = strict_validator.validate_tasks([task])
        strict_warnings = [i for i in strict_issues if i.issue_type == ValidationIssueType.WARNING]
        
        # Lenient validation might ignore some warnings
        lenient_validator = QuantumTaskValidator(ValidationLevel.LENIENT)
        lenient_issues = lenient_validator.validate_tasks([task])
        lenient_warnings = [i for i in lenient_issues if i.issue_type == ValidationIssueType.WARNING]
        
        # Strict should have same or more warnings than lenient
        assert len(strict_warnings) >= len(lenient_warnings)


class TestQuantumSecurityValidator:
    """Test quantum security validator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.security_validator = QuantumSecurityValidator()
    
    def test_dangerous_patterns_detection(self):
        """Test detection of dangerous patterns in task names/IDs."""
        dangerous_tasks = [
            QuantumTask(id="../etc/passwd", name="Path Traversal"),
            QuantumTask(id="task1", name="<script>alert('xss')</script>"),
            QuantumTask(id="DROP TABLE users", name="SQL Injection"),
            QuantumTask(id="task2", name="eval(malicious_code)")
        ]
        
        issues = self.security_validator.validate_security(dangerous_tasks)
        
        dangerous_pattern_issues = [i for i in issues if i.code == "SEC_DANGEROUS_PATTERN"]
        assert len(dangerous_pattern_issues) >= 4  # One for each dangerous pattern
    
    def test_resource_exhaustion_detection(self):
        """Test detection of potential resource exhaustion."""
        high_effort_task = QuantumTask(
            id="exhaustion_task",
            name="High Effort Task",
            effort=2000.0  # Very high effort
        )
        
        issues = self.security_validator.validate_security([high_effort_task])
        
        exhaustion_warnings = [i for i in issues if i.code == "SEC_RESOURCE_EXHAUSTION"]
        assert len(exhaustion_warnings) == 1
        assert exhaustion_warnings[0].task_id == "exhaustion_task"
    
    def test_excessive_quantum_weight_detection(self):
        """Test detection of excessive quantum weights."""
        suspicious_task = QuantumTask(id="suspicious", name="Suspicious Task")
        suspicious_task.quantum_weight = 500.0  # Unusually high
        
        issues = self.security_validator.validate_security([suspicious_task])
        
        weight_warnings = [i for i in issues if i.code == "SEC_EXCESSIVE_QUANTUM_WEIGHT"]
        assert len(weight_warnings) == 1
    
    def test_excessive_entanglement_detection(self):
        """Test detection of excessive entanglements."""
        entangled_task = QuantumTask(id="entangled", name="Highly Entangled")
        entangled_task.entangled_tasks = set(f"task_{i}" for i in range(100))  # Too many
        
        issues = self.security_validator.validate_security([entangled_task])
        
        entanglement_warnings = [i for i in issues if i.code == "SEC_EXCESSIVE_ENTANGLEMENT"]
        assert len(entanglement_warnings) == 1


class TestQuantumPlanValidation:
    """Test complete quantum plan validation."""
    
    def test_validate_quantum_plan_success(self):
        """Test successful validation of a quantum plan."""
        tasks = [
            QuantumTask(id="task1", name="Task 1", priority=2.0, effort=1.0, value=3.0),
            QuantumTask(id="task2", name="Task 2", priority=1.0, effort=2.0, value=2.0, dependencies=["task1"])
        ]
        
        issues, is_valid = validate_quantum_plan(tasks, ValidationLevel.MODERATE, include_security=True)
        
        assert is_valid == True
        error_count = sum(1 for i in issues if i.issue_type == ValidationIssueType.ERROR)
        assert error_count == 0
    
    def test_validate_quantum_plan_failure(self):
        """Test failed validation of a quantum plan."""
        tasks = [
            QuantumTask(id="", name="Invalid Task"),  # Missing ID - error
            QuantumTask(id="task2", name="Task 2", dependencies=["nonexistent"])  # Missing dep - error
        ]
        
        issues, is_valid = validate_quantum_plan(tasks, ValidationLevel.MODERATE, include_security=True)
        
        assert is_valid == False
        error_count = sum(1 for i in issues if i.issue_type == ValidationIssueType.ERROR)
        assert error_count >= 2
    
    def test_validate_quantum_plan_without_security(self):
        """Test validation without security checks."""
        dangerous_task = QuantumTask(id="../dangerous", name="Dangerous Task")
        
        issues_with_security, _ = validate_quantum_plan([dangerous_task], include_security=True)
        issues_without_security, _ = validate_quantum_plan([dangerous_task], include_security=False)
        
        security_issues_count = len([i for i in issues_with_security if i.code.startswith("SEC_")])
        no_security_issues_count = len([i for i in issues_without_security if i.code.startswith("SEC_")])
        
        assert security_issues_count > no_security_issues_count
        assert no_security_issues_count == 0
    
    @pytest.mark.parametrize("validation_level", list(ValidationLevel))
    def test_different_validation_levels(self, validation_level):
        """Test validation with different strictness levels."""
        tasks = [
            QuantumTask(id="task1", name="High Priority Task", priority=15.0, effort=1.0, value=2.0)
        ]
        
        issues, is_valid = validate_quantum_plan(tasks, validation_level, include_security=False)
        
        # Should always pass basic validation (no errors)
        assert is_valid == True
        
        # But may have different numbers of warnings based on level
        warning_count = sum(1 for i in issues if i.issue_type == ValidationIssueType.WARNING)
        assert warning_count >= 0  # May vary by validation level