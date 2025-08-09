"""Security validation and compliance for quantum task planning."""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from .quantum_scheduler import QuantumTask

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security validation levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityIssue:
    """Security validation issue."""
    issue_type: str
    severity: SecurityLevel
    message: str
    task_id: Optional[str] = None
    recommendation: Optional[str] = None


class QuantumSecurityValidator:
    """Security validator for quantum task planning."""

    def __init__(self, security_level: SecurityLevel = SecurityLevel.MEDIUM):
        """Initialize security validator."""
        self.security_level = security_level
        self.session_keys: Dict[str, str] = {}
        self.rate_limits: Dict[str, List[float]] = {}
        self.blocked_patterns = [
            # Dangerous file patterns
            r'\.\./.*',
            r'/etc/.*',
            r'/proc/.*',
            r'/dev/.*',
            # Command injection patterns
            r'[;&|`$]',
            r'\$\(',
            r'`.*`',
            # Script injection patterns
            r'<script.*>.*</script>',
            r'javascript:',
            r'vbscript:',
        ]

    def validate_task_security(self, task: QuantumTask) -> List[SecurityIssue]:
        """Validate task for security issues."""
        issues = []

        # Validate task name for injection patterns
        if self._contains_suspicious_patterns(task.name):
            issues.append(SecurityIssue(
                issue_type="injection_risk",
                severity=SecurityLevel.HIGH,
                message=f"Task name '{task.name}' contains suspicious patterns",
                task_id=task.id,
                recommendation="Sanitize task names to prevent injection attacks"
            ))

        # Check for excessive resource requests
        if task.effort > 100:
            issues.append(SecurityIssue(
                issue_type="resource_abuse",
                severity=SecurityLevel.MEDIUM,
                message=f"Task '{task.id}' requests excessive effort: {task.effort}",
                task_id=task.id,
                recommendation="Review task effort requirements for potential DoS"
            ))

        # Validate coherence time bounds
        if task.coherence_time <= 0 or task.coherence_time > 3600:
            issues.append(SecurityIssue(
                issue_type="invalid_parameter",
                severity=SecurityLevel.LOW,
                message=f"Task coherence time out of bounds: {task.coherence_time}",
                task_id=task.id,
                recommendation="Set coherence time between 0.1 and 3600 seconds"
            ))

        # Check dependency chain depth (prevent dependency bombs)
        if len(task.dependencies) > 20:
            issues.append(SecurityIssue(
                issue_type="dependency_bomb",
                severity=SecurityLevel.HIGH,
                message=f"Task '{task.id}' has excessive dependencies: {len(task.dependencies)}",
                task_id=task.id,
                recommendation="Limit dependency chains to prevent exponential complexity"
            ))

        return issues

    def validate_plan_security(self, tasks: List[QuantumTask]) -> List[SecurityIssue]:
        """Validate entire task plan for security issues."""
        issues = []

        # Check total resource consumption
        total_effort = sum(task.effort for task in tasks)
        if total_effort > 10000:
            issues.append(SecurityIssue(
                issue_type="resource_exhaustion",
                severity=SecurityLevel.CRITICAL,
                message=f"Total plan effort exceeds safe limits: {total_effort}",
                recommendation="Reduce total effort or split into multiple plans"
            ))

        # Check for circular dependencies
        dep_graph = {task.id: set(task.dependencies) for task in tasks}
        cycles = self._detect_dependency_cycles(dep_graph)
        if cycles:
            issues.append(SecurityIssue(
                issue_type="circular_dependency",
                severity=SecurityLevel.HIGH,
                message=f"Circular dependencies detected: {cycles}",
                recommendation="Remove circular dependencies to prevent deadlocks"
            ))

        # Validate individual tasks
        for task in tasks:
            issues.extend(self.validate_task_security(task))

        # Check for task flooding (too many tasks)
        if len(tasks) > 1000:
            issues.append(SecurityIssue(
                issue_type="task_flooding",
                severity=SecurityLevel.MEDIUM,
                message=f"Plan contains excessive tasks: {len(tasks)}",
                recommendation="Split large plans to prevent resource exhaustion"
            ))

        return issues

    def generate_session_token(self, session_id: str) -> str:
        """Generate secure session token."""
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(f"{session_id}:{token}:{time.time()}".encode()).hexdigest()
        self.session_keys[session_id] = token_hash
        logger.info(f"Generated session token for {session_id}")
        return token

    def validate_session_token(self, session_id: str, token: str) -> bool:
        """Validate session token."""
        if session_id not in self.session_keys:
            logger.warning(f"Session token validation failed: unknown session {session_id}")
            return False

        expected_hash = self.session_keys[session_id]
        token_hash = hashlib.sha256(f"{session_id}:{token}:{time.time()}".encode()).hexdigest()

        # Use timing-safe comparison
        is_valid = hmac.compare_digest(expected_hash, token_hash)

        if not is_valid:
            logger.warning(f"Session token validation failed for {session_id}")

        return is_valid

    def check_rate_limit(self, client_id: str, max_requests: int = 100, window_seconds: int = 3600) -> bool:
        """Check if client is within rate limits."""
        current_time = time.time()

        if client_id not in self.rate_limits:
            self.rate_limits[client_id] = []

        # Clean old requests outside window
        self.rate_limits[client_id] = [
            req_time for req_time in self.rate_limits[client_id]
            if current_time - req_time < window_seconds
        ]

        # Check if within limits
        if len(self.rate_limits[client_id]) >= max_requests:
            logger.warning(f"Rate limit exceeded for client {client_id}")
            return False

        # Record current request
        self.rate_limits[client_id].append(current_time)
        return True

    def sanitize_input(self, input_data: Any) -> Any:
        """Sanitize input data to prevent injection attacks."""
        if isinstance(input_data, str):
            # Remove suspicious patterns
            sanitized = input_data
            for pattern in self.blocked_patterns:
                import re
                sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)

            # Limit length
            if len(sanitized) > 1000:
                sanitized = sanitized[:1000]
                logger.warning("Input truncated due to length limit")

            return sanitized.strip()

        elif isinstance(input_data, dict):
            return {k: self.sanitize_input(v) for k, v in input_data.items()}

        elif isinstance(input_data, list):
            return [self.sanitize_input(item) for item in input_data]

        else:
            return input_data

    def audit_log_security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log security events for auditing."""
        audit_entry = {
            "timestamp": time.time(),
            "event_type": event_type,
            "severity": self.security_level.value,
            "details": details
        }

        # In production, this should go to a secure audit log system
        logger.info(f"Security audit: {event_type}", extra={"audit": audit_entry})

    def _contains_suspicious_patterns(self, text: str) -> bool:
        """Check if text contains suspicious patterns."""
        import re
        for pattern in self.blocked_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _detect_dependency_cycles(self, dep_graph: Dict[str, set]) -> List[List[str]]:
        """Detect circular dependencies using DFS."""
        WHITE = 0  # Unvisited
        GRAY = 1   # Being processed
        BLACK = 2  # Fully processed

        colors = dict.fromkeys(dep_graph, WHITE)
        cycles = []

        def dfs(node: str, path: List[str]) -> None:
            if colors[node] == GRAY:
                # Found a cycle
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return

            if colors[node] == BLACK:
                return

            colors[node] = GRAY
            path.append(node)

            for neighbor in dep_graph.get(node, set()):
                if neighbor in dep_graph:  # Only check if neighbor exists
                    dfs(neighbor, path)

            path.pop()
            colors[node] = BLACK

        for node in dep_graph:
            if colors[node] == WHITE:
                dfs(node, [])

        return cycles

    def generate_security_report(self, tasks: List[QuantumTask]) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        issues = self.validate_plan_security(tasks)

        # Categorize issues by severity
        critical = [i for i in issues if i.severity == SecurityLevel.CRITICAL]
        high = [i for i in issues if i.severity == SecurityLevel.HIGH]
        medium = [i for i in issues if i.severity == SecurityLevel.MEDIUM]
        low = [i for i in issues if i.severity == SecurityLevel.LOW]

        # Calculate security score
        security_score = max(0, 100 - (len(critical) * 25 + len(high) * 10 + len(medium) * 5 + len(low) * 2))

        report = {
            "security_score": security_score,
            "total_issues": len(issues),
            "issues_by_severity": {
                "critical": len(critical),
                "high": len(high),
                "medium": len(medium),
                "low": len(low)
            },
            "detailed_issues": [
                {
                    "type": issue.issue_type,
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "task_id": issue.task_id,
                    "recommendation": issue.recommendation
                }
                for issue in issues
            ],
            "compliance_status": "PASS" if security_score >= 80 else "FAIL",
            "recommendations": self._generate_security_recommendations(issues)
        }

        self.audit_log_security_event("security_report_generated", {
            "task_count": len(tasks),
            "security_score": security_score,
            "total_issues": len(issues)
        })

        return report

    def _generate_security_recommendations(self, issues: List[SecurityIssue]) -> List[str]:
        """Generate security recommendations based on issues."""
        recommendations = []

        if any(i.issue_type == "injection_risk" for i in issues):
            recommendations.append("Implement input sanitization for all user-provided data")

        if any(i.issue_type == "resource_abuse" for i in issues):
            recommendations.append("Set resource limits and implement monitoring")

        if any(i.issue_type == "circular_dependency" for i in issues):
            recommendations.append("Validate dependency graphs before execution")

        if any(i.severity == SecurityLevel.CRITICAL for i in issues):
            recommendations.append("Address critical security issues before production deployment")

        if not recommendations:
            recommendations.append("Security validation passed - maintain current security practices")

        return recommendations


# Global security validator instance
_security_validator = None


def get_security_validator(security_level: SecurityLevel = SecurityLevel.MEDIUM) -> QuantumSecurityValidator:
    """Get global security validator instance."""
    global _security_validator
    if _security_validator is None:
        _security_validator = QuantumSecurityValidator(security_level)
    return _security_validator
