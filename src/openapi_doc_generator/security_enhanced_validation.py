"""
Enhanced Security Validation System

This module provides comprehensive security validation for inputs, outputs,
and operations to prevent malicious usage and ensure safe operation.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .enhanced_error_handling import RateLimitExceededError, SecurityValidationError


@dataclass
class SecurityConfig:
    """Security configuration for validation system."""
    max_file_size_mb: int = 100
    max_request_rate: int = 100  # requests per minute
    blocked_patterns: List[str] = field(default_factory=lambda: [
        r'\.\./',  # Path traversal
        r'<script',  # XSS attempts
        r'eval\(',  # Code injection
        r'exec\(',  # Code execution
        r'import\s+os',  # OS module imports
        r'subprocess',  # Subprocess calls
        r'__import__',  # Dynamic imports
    ])
    max_task_effort: float = 1000.0
    suspicious_task_patterns: List[str] = field(default_factory=lambda: [
        r'delete.*database',
        r'drop.*table',
        r'rm\s+-rf',
        r'format.*c:',
        r'malware',
        r'backdoor',
        r'exploit',
    ])
    rate_limit_windows: Dict[str, int] = field(default_factory=lambda: {
        'minute': 100,
        'hour': 1000,
        'day': 10000
    })


class SecurityValidator:
    """Comprehensive security validation system."""

    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.request_history: Dict[str, List[datetime]] = defaultdict(list)
        self.blocked_ips: Set[str] = set()
        self.security_alerts: List[Dict[str, Any]] = []

    def validate_input_safety(self, input_text: str, context: str = "general") -> bool:
        """Validate that input text is safe from malicious patterns."""
        if not input_text:
            return True

        for pattern in self.config.blocked_patterns:
            if re.search(pattern, input_text, re.IGNORECASE):
                self._log_security_alert(
                    alert_type="malicious_pattern",
                    context=context,
                    details=f"Blocked pattern detected: {pattern}",
                    input_sample=input_text[:100]
                )
                raise SecurityValidationError(
                    f"Input contains potentially malicious pattern: {pattern[:20]}..."
                )
        return True

    def validate_file_path_safety(self, file_path: str) -> bool:
        """Validate file path for security issues."""
        try:
            path = Path(file_path).resolve()
        except Exception:
            raise SecurityValidationError(f"Invalid file path: {file_path}")

        # Check for path traversal attempts
        if '..' in str(path) or str(path).startswith('/'):
            if not str(path).startswith('/root/repo'):  # Allow repo paths
                raise SecurityValidationError("Path traversal attempt detected")

        # Check file size
        if path.exists() and path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > self.config.max_file_size_mb:
                raise SecurityValidationError(
                    f"File too large: {size_mb:.1f}MB (max: {self.config.max_file_size_mb}MB)"
                )

        return True

    def validate_task_security(self, task_name: str, task_effort: float) -> bool:
        """Validate task for security issues."""
        # Check suspicious patterns
        for pattern in self.config.suspicious_task_patterns:
            if re.search(pattern, task_name, re.IGNORECASE):
                self._log_security_alert(
                    alert_type="suspicious_task",
                    context="task_validation",
                    details=f"Suspicious task pattern: {pattern}",
                    task_name=task_name
                )
                raise SecurityValidationError("Task name contains suspicious pattern")

        # Check effort bounds
        if task_effort > self.config.max_task_effort:
            raise SecurityValidationError(
                f"Task effort too high: {task_effort} (max: {self.config.max_task_effort})"
            )

        return True

    def validate_rate_limit(self, client_id: str = "default") -> bool:
        """Check if client is within rate limits."""
        now = datetime.now()

        # Clean old entries
        cutoff_minute = now - timedelta(minutes=1)
        cutoff_hour = now - timedelta(hours=1)
        cutoff_day = now - timedelta(days=1)

        history = self.request_history[client_id]
        self.request_history[client_id] = [
            ts for ts in history if ts > cutoff_day
        ]

        # Check limits
        recent_minute = len([ts for ts in history if ts > cutoff_minute])
        recent_hour = len([ts for ts in history if ts > cutoff_hour])
        recent_day = len([ts for ts in history if ts > cutoff_day])

        if (recent_minute > self.config.rate_limit_windows['minute'] or
            recent_hour > self.config.rate_limit_windows['hour'] or
            recent_day > self.config.rate_limit_windows['day']):

            self._log_security_alert(
                alert_type="rate_limit_exceeded",
                context="rate_limiting",
                details=f"Client {client_id} exceeded rate limits",
                rate_info={
                    "minute": recent_minute,
                    "hour": recent_hour,
                    "day": recent_day
                }
            )
            raise RateLimitExceededError(f"Rate limit exceeded for client {client_id}")

        # Record request
        self.request_history[client_id].append(now)
        return True

    def sanitize_input(self, input_text: str) -> str:
        """Sanitize input text for safe processing."""
        if not input_text:
            return ""

        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>&"\']', '', input_text)

        # Limit length
        if len(sanitized) > 10000:
            sanitized = sanitized[:10000] + "..."

        return sanitized

    def get_security_report(self) -> Dict[str, Any]:
        """Generate security validation report."""
        return {
            "total_alerts": len(self.security_alerts),
            "blocked_clients": len(self.blocked_ips),
            "active_clients": len(self.request_history),
            "recent_alerts": self.security_alerts[-10:] if self.security_alerts else [],
            "alert_summary": self._summarize_alerts(),
            "timestamp": datetime.now().isoformat()
        }

    def _log_security_alert(self, alert_type: str, context: str, details: str, **kwargs):
        """Log security alert for monitoring."""
        alert = {
            "type": alert_type,
            "context": context,
            "details": details,
            "timestamp": datetime.now().isoformat(),
            "severity": "high",
            **kwargs
        }
        self.security_alerts.append(alert)

        # Keep only last 1000 alerts
        if len(self.security_alerts) > 1000:
            self.security_alerts = self.security_alerts[-1000:]

    def _summarize_alerts(self) -> Dict[str, int]:
        """Summarize alerts by type."""
        summary = defaultdict(int)
        for alert in self.security_alerts:
            summary[alert["type"]] += 1
        return dict(summary)


# Global security validator instance
_global_validator: Optional[SecurityValidator] = None


def get_security_validator() -> SecurityValidator:
    """Get global security validator instance."""
    global _global_validator
    if _global_validator is None:
        _global_validator = SecurityValidator()
    return _global_validator


def validate_security(input_text: str, context: str = "general") -> bool:
    """Convenience function for security validation."""
    return get_security_validator().validate_input_safety(input_text, context)


def security_check(func):
    """Decorator for automatic security validation."""
    def wrapper(*args, **kwargs):
        validator = get_security_validator()

        # Basic rate limiting
        validator.validate_rate_limit()

        # Validate string inputs
        for arg in args:
            if isinstance(arg, str):
                validator.validate_input_safety(arg, func.__name__)

        for key, value in kwargs.items():
            if isinstance(value, str):
                validator.validate_input_safety(value, f"{func.__name__}.{key}")

        return func(*args, **kwargs)
    return wrapper
