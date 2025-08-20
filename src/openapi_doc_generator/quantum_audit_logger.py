"""Advanced audit logging and security event monitoring."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from .quantum_security import SecurityLevel


class AuditEventType(Enum):
    """Types of security audit events."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    SECURITY_VIOLATION = "security_violation"
    SYSTEM_ACCESS = "system_access"
    CONFIGURATION_CHANGE = "configuration_change"
    RESOURCE_ACCESS = "resource_access"
    ERROR_CONDITION = "error_condition"


@dataclass
class AuditEvent:
    """Security audit event record."""
    event_type: AuditEventType
    timestamp: float
    user_id: str | None
    session_id: str | None
    source_ip: str | None
    resource: str | None
    action: str
    result: str  # "success", "failure", "warning"
    severity: SecurityLevel
    details: dict[str, Any]
    correlation_id: str | None = None


class QuantumAuditLogger:
    """Advanced security audit logger with compliance features."""

    def __init__(self,
                 enable_encryption: bool = True,
                 retention_days: int = 90,
                 compliance_mode: str = "SOX"):
        """Initialize audit logger with compliance settings."""
        self.logger = logging.getLogger(f"{__name__}.audit")
        self.enable_encryption = enable_encryption
        self.retention_days = retention_days
        self.compliance_mode = compliance_mode
        self.audit_buffer: list[AuditEvent] = []
        self.security_alerts: list[AuditEvent] = []

        # Setup structured logging
        self._setup_audit_handler()

    def _setup_audit_handler(self):
        """Setup dedicated audit log handler."""
        # Create audit-specific handler with secure temp directory
        import tempfile
        import os
        audit_dir = os.path.join(tempfile.gettempdir(), 'quantum_audit')
        os.makedirs(audit_dir, mode=0o700, exist_ok=True)
        audit_file = os.path.join(audit_dir, 'quantum_audit.log')
        audit_handler = logging.FileHandler(audit_file)
        audit_formatter = logging.Formatter(
            '%(asctime)s|AUDIT|%(levelname)s|%(message)s'
        )
        audit_handler.setFormatter(audit_formatter)
        self.logger.addHandler(audit_handler)
        self.logger.setLevel(logging.INFO)

    def log_security_event(self,
                          event_type: AuditEventType,
                          action: str,
                          result: str,
                          severity: SecurityLevel = SecurityLevel.MEDIUM,
                          user_id: str | None = None,
                          resource: str | None = None,
                          details: dict[str, Any] | None = None,
                          correlation_id: str | None = None) -> None:
        """Log a security audit event."""
        event = AuditEvent(
            event_type=event_type,
            timestamp=time.time(),
            user_id=user_id,
            session_id=self._extract_session_id(),
            source_ip=self._extract_source_ip(),
            resource=resource,
            action=action,
            result=result,
            severity=severity,
            details=details or {},
            correlation_id=correlation_id
        )

        # Store in buffer for batch processing
        self.audit_buffer.append(event)

        # Immediate logging for critical events
        if severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            self._immediate_log(event)
            self.security_alerts.append(event)

        # Log structured JSON event
        event_dict = asdict(event)
        # Convert enum to string for JSON serialization
        event_dict['event_type'] = event_dict['event_type'].value
        event_dict['severity'] = event_dict['severity'].value
        self.logger.info(json.dumps(event_dict, indent=None))

    def _immediate_log(self, event: AuditEvent) -> None:
        """Immediately log critical security events."""
        self.logger.critical(
            f"CRITICAL_SECURITY_EVENT: {event.event_type.value} - "
            f"{event.action} - {event.result} - Resource: {event.resource}"
        )

    def log_authentication_attempt(self,
                                 user_id: str,
                                 success: bool,
                                 method: str = "api_key",
                                 source_ip: str | None = None) -> None:
        """Log authentication attempt."""
        self.log_security_event(
            event_type=AuditEventType.AUTHENTICATION,
            action=f"authenticate_via_{method}",
            result="success" if success else "failure",
            severity=SecurityLevel.LOW if success else SecurityLevel.MEDIUM,
            user_id=user_id,
            details={
                "authentication_method": method,
                "source_ip": source_ip,
                "timestamp": time.time()
            }
        )

    def log_security_violation(self,
                             violation_type: str,
                             details: dict[str, Any],
                             severity: SecurityLevel = SecurityLevel.HIGH,
                             user_id: str | None = None) -> None:
        """Log security policy violation."""
        self.log_security_event(
            event_type=AuditEventType.SECURITY_VIOLATION,
            action=f"policy_violation_{violation_type}",
            result="blocked",
            severity=severity,
            user_id=user_id,
            details=details
        )

    def log_resource_access(self,
                          resource: str,
                          action: str,
                          user_id: str | None = None,
                          success: bool = True) -> None:
        """Log resource access attempt."""
        self.log_security_event(
            event_type=AuditEventType.RESOURCE_ACCESS,
            action=action,
            result="success" if success else "denied",
            severity=SecurityLevel.LOW,
            user_id=user_id,
            resource=resource
        )

    def log_data_access(self,
                       data_type: str,
                       operation: str,
                       user_id: str | None = None,
                       classification: str = "internal") -> None:
        """Log data access for compliance."""
        self.log_security_event(
            event_type=AuditEventType.DATA_ACCESS,
            action=f"{operation}_{data_type}",
            result="accessed",
            severity=SecurityLevel.LOW,
            user_id=user_id,
            details={
                "data_type": data_type,
                "data_classification": classification,
                "compliance_mode": self.compliance_mode
            }
        )

    def log_configuration_change(self,
                               config_key: str,
                               old_value: Any,
                               new_value: Any,
                               user_id: str | None = None) -> None:
        """Log configuration changes."""
        self.log_security_event(
            event_type=AuditEventType.CONFIGURATION_CHANGE,
            action="configuration_modified",
            result="changed",
            severity=SecurityLevel.MEDIUM,
            user_id=user_id,
            resource=config_key,
            details={
                "config_key": config_key,
                "old_value": str(old_value),
                "new_value": str(new_value)
            }
        )

    def log_error_condition(self,
                          error_type: str,
                          error_details: dict[str, Any],
                          severity: SecurityLevel = SecurityLevel.LOW) -> None:
        """Log error conditions that might indicate security issues."""
        self.log_security_event(
            event_type=AuditEventType.ERROR_CONDITION,
            action=f"error_{error_type}",
            result="error",
            severity=severity,
            details=error_details
        )

    def get_security_alerts(self,
                           severity_filter: SecurityLevel | None = None,
                           limit: int = 100) -> list[AuditEvent]:
        """Get recent security alerts."""
        alerts = self.security_alerts[-limit:]

        if severity_filter:
            alerts = [a for a in alerts if a.severity == severity_filter]

        return alerts

    def flush_audit_buffer(self) -> None:
        """Flush audit buffer to persistent storage."""
        if self.audit_buffer:
            self.logger.info(f"Flushing {len(self.audit_buffer)} audit events")

            # Batch write audit events
            for event in self.audit_buffer:
                self._write_audit_record(event)

            self.audit_buffer.clear()

    def _write_audit_record(self, event: AuditEvent) -> None:
        """Write individual audit record to storage."""
        # In production, this would write to secure audit storage
        # For now, log to file with encryption if enabled
        audit_data = asdict(event)

        if self.enable_encryption:
            audit_data = self._encrypt_audit_data(audit_data)
            audit_data["encrypted"] = True

        self.logger.info(f"AUDIT_RECORD: {json.dumps(audit_data)}")

    def generate_compliance_report(self,
                                 start_time: float,
                                 end_time: float) -> dict[str, Any]:
        """Generate compliance audit report."""
        # Filter events within time range
        filtered_events = [
            e for e in self.audit_buffer + self.security_alerts
            if start_time <= e.timestamp <= end_time
        ]

        # Generate compliance metrics
        report = {
            "compliance_mode": self.compliance_mode,
            "report_period": {
                "start": start_time,
                "end": end_time
            },
            "event_summary": {
                "total_events": len(filtered_events),
                "by_type": {},
                "by_severity": {},
                "by_result": {}
            },
            "security_summary": {
                "authentication_attempts": 0,
                "security_violations": 0,
                "data_access_events": 0,
                "configuration_changes": 0
            }
        }

        # Analyze event patterns
        for event in filtered_events:
            # By type
            event_type = event.event_type.value
            report["event_summary"]["by_type"][event_type] = \
                report["event_summary"]["by_type"].get(event_type, 0) + 1

            # By severity
            severity = event.severity.value
            report["event_summary"]["by_severity"][severity] = \
                report["event_summary"]["by_severity"].get(severity, 0) + 1

            # By result
            result = event.result
            report["event_summary"]["by_result"][result] = \
                report["event_summary"]["by_result"].get(result, 0) + 1

            # Security metrics
            if event.event_type == AuditEventType.AUTHENTICATION:
                report["security_summary"]["authentication_attempts"] += 1
            elif event.event_type == AuditEventType.SECURITY_VIOLATION:
                report["security_summary"]["security_violations"] += 1
            elif event.event_type == AuditEventType.DATA_ACCESS:
                report["security_summary"]["data_access_events"] += 1
            elif event.event_type == AuditEventType.CONFIGURATION_CHANGE:
                report["security_summary"]["configuration_changes"] += 1

        return report

    def _extract_session_id(self) -> str | None:
        """Extract session ID from current context."""
        # In production, this would extract from request context, thread local storage, etc.
        import threading
        thread_name = threading.current_thread().name
        return f"session_{hash(thread_name) & 0xFFFFFF:06x}" if thread_name != "MainThread" else None

    def _extract_source_ip(self) -> str | None:
        """Extract source IP from current request context."""
        # In production, this would extract from HTTP request headers, context variables, etc.
        import os
        # For now, return localhost or environment-based IP
        return os.environ.get("CLIENT_IP", "127.0.0.1")

    def _encrypt_audit_data(self, data: dict) -> dict:
        """Encrypt sensitive audit data fields."""
        # Simple base64 encoding for demonstration - in production use proper encryption
        import base64
        import json
        
        sensitive_fields = ["user_id", "details", "resource"]
        encrypted_data = data.copy()
        
        for field in sensitive_fields:
            if field in encrypted_data and encrypted_data[field] is not None:
                value = json.dumps(encrypted_data[field]) if isinstance(encrypted_data[field], dict) else str(encrypted_data[field])
                encrypted_data[field] = base64.b64encode(value.encode()).decode()
        
        return encrypted_data


# Global audit logger instance
_audit_logger: QuantumAuditLogger | None = None


def get_audit_logger() -> QuantumAuditLogger:
    """Get global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = QuantumAuditLogger()
    return _audit_logger


def audit_security_event(event_type: AuditEventType,
                        action: str,
                        result: str,
                        **kwargs) -> None:
    """Convenience function for logging audit events."""
    logger = get_audit_logger()
    logger.log_security_event(
        event_type=event_type,
        action=action,
        result=result,
        **kwargs
    )
