"""Advanced audit logging and security event monitoring."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

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
    user_id: Optional[str]
    session_id: Optional[str]
    source_ip: Optional[str]
    resource: Optional[str]
    action: str
    result: str  # "success", "failure", "warning"
    severity: SecurityLevel
    details: Dict[str, Any]
    correlation_id: Optional[str] = None


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
        self.audit_buffer: List[AuditEvent] = []
        self.security_alerts: List[AuditEvent] = []

        # Setup structured logging
        self._setup_audit_handler()

    def _setup_audit_handler(self):
        """Setup dedicated audit log handler."""
        # Create audit-specific handler
        audit_handler = logging.FileHandler('/tmp/quantum_audit.log')
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
                          user_id: Optional[str] = None,
                          resource: Optional[str] = None,
                          details: Optional[Dict[str, Any]] = None,
                          correlation_id: Optional[str] = None) -> None:
        """Log a security audit event."""
        event = AuditEvent(
            event_type=event_type,
            timestamp=time.time(),
            user_id=user_id,
            session_id=None,  # TODO: Extract from context
            source_ip=None,   # TODO: Extract from request
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
                                 source_ip: Optional[str] = None) -> None:
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
                             details: Dict[str, Any],
                             severity: SecurityLevel = SecurityLevel.HIGH,
                             user_id: Optional[str] = None) -> None:
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
                          user_id: Optional[str] = None,
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
                       user_id: Optional[str] = None,
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
                               user_id: Optional[str] = None) -> None:
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
                          error_details: Dict[str, Any],
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
                           severity_filter: Optional[SecurityLevel] = None,
                           limit: int = 100) -> List[AuditEvent]:
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
            # TODO: Implement audit log encryption
            audit_data["encrypted"] = True

        self.logger.info(f"AUDIT_RECORD: {json.dumps(audit_data)}")

    def generate_compliance_report(self,
                                 start_time: float,
                                 end_time: float) -> Dict[str, Any]:
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


# Global audit logger instance
_audit_logger: Optional[QuantumAuditLogger] = None


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
