"""Compliance framework for quantum task planning system."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ComplianceStandard(Enum):
    """Supported compliance standards."""
    GDPR = "gdpr"
    CCPA = "ccpa"
    PDPA = "pdpa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    NIST_CSF = "nist_csf"


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class ComplianceEvent:
    """Compliance audit event."""
    event_id: str
    timestamp: float
    event_type: str
    data_classification: DataClassification
    user_id: Optional[str]
    session_id: Optional[str]
    data_processed: bool
    retention_required: bool
    compliance_standards: List[ComplianceStandard]
    metadata: Dict[str, Any]


@dataclass
class DataRetentionPolicy:
    """Data retention policy configuration."""
    classification: DataClassification
    retention_days: int
    auto_deletion: bool
    archive_required: bool
    compliance_standards: List[ComplianceStandard]


@dataclass
class PrivacySettings:
    """Privacy configuration settings."""
    data_anonymization: bool = True
    consent_required: bool = True
    right_to_deletion: bool = True
    data_portability: bool = True
    purpose_limitation: bool = True
    data_minimization: bool = True


class QuantumComplianceManager:
    """Manages compliance for quantum task planning system."""
    
    def __init__(self, enabled_standards: Optional[List[ComplianceStandard]] = None):
        """Initialize compliance manager."""
        self.enabled_standards = enabled_standards or [
            ComplianceStandard.GDPR,
            ComplianceStandard.SOC2,
            ComplianceStandard.NIST_CSF
        ]
        
        self.privacy_settings = PrivacySettings()
        self.audit_log: List[ComplianceEvent] = []
        self.retention_policies = self._setup_default_retention_policies()
        self.consent_records: Dict[str, Dict[str, Any]] = {}
        self.data_inventory: Dict[str, Dict[str, Any]] = {}
        
        # Initialize compliance tracking
        self.compliance_status = {
            standard: {"compliant": True, "issues": [], "last_check": time.time()}
            for standard in self.enabled_standards
        }
        
        logger.info(f"Compliance manager initialized with standards: {[s.value for s in self.enabled_standards]}")
    
    def _setup_default_retention_policies(self) -> Dict[DataClassification, DataRetentionPolicy]:
        """Setup default data retention policies."""
        return {
            DataClassification.PUBLIC: DataRetentionPolicy(
                classification=DataClassification.PUBLIC,
                retention_days=365,
                auto_deletion=False,
                archive_required=False,
                compliance_standards=[ComplianceStandard.GDPR, ComplianceStandard.CCPA]
            ),
            DataClassification.INTERNAL: DataRetentionPolicy(
                classification=DataClassification.INTERNAL,
                retention_days=2555,  # 7 years
                auto_deletion=True,
                archive_required=True,
                compliance_standards=[ComplianceStandard.SOC2, ComplianceStandard.ISO27001]
            ),
            DataClassification.CONFIDENTIAL: DataRetentionPolicy(
                classification=DataClassification.CONFIDENTIAL,
                retention_days=1095,  # 3 years
                auto_deletion=True,
                archive_required=True,
                compliance_standards=[ComplianceStandard.GDPR, ComplianceStandard.SOC2]
            ),
            DataClassification.RESTRICTED: DataRetentionPolicy(
                classification=DataClassification.RESTRICTED,
                retention_days=90,
                auto_deletion=True,
                archive_required=False,
                compliance_standards=[ComplianceStandard.GDPR, ComplianceStandard.CCPA, ComplianceStandard.PDPA]
            )
        }
    
    def log_compliance_event(self,
                           event_type: str,
                           data_classification: DataClassification = DataClassification.INTERNAL,
                           user_id: Optional[str] = None,
                           session_id: Optional[str] = None,
                           data_processed: bool = False,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """Log a compliance event for audit trail."""
        event_id = str(uuid.uuid4())
        
        # Determine applicable compliance standards
        applicable_standards = []
        for standard in self.enabled_standards:
            if self._is_standard_applicable(standard, data_classification, event_type):
                applicable_standards.append(standard)
        
        # Determine retention requirements
        policy = self.retention_policies.get(data_classification)
        retention_required = policy.archive_required if policy else False
        
        event = ComplianceEvent(
            event_id=event_id,
            timestamp=time.time(),
            event_type=event_type,
            data_classification=data_classification,
            user_id=user_id,
            session_id=session_id,
            data_processed=data_processed,
            retention_required=retention_required,
            compliance_standards=applicable_standards,
            metadata=metadata or {}
        )
        
        self.audit_log.append(event)
        
        # Update data inventory
        if data_processed:
            self._update_data_inventory(event)
        
        # Check for compliance violations
        self._validate_event_compliance(event)
        
        logger.info(f"Compliance event logged: {event_type} [{event_id}]")
        return event_id
    
    def record_consent(self,
                      user_id: str,
                      purpose: str,
                      consent_given: bool,
                      data_types: List[str],
                      retention_period: Optional[int] = None) -> str:
        """Record user consent for data processing."""
        consent_id = str(uuid.uuid4())
        
        consent_record = {
            "consent_id": consent_id,
            "user_id": user_id,
            "timestamp": time.time(),
            "purpose": purpose,
            "consent_given": consent_given,
            "data_types": data_types,
            "retention_period": retention_period,
            "ip_address": "anonymized",  # In production, collect actual IP
            "user_agent": "anonymized",  # In production, collect actual user agent
            "withdrawal_method": None,
            "withdrawn_at": None
        }
        
        if user_id not in self.consent_records:
            self.consent_records[user_id] = {}
        
        self.consent_records[user_id][consent_id] = consent_record
        
        # Log compliance event
        self.log_compliance_event(
            event_type="consent_recorded",
            data_classification=DataClassification.CONFIDENTIAL,
            user_id=user_id,
            data_processed=True,
            metadata={
                "consent_id": consent_id,
                "purpose": purpose,
                "consent_given": consent_given
            }
        )
        
        return consent_id
    
    def withdraw_consent(self, user_id: str, consent_id: str, withdrawal_method: str = "api") -> bool:
        """Record consent withdrawal."""
        if user_id not in self.consent_records or consent_id not in self.consent_records[user_id]:
            return False
        
        consent_record = self.consent_records[user_id][consent_id]
        consent_record["withdrawal_method"] = withdrawal_method
        consent_record["withdrawn_at"] = time.time()
        consent_record["consent_given"] = False
        
        # Log compliance event
        self.log_compliance_event(
            event_type="consent_withdrawn",
            data_classification=DataClassification.CONFIDENTIAL,
            user_id=user_id,
            data_processed=True,
            metadata={
                "consent_id": consent_id,
                "withdrawal_method": withdrawal_method
            }
        )
        
        # Trigger data deletion if required
        if self.privacy_settings.right_to_deletion:
            self._schedule_data_deletion(user_id, consent_id)
        
        return True
    
    def anonymize_data(self, data: Dict[str, Any], fields_to_anonymize: List[str]) -> Dict[str, Any]:
        """Anonymize sensitive data fields."""
        if not self.privacy_settings.data_anonymization:
            return data
        
        anonymized = data.copy()
        
        for field in fields_to_anonymize:
            if field in anonymized:
                if isinstance(anonymized[field], str):
                    # Hash-based anonymization
                    import hashlib
                    hash_value = hashlib.sha256(str(anonymized[field]).encode()).hexdigest()
                    anonymized[field] = f"anon_{hash_value[:8]}"
                elif isinstance(anonymized[field], (int, float)):
                    # Numeric anonymization
                    anonymized[field] = -1
                else:
                    anonymized[field] = "anonymized"
        
        return anonymized
    
    def validate_data_processing(self,
                                user_id: Optional[str],
                                purpose: str,
                                data_types: List[str],
                                session_id: Optional[str] = None) -> Tuple[bool, List[str]]:
        """Validate if data processing is compliant."""
        violations = []
        
        # Check consent requirements
        if self.privacy_settings.consent_required and user_id:
            if not self._has_valid_consent(user_id, purpose, data_types):
                violations.append("Missing or invalid user consent")
        
        # Check purpose limitation
        if self.privacy_settings.purpose_limitation:
            if not self._is_purpose_valid(purpose):
                violations.append("Purpose not within specified limitations")
        
        # Check data minimization
        if self.privacy_settings.data_minimization:
            if not self._is_data_minimal(data_types, purpose):
                violations.append("Data collection exceeds minimum required")
        
        # Log the validation attempt
        self.log_compliance_event(
            event_type="data_processing_validation",
            data_classification=DataClassification.CONFIDENTIAL,
            user_id=user_id,
            session_id=session_id,
            data_processed=len(violations) == 0,
            metadata={
                "purpose": purpose,
                "data_types": data_types,
                "violations": violations
            }
        )
        
        return len(violations) == 0, violations
    
    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all user data for portability (GDPR Article 20)."""
        if not self.privacy_settings.data_portability:
            raise ValueError("Data portability not enabled")
        
        user_data = {
            "user_id": user_id,
            "export_timestamp": time.time(),
            "consent_records": self.consent_records.get(user_id, {}),
            "audit_events": [],
            "data_inventory": []
        }
        
        # Find all audit events for this user
        for event in self.audit_log:
            if event.user_id == user_id:
                user_data["audit_events"].append(asdict(event))
        
        # Find all data inventory entries for this user
        for data_id, inventory in self.data_inventory.items():
            if inventory.get("user_id") == user_id:
                user_data["data_inventory"].append(inventory)
        
        # Log the export
        self.log_compliance_event(
            event_type="data_export",
            data_classification=DataClassification.CONFIDENTIAL,
            user_id=user_id,
            data_processed=True,
            metadata={"export_size": len(user_data["audit_events"]) + len(user_data["data_inventory"])}
        )
        
        return user_data
    
    def delete_user_data(self, user_id: str, reason: str = "user_request") -> Dict[str, Any]:
        """Delete all user data (GDPR Article 17 - Right to Erasure)."""
        if not self.privacy_settings.right_to_deletion:
            raise ValueError("Right to deletion not enabled")
        
        deletion_summary = {
            "user_id": user_id,
            "deletion_timestamp": time.time(),
            "reason": reason,
            "consent_records_deleted": 0,
            "audit_events_anonymized": 0,
            "data_inventory_deleted": 0
        }
        
        # Delete consent records
        if user_id in self.consent_records:
            deletion_summary["consent_records_deleted"] = len(self.consent_records[user_id])
            del self.consent_records[user_id]
        
        # Anonymize audit events (can't delete for audit trail integrity)
        for event in self.audit_log:
            if event.user_id == user_id:
                event.user_id = "deleted_user"
                event.metadata = {"anonymized": True}
                deletion_summary["audit_events_anonymized"] += 1
        
        # Delete data inventory entries
        to_delete = []
        for data_id, inventory in self.data_inventory.items():
            if inventory.get("user_id") == user_id:
                to_delete.append(data_id)
        
        for data_id in to_delete:
            del self.data_inventory[data_id]
            deletion_summary["data_inventory_deleted"] += 1
        
        # Log the deletion
        self.log_compliance_event(
            event_type="data_deletion",
            data_classification=DataClassification.RESTRICTED,
            user_id=None,  # User already deleted
            data_processed=True,
            metadata=deletion_summary
        )
        
        return deletion_summary
    
    def run_compliance_audit(self) -> Dict[str, Any]:
        """Run comprehensive compliance audit."""
        audit_results = {
            "audit_timestamp": time.time(),
            "standards_evaluated": [s.value for s in self.enabled_standards],
            "overall_compliance": True,
            "standards_compliance": {},
            "recommendations": [],
            "risk_level": "low"
        }
        
        for standard in self.enabled_standards:
            compliance_result = self._audit_standard_compliance(standard)
            audit_results["standards_compliance"][standard.value] = compliance_result
            
            if not compliance_result["compliant"]:
                audit_results["overall_compliance"] = False
                audit_results["recommendations"].extend(compliance_result["recommendations"])
        
        # Determine risk level
        total_violations = sum(
            len(result.get("violations", []))
            for result in audit_results["standards_compliance"].values()
        )
        
        if total_violations == 0:
            audit_results["risk_level"] = "low"
        elif total_violations <= 5:
            audit_results["risk_level"] = "medium"
        else:
            audit_results["risk_level"] = "high"
        
        # Update compliance status
        for standard in self.enabled_standards:
            self.compliance_status[standard] = {
                "compliant": audit_results["standards_compliance"][standard.value]["compliant"],
                "issues": audit_results["standards_compliance"][standard.value].get("violations", []),
                "last_check": time.time()
            }
        
        # Log audit event
        self.log_compliance_event(
            event_type="compliance_audit",
            data_classification=DataClassification.INTERNAL,
            data_processed=True,
            metadata={
                "overall_compliance": audit_results["overall_compliance"],
                "risk_level": audit_results["risk_level"],
                "violations_count": total_violations
            }
        )
        
        return audit_results
    
    def _audit_standard_compliance(self, standard: ComplianceStandard) -> Dict[str, Any]:
        """Audit compliance for specific standard."""
        result = {
            "standard": standard.value,
            "compliant": True,
            "violations": [],
            "recommendations": [],
            "score": 100
        }
        
        if standard == ComplianceStandard.GDPR:
            result.update(self._audit_gdpr_compliance())
        elif standard == ComplianceStandard.CCPA:
            result.update(self._audit_ccpa_compliance())
        elif standard == ComplianceStandard.SOC2:
            result.update(self._audit_soc2_compliance())
        elif standard == ComplianceStandard.NIST_CSF:
            result.update(self._audit_nist_compliance())
        
        result["compliant"] = len(result["violations"]) == 0
        result["score"] = max(0, 100 - len(result["violations"]) * 10)
        
        return result
    
    def _audit_gdpr_compliance(self) -> Dict[str, Any]:
        """Audit GDPR compliance."""
        violations = []
        recommendations = []
        
        # Check consent management
        if not self.privacy_settings.consent_required:
            violations.append("GDPR Art. 6: Consent not required for data processing")
            recommendations.append("Enable consent requirement for all data processing")
        
        # Check right to deletion
        if not self.privacy_settings.right_to_deletion:
            violations.append("GDPR Art. 17: Right to erasure not implemented")
            recommendations.append("Implement right to deletion functionality")
        
        # Check data portability
        if not self.privacy_settings.data_portability:
            violations.append("GDPR Art. 20: Data portability not available")
            recommendations.append("Enable data export functionality")
        
        # Check data minimization
        if not self.privacy_settings.data_minimization:
            violations.append("GDPR Art. 5(1)(c): Data minimization not enforced")
            recommendations.append("Implement data minimization checks")
        
        # Check purpose limitation
        if not self.privacy_settings.purpose_limitation:
            violations.append("GDPR Art. 5(1)(b): Purpose limitation not enforced")
            recommendations.append("Implement purpose limitation validation")
        
        return {"violations": violations, "recommendations": recommendations}
    
    def _audit_ccpa_compliance(self) -> Dict[str, Any]:
        """Audit CCPA compliance."""
        violations = []
        recommendations = []
        
        # CCPA focuses on consumer rights
        if not self.privacy_settings.data_portability:
            violations.append("CCPA Sec. 1798.110: Right to know not fully implemented")
            recommendations.append("Enable comprehensive data export")
        
        if not self.privacy_settings.right_to_deletion:
            violations.append("CCPA Sec. 1798.105: Right to delete not implemented")
            recommendations.append("Implement data deletion functionality")
        
        return {"violations": violations, "recommendations": recommendations}
    
    def _audit_soc2_compliance(self) -> Dict[str, Any]:
        """Audit SOC2 compliance."""
        violations = []
        recommendations = []
        
        # SOC2 focuses on security controls
        audit_events_count = len(self.audit_log)
        if audit_events_count < 10:  # Arbitrary threshold
            violations.append("CC6.1: Insufficient audit logging")
            recommendations.append("Increase audit event logging coverage")
        
        # Check data encryption (simulated)
        if not self.privacy_settings.data_anonymization:
            violations.append("CC6.7: Data anonymization not enabled")
            recommendations.append("Enable data anonymization for sensitive data")
        
        return {"violations": violations, "recommendations": recommendations}
    
    def _audit_nist_compliance(self) -> Dict[str, Any]:
        """Audit NIST Cybersecurity Framework compliance."""
        violations = []
        recommendations = []
        
        # NIST CSF core functions check
        if not self.audit_log:
            violations.append("DE.CM-1: Monitoring insufficient")
            recommendations.append("Implement continuous monitoring")
        
        if not self.privacy_settings.data_anonymization:
            violations.append("PR.DS-5: Data protection insufficient")
            recommendations.append("Implement data protection measures")
        
        return {"violations": violations, "recommendations": recommendations}
    
    def _is_standard_applicable(self, standard: ComplianceStandard, classification: DataClassification, event_type: str) -> bool:
        """Check if compliance standard applies to event."""
        # All standards apply to restricted data
        if classification == DataClassification.RESTRICTED:
            return True
        
        # GDPR applies to personal data processing
        if standard == ComplianceStandard.GDPR and "user" in event_type:
            return True
        
        # SOC2 applies to all internal operations
        if standard == ComplianceStandard.SOC2 and classification in [DataClassification.INTERNAL, DataClassification.CONFIDENTIAL]:
            return True
        
        return False
    
    def _update_data_inventory(self, event: ComplianceEvent):
        """Update data inventory based on compliance event."""
        inventory_id = f"{event.session_id or 'system'}_{event.event_type}_{int(event.timestamp)}"
        
        self.data_inventory[inventory_id] = {
            "inventory_id": inventory_id,
            "user_id": event.user_id,
            "session_id": event.session_id,
            "data_type": event.event_type,
            "classification": event.data_classification.value,
            "created_at": event.timestamp,
            "retention_until": event.timestamp + self.retention_policies[event.data_classification].retention_days * 24 * 3600,
            "auto_delete": self.retention_policies[event.data_classification].auto_deletion
        }
    
    def _validate_event_compliance(self, event: ComplianceEvent):
        """Validate event against compliance requirements."""
        # Check if event requires consent but none exists
        if event.user_id and event.data_processed:
            if self.privacy_settings.consent_required:
                if not self._has_any_consent(event.user_id):
                    logger.warning(f"Data processing without consent: {event.event_id}")
    
    def _has_valid_consent(self, user_id: str, purpose: str, data_types: List[str]) -> bool:
        """Check if user has valid consent for data processing."""
        if user_id not in self.consent_records:
            return False
        
        for consent_id, consent in self.consent_records[user_id].items():
            if (consent["consent_given"] and 
                consent["purpose"] == purpose and
                consent["withdrawn_at"] is None and
                all(dtype in consent["data_types"] for dtype in data_types)):
                return True
        
        return False
    
    def _has_any_consent(self, user_id: str) -> bool:
        """Check if user has any valid consent."""
        if user_id not in self.consent_records:
            return False
        
        for consent_id, consent in self.consent_records[user_id].items():
            if consent["consent_given"] and consent["withdrawn_at"] is None:
                return True
        
        return False
    
    def _is_purpose_valid(self, purpose: str) -> bool:
        """Validate if purpose is within allowed purposes."""
        allowed_purposes = [
            "task_planning",
            "performance_optimization", 
            "security_monitoring",
            "compliance_audit"
        ]
        return purpose in allowed_purposes
    
    def _is_data_minimal(self, data_types: List[str], purpose: str) -> bool:
        """Check if data collection is minimal for purpose."""
        purpose_requirements = {
            "task_planning": ["task_id", "user_id", "session_id"],
            "performance_optimization": ["task_id", "performance_metrics"],
            "security_monitoring": ["user_id", "session_id", "security_events"],
            "compliance_audit": ["user_id", "audit_events"]
        }
        
        required = set(purpose_requirements.get(purpose, []))
        provided = set(data_types)
        
        # Data is minimal if it doesn't exceed requirements significantly
        return len(provided - required) <= 2
    
    def _schedule_data_deletion(self, user_id: str, consent_id: str):
        """Schedule data deletion after consent withdrawal."""
        # In a real implementation, this would schedule an async job
        logger.info(f"Scheduled data deletion for user {user_id}, consent {consent_id}")
    
    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get compliance dashboard data."""
        return {
            "overall_status": all(status["compliant"] for status in self.compliance_status.values()),
            "standards_status": {
                standard.value: status for standard, status in self.compliance_status.items()
            },
            "total_audit_events": len(self.audit_log),
            "active_consents": sum(
                sum(1 for consent in user_consents.values() 
                    if consent["consent_given"] and consent["withdrawn_at"] is None)
                for user_consents in self.consent_records.values()
            ),
            "data_inventory_size": len(self.data_inventory),
            "privacy_settings": asdict(self.privacy_settings),
            "retention_policies": {
                classification.value: asdict(policy) 
                for classification, policy in self.retention_policies.items()
            }
        }


# Global compliance manager
_compliance_manager = None


def get_compliance_manager(standards: Optional[List[ComplianceStandard]] = None) -> QuantumComplianceManager:
    """Get global compliance manager instance."""
    global _compliance_manager
    if _compliance_manager is None:
        _compliance_manager = QuantumComplianceManager(standards)
    return _compliance_manager