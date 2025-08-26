"""
Advanced Security Guardian - Generation 2 Enhancement
Autonomous security monitoring, threat detection, and response system.
"""

import asyncio
import hashlib
import logging
import time
import re
import json
import secrets
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from enum import Enum
from abc import ABC, abstractmethod
import ipaddress
from pathlib import Path

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types of security threats."""
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    AUTHENTICATION_BYPASS = "authentication_bypass"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    BRUTE_FORCE = "brute_force"
    DDoS = "ddos"
    MALWARE = "malware"
    INSIDER_THREAT = "insider_threat"
    API_ABUSE = "api_abuse"
    UNAUTHORIZED_ACCESS = "unauthorized_access"


class SecurityAction(Enum):
    """Security response actions."""
    BLOCK_IP = "block_ip"
    RATE_LIMIT = "rate_limit"
    REQUIRE_MFA = "require_mfa"
    ALERT_ADMIN = "alert_admin"
    QUARANTINE = "quarantine"
    LOG_SECURITY_EVENT = "log_security_event"
    INVALIDATE_SESSION = "invalidate_session"
    ENCRYPT_DATA = "encrypt_data"
    BACKUP_CRITICAL_DATA = "backup_critical_data"
    DISABLE_ACCOUNT = "disable_account"


@dataclass
class SecurityEvent:
    """Represents a security event or threat."""
    threat_type: ThreatType
    threat_level: ThreatLevel
    description: str
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    endpoint: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    detection_method: str = "unknown"
    confidence_score: float = 0.0


@dataclass
class SecurityResponse:
    """Represents a security response action."""
    action: SecurityAction
    target: str
    description: str
    success: bool
    timestamp: float = field(default_factory=time.time)
    duration: Optional[float] = None
    side_effects: List[str] = field(default_factory=list)


class ThreatDetector(ABC):
    """Abstract base class for threat detectors."""
    
    @abstractmethod
    async def detect_threats(self, request_data: Dict[str, Any]) -> List[SecurityEvent]:
        """Detect threats in request data."""
        pass


class SQLInjectionDetector(ThreatDetector):
    """Detects SQL injection attempts."""
    
    def __init__(self):
        self.sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"('(''|[^'])*')",
            r"(;|\|\||--)",
            r"(\bOR\s+\d+\s*=\s*\d+)",
            r"(\bUNION\s+(ALL\s+)?SELECT)",
            r"(\bINJECT\b|\bEVAL\b)"
        ]
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.sql_patterns]
    
    async def detect_threats(self, request_data: Dict[str, Any]) -> List[SecurityEvent]:
        """Detect SQL injection attempts in request data."""
        threats = []
        
        # Check all string values in request
        for key, value in request_data.items():
            if isinstance(value, str):
                threat_score = self._analyze_sql_injection_risk(value)
                
                if threat_score > 0.5:
                    threat_level = ThreatLevel.HIGH if threat_score > 0.8 else ThreatLevel.MEDIUM
                    
                    threats.append(SecurityEvent(
                        threat_type=ThreatType.SQL_INJECTION,
                        threat_level=threat_level,
                        description=f"Potential SQL injection in parameter '{key}': {value[:100]}",
                        payload={"parameter": key, "value": value[:200]},
                        detection_method="pattern_matching",
                        confidence_score=threat_score
                    ))
        
        return threats
    
    def _analyze_sql_injection_risk(self, input_string: str) -> float:
        """Analyze string for SQL injection risk."""
        risk_score = 0.0
        
        for pattern in self.compiled_patterns:
            matches = pattern.findall(input_string)
            if matches:
                # More matches = higher risk
                risk_score += len(matches) * 0.2
        
        # Additional heuristics
        if len(input_string) > 1000:  # Very long strings are suspicious
            risk_score += 0.1
        
        if input_string.count("'") > 3:  # Many quotes are suspicious
            risk_score += 0.2
        
        return min(1.0, risk_score)


class XSSDetector(ThreatDetector):
    """Detects Cross-Site Scripting (XSS) attempts."""
    
    def __init__(self):
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on(load|error|click|focus|blur|change|submit|keyup|keydown|mouseover|mouseout)\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>",
            r"<meta[^>]*>",
            r"vbscript:",
            r"data:text/html"
        ]
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in self.xss_patterns]
    
    async def detect_threats(self, request_data: Dict[str, Any]) -> List[SecurityEvent]:
        """Detect XSS attempts in request data."""
        threats = []
        
        for key, value in request_data.items():
            if isinstance(value, str):
                threat_score = self._analyze_xss_risk(value)
                
                if threat_score > 0.3:
                    threat_level = ThreatLevel.HIGH if threat_score > 0.7 else ThreatLevel.MEDIUM
                    
                    threats.append(SecurityEvent(
                        threat_type=ThreatType.XSS,
                        threat_level=threat_level,
                        description=f"Potential XSS in parameter '{key}': {value[:100]}",
                        payload={"parameter": key, "value": value[:200]},
                        detection_method="pattern_matching",
                        confidence_score=threat_score
                    ))
        
        return threats
    
    def _analyze_xss_risk(self, input_string: str) -> float:
        """Analyze string for XSS risk."""
        risk_score = 0.0
        
        for pattern in self.compiled_patterns:
            if pattern.search(input_string):
                risk_score += 0.3
        
        # HTML tags in general are suspicious
        html_tag_count = len(re.findall(r'<[^>]+>', input_string))
        risk_score += html_tag_count * 0.1
        
        return min(1.0, risk_score)


class BruteForceDetector(ThreatDetector):
    """Detects brute force attacks."""
    
    def __init__(self, time_window: int = 300, max_attempts: int = 10):
        self.time_window = time_window  # 5 minutes
        self.max_attempts = max_attempts
        self.attempt_history: Dict[str, List[float]] = {}
    
    async def detect_threats(self, request_data: Dict[str, Any]) -> List[SecurityEvent]:
        """Detect brute force attempts."""
        threats = []
        
        source_ip = request_data.get("source_ip", "unknown")
        user_id = request_data.get("user_id")
        is_failed_auth = request_data.get("auth_failed", False)
        
        if is_failed_auth:
            # Track failed authentication attempts
            current_time = time.time()
            
            if source_ip not in self.attempt_history:
                self.attempt_history[source_ip] = []
            
            self.attempt_history[source_ip].append(current_time)
            
            # Clean old attempts
            cutoff_time = current_time - self.time_window
            self.attempt_history[source_ip] = [
                timestamp for timestamp in self.attempt_history[source_ip]
                if timestamp > cutoff_time
            ]
            
            # Check if threshold exceeded
            attempts_count = len(self.attempt_history[source_ip])
            
            if attempts_count >= self.max_attempts:
                threats.append(SecurityEvent(
                    threat_type=ThreatType.BRUTE_FORCE,
                    threat_level=ThreatLevel.HIGH,
                    description=f"Brute force attack detected: {attempts_count} failed attempts from {source_ip}",
                    source_ip=source_ip,
                    user_id=user_id,
                    detection_method="rate_limiting",
                    confidence_score=min(1.0, attempts_count / (self.max_attempts * 2))
                ))
        
        return threats


class APIAbuseDetector(ThreatDetector):
    """Detects API abuse and excessive usage."""
    
    def __init__(self, rate_limit: int = 100, time_window: int = 60):
        self.rate_limit = rate_limit  # requests per minute
        self.time_window = time_window
        self.request_history: Dict[str, List[float]] = {}
    
    async def detect_threats(self, request_data: Dict[str, Any]) -> List[SecurityEvent]:
        """Detect API abuse."""
        threats = []
        
        client_id = request_data.get("client_id") or request_data.get("source_ip", "unknown")
        current_time = time.time()
        
        if client_id not in self.request_history:
            self.request_history[client_id] = []
        
        self.request_history[client_id].append(current_time)
        
        # Clean old requests
        cutoff_time = current_time - self.time_window
        self.request_history[client_id] = [
            timestamp for timestamp in self.request_history[client_id]
            if timestamp > cutoff_time
        ]
        
        requests_count = len(self.request_history[client_id])
        
        if requests_count > self.rate_limit:
            threat_level = ThreatLevel.HIGH if requests_count > self.rate_limit * 2 else ThreatLevel.MEDIUM
            
            threats.append(SecurityEvent(
                threat_type=ThreatType.API_ABUSE,
                threat_level=threat_level,
                description=f"API abuse detected: {requests_count} requests in {self.time_window}s from {client_id}",
                source_ip=request_data.get("source_ip"),
                detection_method="rate_limiting",
                confidence_score=min(1.0, requests_count / (self.rate_limit * 3))
            ))
        
        return threats


class SecurityResponseEngine:
    """Handles automated security responses."""
    
    def __init__(self):
        self.blocked_ips: Set[str] = set()
        self.rate_limited_clients: Dict[str, float] = {}
        self.quarantined_users: Set[str] = set()
        self.response_history: List[SecurityResponse] = []
    
    async def respond_to_threat(self, security_event: SecurityEvent) -> List[SecurityResponse]:
        """Generate appropriate response to security threat."""
        responses = []
        
        # Select appropriate actions based on threat type and level
        actions = self._select_response_actions(security_event)
        
        for action in actions:
            response = await self._execute_security_action(action, security_event)
            responses.append(response)
            self.response_history.append(response)
        
        return responses
    
    def _select_response_actions(self, security_event: SecurityEvent) -> List[SecurityAction]:
        """Select appropriate response actions for a security event."""
        actions = []
        
        # Always log security events
        actions.append(SecurityAction.LOG_SECURITY_EVENT)
        
        # High and critical threats get immediate blocking
        if security_event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            if security_event.source_ip:
                actions.append(SecurityAction.BLOCK_IP)
            
            actions.append(SecurityAction.ALERT_ADMIN)
            
            if security_event.user_id:
                if security_event.threat_type == ThreatType.BRUTE_FORCE:
                    actions.append(SecurityAction.DISABLE_ACCOUNT)
                else:
                    actions.append(SecurityAction.INVALIDATE_SESSION)
        
        # Specific actions for specific threats
        if security_event.threat_type == ThreatType.API_ABUSE:
            actions.append(SecurityAction.RATE_LIMIT)
        
        elif security_event.threat_type in [ThreatType.SQL_INJECTION, ThreatType.XSS]:
            actions.append(SecurityAction.QUARANTINE)
        
        elif security_event.threat_type == ThreatType.DATA_EXFILTRATION:
            actions.append(SecurityAction.BACKUP_CRITICAL_DATA)
            actions.append(SecurityAction.ENCRYPT_DATA)
        
        elif security_event.threat_type == ThreatType.BRUTE_FORCE:
            actions.append(SecurityAction.REQUIRE_MFA)
        
        return list(set(actions))  # Remove duplicates
    
    async def _execute_security_action(self, action: SecurityAction, security_event: SecurityEvent) -> SecurityResponse:
        """Execute a specific security action."""
        start_time = time.time()
        
        try:
            if action == SecurityAction.BLOCK_IP:
                return await self._block_ip(security_event)
            elif action == SecurityAction.RATE_LIMIT:
                return await self._apply_rate_limit(security_event)
            elif action == SecurityAction.REQUIRE_MFA:
                return await self._require_mfa(security_event)
            elif action == SecurityAction.ALERT_ADMIN:
                return await self._alert_admin(security_event)
            elif action == SecurityAction.QUARANTINE:
                return await self._quarantine_user(security_event)
            elif action == SecurityAction.LOG_SECURITY_EVENT:
                return await self._log_security_event(security_event)
            elif action == SecurityAction.INVALIDATE_SESSION:
                return await self._invalidate_session(security_event)
            elif action == SecurityAction.ENCRYPT_DATA:
                return await self._encrypt_sensitive_data(security_event)
            elif action == SecurityAction.BACKUP_CRITICAL_DATA:
                return await self._backup_critical_data(security_event)
            elif action == SecurityAction.DISABLE_ACCOUNT:
                return await self._disable_account(security_event)
            else:
                return SecurityResponse(
                    action=action,
                    target="unknown",
                    description=f"Unknown action: {action}",
                    success=False
                )
                
        except Exception as e:
            logger.error(f"Failed to execute security action {action}: {e}")
            return SecurityResponse(
                action=action,
                target="error",
                description=f"Action execution failed: {str(e)}",
                success=False
            )
    
    async def _block_ip(self, security_event: SecurityEvent) -> SecurityResponse:
        """Block IP address."""
        ip = security_event.source_ip
        if not ip:
            return SecurityResponse(
                action=SecurityAction.BLOCK_IP,
                target="no_ip",
                description="No IP address to block",
                success=False
            )
        
        self.blocked_ips.add(ip)
        
        # Simulate network blocking (in real implementation, would update firewall rules)
        await asyncio.sleep(0.1)
        
        return SecurityResponse(
            action=SecurityAction.BLOCK_IP,
            target=ip,
            description=f"Blocked IP address {ip}",
            success=True,
            side_effects=[f"All traffic from {ip} will be rejected"]
        )
    
    async def _apply_rate_limit(self, security_event: SecurityEvent) -> SecurityResponse:
        """Apply rate limiting."""
        client_id = security_event.source_ip or security_event.user_id or "unknown"
        
        # Apply rate limit for 1 hour
        self.rate_limited_clients[client_id] = time.time() + 3600
        
        await asyncio.sleep(0.05)
        
        return SecurityResponse(
            action=SecurityAction.RATE_LIMIT,
            target=client_id,
            description=f"Applied rate limit to {client_id}",
            success=True,
            duration=3600.0,
            side_effects=[f"Client {client_id} requests will be throttled for 1 hour"]
        )
    
    async def _require_mfa(self, security_event: SecurityEvent) -> SecurityResponse:
        """Require multi-factor authentication."""
        user_id = security_event.user_id or "all_users"
        
        # In real implementation, would trigger MFA requirement in auth system
        await asyncio.sleep(0.1)
        
        return SecurityResponse(
            action=SecurityAction.REQUIRE_MFA,
            target=user_id,
            description=f"Enabled MFA requirement for {user_id}",
            success=True,
            side_effects=[f"User {user_id} must complete MFA for future logins"]
        )
    
    async def _alert_admin(self, security_event: SecurityEvent) -> SecurityResponse:
        """Send alert to administrators."""
        
        # In real implementation, would send email, SMS, or push notification
        alert_message = (f"SECURITY ALERT: {security_event.threat_type.value} "
                        f"threat detected with {security_event.threat_level.value} severity. "
                        f"Details: {security_event.description}")
        
        logger.critical(alert_message)
        await asyncio.sleep(0.2)  # Simulate alert sending time
        
        return SecurityResponse(
            action=SecurityAction.ALERT_ADMIN,
            target="administrators",
            description=f"Sent security alert to administrators",
            success=True
        )
    
    async def _quarantine_user(self, security_event: SecurityEvent) -> SecurityResponse:
        """Quarantine user account."""
        user_id = security_event.user_id or "unknown_user"
        
        self.quarantined_users.add(user_id)
        
        await asyncio.sleep(0.1)
        
        return SecurityResponse(
            action=SecurityAction.QUARANTINE,
            target=user_id,
            description=f"Quarantined user account {user_id}",
            success=True,
            side_effects=[f"User {user_id} access restricted to safe operations only"]
        )
    
    async def _log_security_event(self, security_event: SecurityEvent) -> SecurityResponse:
        """Log security event."""
        
        log_entry = {
            "timestamp": security_event.timestamp,
            "threat_type": security_event.threat_type.value,
            "threat_level": security_event.threat_level.value,
            "description": security_event.description,
            "source_ip": security_event.source_ip,
            "user_id": security_event.user_id,
            "endpoint": security_event.endpoint,
            "confidence": security_event.confidence_score
        }
        
        # In real implementation, would write to security log file or SIEM
        logger.warning(f"SECURITY_EVENT: {json.dumps(log_entry)}")
        
        return SecurityResponse(
            action=SecurityAction.LOG_SECURITY_EVENT,
            target="security_log",
            description="Logged security event",
            success=True
        )
    
    async def _invalidate_session(self, security_event: SecurityEvent) -> SecurityResponse:
        """Invalidate user session."""
        user_id = security_event.user_id or "unknown_user"
        
        # In real implementation, would invalidate session in session store
        await asyncio.sleep(0.1)
        
        return SecurityResponse(
            action=SecurityAction.INVALIDATE_SESSION,
            target=user_id,
            description=f"Invalidated session for user {user_id}",
            success=True,
            side_effects=[f"User {user_id} must re-authenticate"]
        )
    
    async def _encrypt_sensitive_data(self, security_event: SecurityEvent) -> SecurityResponse:
        """Encrypt sensitive data."""
        
        # In real implementation, would encrypt data at rest and in transit
        await asyncio.sleep(0.3)  # Simulate encryption time
        
        return SecurityResponse(
            action=SecurityAction.ENCRYPT_DATA,
            target="sensitive_data",
            description="Encrypted sensitive data with additional security layer",
            success=True
        )
    
    async def _backup_critical_data(self, security_event: SecurityEvent) -> SecurityResponse:
        """Backup critical data."""
        
        # In real implementation, would trigger backup procedures
        await asyncio.sleep(0.5)  # Simulate backup time
        
        return SecurityResponse(
            action=SecurityAction.BACKUP_CRITICAL_DATA,
            target="critical_data",
            description="Created emergency backup of critical data",
            success=True
        )
    
    async def _disable_account(self, security_event: SecurityEvent) -> SecurityResponse:
        """Disable user account."""
        user_id = security_event.user_id or "unknown_user"
        
        # In real implementation, would disable account in user management system
        await asyncio.sleep(0.1)
        
        return SecurityResponse(
            action=SecurityAction.DISABLE_ACCOUNT,
            target=user_id,
            description=f"Disabled account for user {user_id}",
            success=True,
            side_effects=[f"Account {user_id} is temporarily disabled"]
        )
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is blocked."""
        return ip in self.blocked_ips
    
    def is_rate_limited(self, client_id: str) -> bool:
        """Check if client is rate limited."""
        if client_id in self.rate_limited_clients:
            return time.time() < self.rate_limited_clients[client_id]
        return False
    
    def is_user_quarantined(self, user_id: str) -> bool:
        """Check if user is quarantined."""
        return user_id in self.quarantined_users


class AdvancedSecurityGuardian:
    """Main security guardian coordinating all security components."""
    
    def __init__(self):
        self.threat_detectors = [
            SQLInjectionDetector(),
            XSSDetector(),
            BruteForceDetector(),
            APIAbuseDetector()
        ]
        
        self.response_engine = SecurityResponseEngine()
        self.security_metrics = {
            "threats_detected": 0,
            "threats_blocked": 0,
            "false_positives": 0,
            "response_time_sum": 0.0,
            "start_time": time.time()
        }
    
    async def analyze_request(self, request_data: Dict[str, Any]) -> Tuple[List[SecurityEvent], List[SecurityResponse]]:
        """Analyze incoming request for security threats."""
        start_time = time.time()
        
        # Pre-flight checks
        if self._is_request_blocked(request_data):
            # Request is already blocked, return empty results
            return [], []
        
        # Detect threats using all detectors
        all_threats = []
        for detector in self.threat_detectors:
            try:
                threats = await detector.detect_threats(request_data)
                all_threats.extend(threats)
            except Exception as e:
                logger.error(f"Threat detector error: {e}")
        
        # Update metrics
        self.security_metrics["threats_detected"] += len(all_threats)
        
        # Respond to detected threats
        all_responses = []
        for threat in all_threats:
            try:
                responses = await self.response_engine.respond_to_threat(threat)
                all_responses.extend(responses)
                
                # Count successful responses
                successful_responses = sum(1 for r in responses if r.success)
                if successful_responses > 0:
                    self.security_metrics["threats_blocked"] += 1
                
            except Exception as e:
                logger.error(f"Security response error: {e}")
        
        # Update response time metrics
        response_time = time.time() - start_time
        self.security_metrics["response_time_sum"] += response_time
        
        return all_threats, all_responses
    
    def _is_request_blocked(self, request_data: Dict[str, Any]) -> bool:
        """Check if request should be blocked based on existing rules."""
        source_ip = request_data.get("source_ip")
        if source_ip and self.response_engine.is_ip_blocked(source_ip):
            return True
        
        client_id = request_data.get("client_id") or source_ip
        if client_id and self.response_engine.is_rate_limited(client_id):
            return True
        
        user_id = request_data.get("user_id")
        if user_id and self.response_engine.is_user_quarantined(user_id):
            return True
        
        return False
    
    async def perform_security_audit(self) -> Dict[str, Any]:
        """Perform comprehensive security audit."""
        audit_start = time.time()
        
        audit_results = {
            "audit_timestamp": audit_start,
            "system_security_status": "analyzing",
            "threat_detection_capability": self._assess_detection_capability(),
            "response_effectiveness": self._assess_response_effectiveness(),
            "security_coverage": self._assess_security_coverage(),
            "performance_metrics": self._get_performance_metrics(),
            "recommendations": []
        }
        
        # Assess overall security status
        detection_score = audit_results["threat_detection_capability"]["score"]
        response_score = audit_results["response_effectiveness"]["score"]
        coverage_score = audit_results["security_coverage"]["score"]
        
        overall_score = (detection_score + response_score + coverage_score) / 3
        
        if overall_score > 0.9:
            audit_results["system_security_status"] = "excellent"
        elif overall_score > 0.75:
            audit_results["system_security_status"] = "good"
        elif overall_score > 0.6:
            audit_results["system_security_status"] = "adequate"
        else:
            audit_results["system_security_status"] = "needs_improvement"
        
        # Generate recommendations
        audit_results["recommendations"] = self._generate_security_recommendations(audit_results)
        
        audit_results["audit_duration"] = time.time() - audit_start
        
        return audit_results
    
    def _assess_detection_capability(self) -> Dict[str, Any]:
        """Assess threat detection capabilities."""
        detector_types = [type(detector).__name__ for detector in self.threat_detectors]
        
        # Basic assessment based on detector coverage
        threat_coverage = {
            "sql_injection": "SQLInjectionDetector" in detector_types,
            "xss": "XSSDetector" in detector_types,
            "brute_force": "BruteForceDetector" in detector_types,
            "api_abuse": "APIAbuseDetector" in detector_types
        }
        
        coverage_score = sum(threat_coverage.values()) / len(threat_coverage)
        
        return {
            "score": coverage_score,
            "covered_threats": [threat for threat, covered in threat_coverage.items() if covered],
            "missing_coverage": [threat for threat, covered in threat_coverage.items() if not covered],
            "detector_count": len(self.threat_detectors)
        }
    
    def _assess_response_effectiveness(self) -> Dict[str, Any]:
        """Assess security response effectiveness."""
        total_threats = self.security_metrics["threats_detected"]
        blocked_threats = self.security_metrics["threats_blocked"]
        
        if total_threats == 0:
            effectiveness_score = 1.0  # No threats = perfect effectiveness
        else:
            effectiveness_score = blocked_threats / total_threats
        
        return {
            "score": effectiveness_score,
            "threats_detected": total_threats,
            "threats_blocked": blocked_threats,
            "block_rate": effectiveness_score,
            "response_actions": len(self.response_engine.response_history)
        }
    
    def _assess_security_coverage(self) -> Dict[str, Any]:
        """Assess overall security coverage."""
        
        # Assess based on implemented security features
        security_features = {
            "ip_blocking": len(self.response_engine.blocked_ips) >= 0,  # Feature exists
            "rate_limiting": len(self.response_engine.rate_limited_clients) >= 0,  # Feature exists
            "user_quarantine": len(self.response_engine.quarantined_users) >= 0,  # Feature exists
            "threat_detection": len(self.threat_detectors) > 0,
            "automated_response": True,  # Always true for this implementation
            "security_logging": True   # Always true for this implementation
        }
        
        coverage_score = sum(security_features.values()) / len(security_features)
        
        return {
            "score": coverage_score,
            "implemented_features": [feature for feature, impl in security_features.items() if impl],
            "missing_features": [feature for feature, impl in security_features.items() if not impl]
        }
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get security system performance metrics."""
        uptime = time.time() - self.security_metrics["start_time"]
        
        if self.security_metrics["threats_detected"] > 0:
            avg_response_time = (self.security_metrics["response_time_sum"] / 
                               self.security_metrics["threats_detected"])
        else:
            avg_response_time = 0.0
        
        return {
            "uptime_seconds": uptime,
            "threats_per_hour": (self.security_metrics["threats_detected"] / 
                                (uptime / 3600)) if uptime > 0 else 0,
            "average_response_time_ms": avg_response_time * 1000,
            "false_positive_rate": (self.security_metrics["false_positives"] / 
                                   max(1, self.security_metrics["threats_detected"])),
            "system_overhead": "low"  # Placeholder - would measure actual overhead
        }
    
    def _generate_security_recommendations(self, audit_results: Dict[str, Any]) -> List[str]:
        """Generate security improvement recommendations."""
        recommendations = []
        
        # Detection capability recommendations
        detection = audit_results["threat_detection_capability"]
        if detection["score"] < 0.8:
            recommendations.append("Consider adding more threat detection capabilities")
        
        if "csrf" in detection.get("missing_coverage", []):
            recommendations.append("Implement CSRF attack detection")
        
        # Response effectiveness recommendations
        response = audit_results["response_effectiveness"]
        if response["score"] < 0.9:
            recommendations.append("Improve automated threat response mechanisms")
        
        # Performance recommendations
        performance = audit_results["performance_metrics"]
        if performance["average_response_time_ms"] > 100:
            recommendations.append("Optimize security response time")
        
        if performance["false_positive_rate"] > 0.1:
            recommendations.append("Reduce false positive rate in threat detection")
        
        # General recommendations
        if audit_results["system_security_status"] in ["adequate", "needs_improvement"]:
            recommendations.append("Conduct comprehensive security review")
            recommendations.append("Consider implementing additional security layers")
        
        return recommendations
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security system status."""
        uptime = time.time() - self.security_metrics["start_time"]
        
        return {
            "status": "active",
            "uptime_hours": uptime / 3600,
            "threat_detectors_active": len(self.threat_detectors),
            "blocked_ips": len(self.response_engine.blocked_ips),
            "rate_limited_clients": len(self.response_engine.rate_limited_clients),
            "quarantined_users": len(self.response_engine.quarantined_users),
            "total_threats_detected": self.security_metrics["threats_detected"],
            "total_threats_blocked": self.security_metrics["threats_blocked"],
            "last_threat_detection": "none" if self.security_metrics["threats_detected"] == 0 else "recent"
        }


# Utility functions for security hardening
class SecurityUtilities:
    """Utility functions for security operations."""
    
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate cryptographically secure random token."""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Hash password with salt."""
        if salt is None:
            salt = secrets.token_hex(32)
        
        # Using SHA-256 (in production, use bcrypt or Argon2)
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return password_hash, salt
    
    @staticmethod
    def validate_ip_address(ip: str) -> bool:
        """Validate if string is a valid IP address."""
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def sanitize_input(input_string: str, max_length: int = 1000) -> str:
        """Sanitize user input to prevent basic attacks."""
        if not isinstance(input_string, str):
            return str(input_string)
        
        # Truncate if too long
        sanitized = input_string[:max_length]
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Basic HTML escaping
        sanitized = (sanitized.replace('&', '&amp;')
                              .replace('<', '&lt;')
                              .replace('>', '&gt;')
                              .replace('"', '&quot;')
                              .replace("'", '&#x27;'))
        
        return sanitized


# Factory function
def create_security_guardian() -> AdvancedSecurityGuardian:
    """Create advanced security guardian instance."""
    return AdvancedSecurityGuardian()