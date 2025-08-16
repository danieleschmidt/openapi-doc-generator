"""Global-first deployment system with multi-region and compliance support."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from .i18n import ComplianceRegion, SupportedLanguage, get_i18n_manager
from .quantum_audit_logger import AuditEventType, get_audit_logger
from .quantum_health_monitor import get_health_monitor
from .quantum_performance_optimizer import get_performance_optimizer
from .quantum_quality_gates import QualityResult, get_quality_gates
from .quantum_security import SecurityLevel


class DeploymentStage(Enum):
    """Deployment pipeline stages."""
    BUILD = "build"
    TEST = "test"
    SECURITY_SCAN = "security_scan"
    QUALITY_GATES = "quality_gates"
    STAGING = "staging"
    CANARY = "canary"
    PRODUCTION = "production"
    POST_DEPLOYMENT = "post_deployment"


class DeploymentRegion(Enum):
    """Global deployment regions."""
    US_EAST = "us-east"
    US_WEST = "us-west"
    EU_WEST = "eu-west"
    EU_CENTRAL = "eu-central"
    ASIA_PACIFIC = "asia-pacific"
    CANADA_CENTRAL = "canada-central"
    AUSTRALIA = "australia"
    BRAZIL = "brazil"


@dataclass
class DeploymentConfig:
    """Global deployment configuration."""
    target_regions: List[DeploymentRegion]
    supported_languages: List[SupportedLanguage]
    compliance_regions: List[ComplianceRegion]
    enable_canary: bool = True
    canary_traffic_percent: float = 5.0
    rollback_on_errors: bool = True
    health_check_timeout: int = 300
    performance_threshold: float = 2000.0  # ms
    quality_gate_required: bool = True


@dataclass
class RegionDeploymentStatus:
    """Status of deployment in a specific region."""
    region: DeploymentRegion
    stage: DeploymentStage
    status: str  # success, failed, in_progress, pending
    start_time: float
    end_time: Optional[float]
    health_score: float
    performance_metrics: Dict[str, Any]
    compliance_status: Dict[str, bool]
    error_message: Optional[str] = None


@dataclass
class GlobalDeploymentResult:
    """Result of global deployment."""
    deployment_id: str
    overall_status: str
    start_time: float
    end_time: Optional[float]
    region_statuses: List[RegionDeploymentStatus]
    quality_gate_result: Optional[QualityResult]
    deployment_config: DeploymentConfig
    rollback_executed: bool = False
    metadata: Dict[str, Any] = None


class QuantumGlobalDeployment:
    """Advanced global deployment system with autonomous operations."""

    def __init__(self):
        """Initialize global deployment system."""
        # Core dependencies
        self.audit_logger = get_audit_logger()
        self.health_monitor = get_health_monitor()
        self.performance_optimizer = get_performance_optimizer()
        self.quality_gates = get_quality_gates()
        self.i18n_manager = get_i18n_manager()

        self.logger = logging.getLogger(__name__)

        # Deployment state
        self.active_deployments: Dict[str, GlobalDeploymentResult] = {}
        self.deployment_history: List[GlobalDeploymentResult] = []

        # Region-specific configurations
        self.region_configs = {
            DeploymentRegion.US_EAST: {
                "primary_language": SupportedLanguage.ENGLISH,
                "compliance": [ComplianceRegion.CCPA],
                "performance_target": 1500.0
            },
            DeploymentRegion.EU_WEST: {
                "primary_language": SupportedLanguage.ENGLISH,
                "compliance": [ComplianceRegion.GDPR],
                "performance_target": 2000.0
            },
            DeploymentRegion.EU_CENTRAL: {
                "primary_language": SupportedLanguage.GERMAN,
                "compliance": [ComplianceRegion.GDPR],
                "performance_target": 1800.0
            },
            DeploymentRegion.ASIA_PACIFIC: {
                "primary_language": SupportedLanguage.JAPANESE,
                "compliance": [ComplianceRegion.PDPA_SINGAPORE],
                "performance_target": 2500.0
            },
            DeploymentRegion.CANADA_CENTRAL: {
                "primary_language": SupportedLanguage.ENGLISH,
                "compliance": [ComplianceRegion.PIPEDA],
                "performance_target": 1600.0
            },
            DeploymentRegion.AUSTRALIA: {
                "primary_language": SupportedLanguage.ENGLISH,
                "compliance": [ComplianceRegion.PDPA_SINGAPORE],
                "performance_target": 2200.0
            },
            DeploymentRegion.BRAZIL: {
                "primary_language": SupportedLanguage.SPANISH,
                "compliance": [ComplianceRegion.LGPD],
                "performance_target": 2800.0
            }
        }

    def deploy_globally(self,
                       deployment_config: DeploymentConfig,
                       deployment_id: Optional[str] = None) -> GlobalDeploymentResult:
        """Execute global deployment with autonomous operations."""
        deployment_id = deployment_id or f"deploy_{int(time.time())}"
        start_time = time.time()

        self.logger.info(f"Starting global deployment: {deployment_id}")

        # Create deployment result
        deployment_result = GlobalDeploymentResult(
            deployment_id=deployment_id,
            overall_status="in_progress",
            start_time=start_time,
            end_time=None,
            region_statuses=[],
            quality_gate_result=None,
            deployment_config=deployment_config,
            metadata={}
        )

        # Store active deployment
        self.active_deployments[deployment_id] = deployment_result

        # Log deployment start
        self.audit_logger.log_security_event(
            event_type=AuditEventType.SYSTEM_ACCESS,
            action="global_deployment_start",
            result="initiated",
            severity=SecurityLevel.MEDIUM,
            details={
                "deployment_id": deployment_id,
                "target_regions": [r.value for r in deployment_config.target_regions],
                "supported_languages": [l.value for l in deployment_config.supported_languages]
            }
        )

        try:
            # Stage 1: Quality Gates
            if deployment_config.quality_gate_required:
                quality_result = self._execute_quality_gates()
                deployment_result.quality_gate_result = quality_result.overall_result

                if quality_result.overall_result == QualityResult.FAIL:
                    deployment_result.overall_status = "failed"
                    deployment_result.end_time = time.time()

                    self.logger.error(f"Deployment {deployment_id} failed quality gates")
                    return deployment_result

            # Stage 2: Pre-deployment validation
            self._validate_deployment_requirements(deployment_config)

            # Stage 3: Regional deployments
            for region in deployment_config.target_regions:
                region_status = self._deploy_to_region(
                    region, deployment_config, deployment_id
                )
                deployment_result.region_statuses.append(region_status)

                # Check for failures and rollback if configured
                if (region_status.status == "failed" and
                    deployment_config.rollback_on_errors):

                    self.logger.warning(f"Region {region.value} failed, initiating rollback")
                    self._execute_rollback(deployment_result)
                    deployment_result.rollback_executed = True
                    break

            # Stage 4: Post-deployment validation
            self._post_deployment_validation(deployment_result)

            # Determine overall status
            failed_regions = [r for r in deployment_result.region_statuses if r.status == "failed"]
            if failed_regions:
                deployment_result.overall_status = "partial_success"
            else:
                deployment_result.overall_status = "success"

            deployment_result.end_time = time.time()

            # Store in history
            self.deployment_history.append(deployment_result)
            if len(self.deployment_history) > 100:
                self.deployment_history.pop(0)

            # Clean up active deployments
            del self.active_deployments[deployment_id]

            duration = (deployment_result.end_time - deployment_result.start_time) / 60
            self.logger.info(
                f"Global deployment {deployment_id} completed in {duration:.2f} minutes. "
                f"Status: {deployment_result.overall_status}"
            )

            # Log completion
            self.audit_logger.log_security_event(
                event_type=AuditEventType.SYSTEM_ACCESS,
                action="global_deployment_complete",
                result="success" if deployment_result.overall_status == "success" else "partial",
                severity=SecurityLevel.LOW,
                details={
                    "deployment_id": deployment_id,
                    "duration_minutes": duration,
                    "regions_deployed": len(deployment_result.region_statuses),
                    "failed_regions": len(failed_regions)
                }
            )

            return deployment_result

        except Exception as e:
            self.logger.error(f"Global deployment {deployment_id} failed: {e}")

            deployment_result.overall_status = "failed"
            deployment_result.end_time = time.time()
            deployment_result.metadata = {"error": str(e)}

            # Log failure
            self.audit_logger.log_security_event(
                event_type=AuditEventType.ERROR_CONDITION,
                action="global_deployment_failed",
                result="error",
                severity=SecurityLevel.HIGH,
                details={
                    "deployment_id": deployment_id,
                    "error": str(e)
                }
            )

            # Clean up
            if deployment_id in self.active_deployments:
                del self.active_deployments[deployment_id]

            return deployment_result

    def _execute_quality_gates(self) -> Any:
        """Execute quality gates before deployment."""
        self.logger.info("Executing quality gates for deployment")

        quality_report = self.quality_gates.execute_all_gates()

        if not quality_report.deployment_ready:
            self.audit_logger.log_security_event(
                event_type=AuditEventType.ERROR_CONDITION,
                action="quality_gates_failed",
                result="blocked_deployment",
                severity=SecurityLevel.HIGH,
                details={
                    "overall_score": quality_report.overall_score,
                    "critical_issues": len(quality_report.critical_issues)
                }
            )

        return quality_report

    def _validate_deployment_requirements(self,
                                        config: DeploymentConfig) -> None:
        """Validate deployment requirements and dependencies."""
        self.logger.info("Validating deployment requirements")

        # Check system health
        health = self.health_monitor.get_system_health()
        if health.overall_status.value in ['critical', 'unknown']:
            raise RuntimeError(f"System health check failed: {health.overall_status.value}")

        # Validate i18n support for target languages
        for language in config.supported_languages:
            if not self.i18n_manager.is_language_supported(language):
                self.logger.warning(f"Language {language.value} may not be fully supported")

        # Validate compliance requirements
        for region in config.target_regions:
            region_config = self.region_configs.get(region, {})
            required_compliance = region_config.get('compliance', [])

            for compliance in required_compliance:
                if compliance not in config.compliance_regions:
                    raise ValueError(f"Region {region.value} requires compliance: {compliance.value}")

        self.logger.info("Deployment requirements validated successfully")

    def _deploy_to_region(self,
                         region: DeploymentRegion,
                         config: DeploymentConfig,
                         deployment_id: str) -> RegionDeploymentStatus:
        """Deploy to specific region with localization and compliance."""
        start_time = time.time()

        self.logger.info(f"Deploying to region: {region.value}")

        region_status = RegionDeploymentStatus(
            region=region,
            stage=DeploymentStage.BUILD,
            status="in_progress",
            start_time=start_time,
            end_time=None,
            health_score=0.0,
            performance_metrics={},
            compliance_status={}
        )

        try:
            # Get region-specific configuration
            region_config = self.region_configs.get(region, {})
            primary_language = region_config.get('primary_language', SupportedLanguage.ENGLISH)
            performance_target = region_config.get('performance_target', 2000.0)

            # Stage 1: Build with localization
            region_status.stage = DeploymentStage.BUILD
            self._build_for_region(region, primary_language)

            # Stage 2: Security and compliance checks
            region_status.stage = DeploymentStage.SECURITY_SCAN
            compliance_results = self._validate_regional_compliance(region, config.compliance_regions)
            region_status.compliance_status = compliance_results

            # Stage 3: Performance validation
            region_status.stage = DeploymentStage.TEST
            performance_metrics = self._validate_regional_performance(region, performance_target)
            region_status.performance_metrics = performance_metrics

            # Stage 4: Canary deployment (if enabled)
            if config.enable_canary:
                region_status.stage = DeploymentStage.CANARY
                self._canary_deployment(region, config.canary_traffic_percent)

            # Stage 5: Full production deployment
            region_status.stage = DeploymentStage.PRODUCTION
            self._production_deployment(region)

            # Stage 6: Post-deployment health check
            region_status.stage = DeploymentStage.POST_DEPLOYMENT
            health_score = self._post_deployment_health_check(region)
            region_status.health_score = health_score

            # Success
            region_status.status = "success"
            region_status.end_time = time.time()

            duration = (region_status.end_time - region_status.start_time) / 60
            self.logger.info(f"Region {region.value} deployed successfully in {duration:.2f} minutes")

            # Log regional deployment success
            self.audit_logger.log_security_event(
                event_type=AuditEventType.SYSTEM_ACCESS,
                action="region_deployment_success",
                result="success",
                severity=SecurityLevel.LOW,
                details={
                    "deployment_id": deployment_id,
                    "region": region.value,
                    "duration_minutes": duration,
                    "health_score": health_score
                }
            )

            return region_status

        except Exception as e:
            self.logger.error(f"Region {region.value} deployment failed: {e}")

            region_status.status = "failed"
            region_status.end_time = time.time()
            region_status.error_message = str(e)

            # Log regional deployment failure
            self.audit_logger.log_security_event(
                event_type=AuditEventType.ERROR_CONDITION,
                action="region_deployment_failed",
                result="error",
                severity=SecurityLevel.HIGH,
                details={
                    "deployment_id": deployment_id,
                    "region": region.value,
                    "error": str(e)
                }
            )

            return region_status

    def _build_for_region(self,
                         region: DeploymentRegion,
                         primary_language: SupportedLanguage) -> None:
        """Build application with region-specific localization."""
        self.logger.info(f"Building for region {region.value} with language {primary_language.value}")

        # Simulate localization build process
        time.sleep(0.1)  # Simulate build time

        # Validate translations exist
        if not self.i18n_manager.is_language_supported(primary_language):
            self.logger.warning(f"Limited translation support for {primary_language.value}")

    def _validate_regional_compliance(self,
                                    region: DeploymentRegion,
                                    compliance_regions: List[ComplianceRegion]) -> Dict[str, bool]:
        """Validate compliance requirements for region."""
        self.logger.info(f"Validating compliance for region {region.value}")

        region_config = self.region_configs.get(region, {})
        required_compliance = region_config.get('compliance', [])

        compliance_results = {}

        for compliance in required_compliance:
            is_compliant = compliance in compliance_regions
            compliance_results[compliance.value] = is_compliant

            if not is_compliant:
                raise ValueError(f"Compliance requirement not met: {compliance.value}")

        return compliance_results

    def _validate_regional_performance(self,
                                     region: DeploymentRegion,
                                     target_performance: float) -> Dict[str, Any]:
        """Validate performance requirements for region."""
        self.logger.info(f"Validating performance for region {region.value}")

        # Get performance statistics
        perf_stats = self.performance_optimizer.get_performance_stats()

        # Simulate region-specific performance test
        simulated_latency = target_performance * 0.8  # 20% better than target

        metrics = {
            "target_latency_ms": target_performance,
            "actual_latency_ms": simulated_latency,
            "performance_ratio": target_performance / simulated_latency,
            "meets_target": simulated_latency <= target_performance
        }

        if not metrics["meets_target"]:
            raise RuntimeError(f"Performance target not met: {simulated_latency}ms > {target_performance}ms")

        return metrics

    def _canary_deployment(self,
                          region: DeploymentRegion,
                          traffic_percent: float) -> None:
        """Execute canary deployment for region."""
        self.logger.info(f"Canary deployment to {region.value} with {traffic_percent}% traffic")

        # Simulate canary deployment
        time.sleep(0.05)

        # Simulate canary health check
        canary_health = 95.0  # Simulate healthy canary

        if canary_health < 90.0:
            raise RuntimeError(f"Canary deployment unhealthy: {canary_health}%")

    def _production_deployment(self, region: DeploymentRegion) -> None:
        """Execute full production deployment for region."""
        self.logger.info(f"Production deployment to {region.value}")

        # Simulate production deployment
        time.sleep(0.1)

    def _post_deployment_health_check(self, region: DeploymentRegion) -> float:
        """Execute post-deployment health check for region."""
        self.logger.info(f"Post-deployment health check for {region.value}")

        # Get system health
        health = self.health_monitor.get_system_health()

        # Convert health status to score
        health_scores = {
            'healthy': 100.0,
            'degraded': 75.0,
            'critical': 25.0,
            'unknown': 50.0
        }

        return health_scores.get(health.overall_status.value, 50.0)

    def _post_deployment_validation(self, deployment_result: GlobalDeploymentResult) -> None:
        """Execute post-deployment validation across all regions."""
        self.logger.info("Executing post-deployment validation")

        # Check overall health across regions
        avg_health = sum(r.health_score for r in deployment_result.region_statuses) / len(deployment_result.region_statuses)

        if avg_health < 80.0:
            self.logger.warning(f"Average health score below threshold: {avg_health:.1f}%")

        # Validate performance across regions
        for region_status in deployment_result.region_statuses:
            perf_metrics = region_status.performance_metrics
            if not perf_metrics.get('meets_target', True):
                self.logger.warning(f"Performance target not met in {region_status.region.value}")

    def _execute_rollback(self, deployment_result: GlobalDeploymentResult) -> None:
        """Execute rollback of deployment."""
        self.logger.warning(f"Executing rollback for deployment {deployment_result.deployment_id}")

        # Mark all successful regions for rollback
        for region_status in deployment_result.region_statuses:
            if region_status.status == "success":
                self.logger.info(f"Rolling back region {region_status.region.value}")

                # Simulate rollback
                time.sleep(0.05)

        # Log rollback
        self.audit_logger.log_security_event(
            event_type=AuditEventType.SYSTEM_ACCESS,
            action="deployment_rollback",
            result="success",
            severity=SecurityLevel.MEDIUM,
            details={
                "deployment_id": deployment_result.deployment_id,
                "rolled_back_regions": len([r for r in deployment_result.region_statuses if r.status == "success"])
            }
        )

    def get_deployment_status(self, deployment_id: str) -> Optional[GlobalDeploymentResult]:
        """Get status of active or historical deployment."""
        # Check active deployments first
        if deployment_id in self.active_deployments:
            return self.active_deployments[deployment_id]

        # Check deployment history
        for deployment in self.deployment_history:
            if deployment.deployment_id == deployment_id:
                return deployment

        return None

    def list_active_deployments(self) -> List[GlobalDeploymentResult]:
        """List all active deployments."""
        return list(self.active_deployments.values())

    def get_deployment_history(self, limit: int = 50) -> List[GlobalDeploymentResult]:
        """Get deployment history."""
        return self.deployment_history[-limit:]

    def get_global_health_status(self) -> Dict[str, Any]:
        """Get global health status across all regions."""
        system_health = self.health_monitor.get_system_health()
        perf_stats = self.performance_optimizer.get_performance_stats()

        return {
            "system_health": {
                "status": system_health.overall_status.value,
                "components": len(system_health.components),
                "alerts": len(system_health.alerts)
            },
            "performance": {
                "optimization_profiles": perf_stats.get("optimization_profiles", 0),
                "avg_improvement": perf_stats.get("optimization_effectiveness", {}).get("avg_improvement_factor", 1.0)
            },
            "active_deployments": len(self.active_deployments),
            "recent_deployments": len([d for d in self.deployment_history if time.time() - d.start_time < 3600])
        }

    def create_default_global_config(self) -> DeploymentConfig:
        """Create default global deployment configuration."""
        return DeploymentConfig(
            target_regions=[
                DeploymentRegion.US_EAST,
                DeploymentRegion.EU_WEST,
                DeploymentRegion.ASIA_PACIFIC
            ],
            supported_languages=[
                SupportedLanguage.ENGLISH,
                SupportedLanguage.SPANISH,
                SupportedLanguage.JAPANESE
            ],
            compliance_regions=[
                ComplianceRegion.CCPA,
                ComplianceRegion.GDPR,
                ComplianceRegion.PDPA_SINGAPORE
            ],
            enable_canary=True,
            canary_traffic_percent=5.0,
            rollback_on_errors=True,
            health_check_timeout=300,
            performance_threshold=2000.0,
            quality_gate_required=True
        )


# Global deployment system instance
_global_deployment: Optional[QuantumGlobalDeployment] = None


def get_global_deployment() -> QuantumGlobalDeployment:
    """Get global deployment system instance."""
    global _global_deployment
    if _global_deployment is None:
        _global_deployment = QuantumGlobalDeployment()
    return _global_deployment
