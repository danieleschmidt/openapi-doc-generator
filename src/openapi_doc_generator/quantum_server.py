"""FastAPI server for Quantum Task Planner production deployment."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from .quantum_api import QuantumPlannerAPI, get_quantum_api
from .quantum_compliance import (
    ComplianceStandard,
    QuantumComplianceManager,
    get_compliance_manager,
)
from .quantum_recovery import get_recovery_manager
from .quantum_scaler import QuantumTaskScaler, ScalingConfig, get_quantum_scaler
from .quantum_security import (
    QuantumSecurityValidator,
    SecurityLevel,
    get_security_validator,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('quantum_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('quantum_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
ACTIVE_SESSIONS = Gauge('quantum_active_sessions', 'Number of active sessions')
TASK_COUNT = Gauge('quantum_tasks_total', 'Total tasks in system')
SECURITY_SCORE = Gauge('quantum_security_score', 'Current security score')

# Pydantic models
class CreateSessionRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=100)
    temperature: float = Field(default=2.0, ge=0.1, le=10.0)
    cooling_rate: float = Field(default=0.95, ge=0.1, le=1.0)
    num_resources: int = Field(default=4, ge=1, le=64)
    validation_level: str = Field(default="moderate", regex="^(strict|moderate|lenient)$")
    enable_monitoring: bool = Field(default=True)


class AddTaskRequest(BaseModel):
    id: str = Field(..., min_length=1, max_length=100)
    name: str = Field(..., min_length=1, max_length=500)
    priority: float = Field(default=1.0, ge=0.0, le=10.0)
    effort: float = Field(default=1.0, ge=0.0, le=1000.0)
    value: float = Field(default=1.0, ge=0.0, le=10000.0)
    dependencies: List[str] = Field(default_factory=list)
    coherence_time: float = Field(default=10.0, ge=0.1, le=3600.0)


class ExportPlanRequest(BaseModel):
    format: str = Field(..., regex="^(json|markdown)$")
    output_path: Optional[str] = Field(default=None)


class HealthResponse(BaseModel):
    status: str
    timestamp: float
    version: str
    services: Dict[str, str]


class ConsentRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=100)
    purpose: str = Field(..., min_length=1, max_length=200)
    consent_given: bool
    data_types: List[str]
    retention_period: Optional[int] = Field(default=None, ge=1, le=3650)


# Global instances
api_instance: Optional[QuantumPlannerAPI] = None
compliance_manager: Optional[QuantumComplianceManager] = None
security_validator: Optional[QuantumSecurityValidator] = None
scaler: Optional[QuantumTaskScaler] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global api_instance, compliance_manager, security_validator, scaler

    # Startup
    logger.info("Starting Quantum Task Planner server...")

    # Initialize components
    api_instance = get_quantum_api()
    compliance_manager = get_compliance_manager([
        ComplianceStandard.GDPR,
        ComplianceStandard.SOC2,
        ComplianceStandard.NIST_CSF
    ])
    security_validator = get_security_validator(SecurityLevel.MEDIUM)
    scaler = get_quantum_scaler(ScalingConfig(
        min_workers=int(os.getenv('MIN_WORKERS', '4')),
        max_workers=int(os.getenv('MAX_WORKERS', '16'))
    ))

    # Log compliance event
    compliance_manager.log_compliance_event(
        event_type="system_startup",
        metadata={"server_version": "1.0.0"}
    )

    logger.info("Quantum Task Planner server started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Quantum Task Planner server...")

    # Log compliance event
    compliance_manager.log_compliance_event(
        event_type="system_shutdown",
        metadata={"shutdown_time": time.time()}
    )

    # Cleanup resources
    if scaler:
        scaler.shutdown()

    logger.info("Quantum Task Planner server shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Quantum Task Planner API",
    description="Advanced quantum-inspired task planning and optimization system",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if os.getenv('ENVIRONMENT') != 'production' else None,
    redoc_url="/redoc" if os.getenv('ENVIRONMENT') != 'production' else None
)

# Middleware configuration
if os.getenv('ENABLE_CORS', 'false').lower() == 'true':
    allowed_origins = os.getenv('ALLOWED_ORIGINS', '').split(',')
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins or ["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

app.add_middleware(GZipMiddleware, minimum_size=1000)

if os.getenv('ENABLE_SECURITY_HEADERS', 'true').lower() == 'true':
    allowed_hosts = os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)


# Dependency functions
async def get_api() -> QuantumPlannerAPI:
    """Get API instance."""
    if api_instance is None:
        raise HTTPException(status_code=503, detail="API not initialized")
    return api_instance


async def get_compliance() -> QuantumComplianceManager:
    """Get compliance manager."""
    if compliance_manager is None:
        raise HTTPException(status_code=503, detail="Compliance manager not initialized")
    return compliance_manager


async def get_security() -> QuantumSecurityValidator:
    """Get security validator."""
    if security_validator is None:
        raise HTTPException(status_code=503, detail="Security validator not initialized")
    return security_validator


async def rate_limit_check(request: Request, security: QuantumSecurityValidator = Depends(get_security)):
    """Rate limiting middleware."""
    if os.getenv('ENABLE_RATE_LIMITING', 'true').lower() != 'true':
        return

    client_ip = request.client.host
    if not security.check_rate_limit(client_ip, max_requests=100, window_seconds=3600):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


# Middleware for request metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Collect request metrics."""
    start_time = time.time()
    method = request.method
    path = request.url.path

    response = await call_next(request)

    duration = time.time() - start_time
    status = response.status_code

    # Record metrics
    REQUEST_COUNT.labels(method=method, endpoint=path, status=status).inc()
    REQUEST_DURATION.labels(method=method, endpoint=path).observe(duration)

    return response


# Health endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check."""
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="1.0.0",
        services={
            "api": "healthy" if api_instance else "unhealthy",
            "compliance": "healthy" if compliance_manager else "unhealthy",
            "security": "healthy" if security_validator else "unhealthy",
            "scaler": "healthy" if scaler else "unhealthy"
        }
    )


@app.get("/health/ready")
async def readiness_check():
    """Readiness check for Kubernetes."""
    if not all([api_instance, compliance_manager, security_validator, scaler]):
        raise HTTPException(status_code=503, detail="Service not ready")

    return {"status": "ready", "timestamp": time.time()}


@app.get("/health/live")
async def liveness_check():
    """Liveness check for Kubernetes."""
    return {"status": "alive", "timestamp": time.time()}


# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    # Update gauge metrics
    if api_instance:
        sessions = api_instance.list_sessions()
        ACTIVE_SESSIONS.set(sessions.get("total_sessions", 0))

    if security_validator:
        # Mock security score - in production would be calculated
        SECURITY_SCORE.set(85.0)

    return Response(generate_latest(), media_type="text/plain")


# Session management endpoints
@app.post("/api/v1/sessions")
async def create_session(
    request: CreateSessionRequest,
    api: QuantumPlannerAPI = Depends(get_api),
    compliance: QuantumComplianceManager = Depends(get_compliance),
    _: None = Depends(rate_limit_check)
):
    """Create a new quantum planning session."""
    try:
        response = api.create_session(
            session_id=request.session_id,
            temperature=request.temperature,
            cooling_rate=request.cooling_rate,
            num_resources=request.num_resources,
            validation_level=request.validation_level,
            enable_monitoring=request.enable_monitoring
        )

        # Log compliance event
        compliance.log_compliance_event(
            event_type="session_created",
            session_id=request.session_id,
            metadata={"temperature": request.temperature, "resources": request.num_resources}
        )

        return response

    except Exception as e:
        logger.error(f"Failed to create session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/sessions")
async def list_sessions(
    api: QuantumPlannerAPI = Depends(get_api),
    _: None = Depends(rate_limit_check)
):
    """List all active sessions."""
    return api.list_sessions()


@app.get("/api/v1/sessions/{session_id}")
async def get_session_status(
    session_id: str,
    api: QuantumPlannerAPI = Depends(get_api),
    _: None = Depends(rate_limit_check)
):
    """Get session status and metrics."""
    return api.get_session_status(session_id)


@app.delete("/api/v1/sessions/{session_id}")
async def delete_session(
    session_id: str,
    api: QuantumPlannerAPI = Depends(get_api),
    compliance: QuantumComplianceManager = Depends(get_compliance),
    _: None = Depends(rate_limit_check)
):
    """Delete a session."""
    response = api.delete_session(session_id)

    # Log compliance event
    compliance.log_compliance_event(
        event_type="session_deleted",
        session_id=session_id
    )

    return response


# Task management endpoints
@app.post("/api/v1/sessions/{session_id}/tasks")
async def add_task(
    session_id: str,
    request: AddTaskRequest,
    api: QuantumPlannerAPI = Depends(get_api),
    compliance: QuantumComplianceManager = Depends(get_compliance),
    _: None = Depends(rate_limit_check)
):
    """Add a task to a session."""
    try:
        task_data = request.dict()
        response = api.add_task(session_id, task_data)

        # Log compliance event
        compliance.log_compliance_event(
            event_type="task_added",
            session_id=session_id,
            metadata={"task_id": request.id, "task_name": request.name}
        )

        return response

    except Exception as e:
        logger.error(f"Failed to add task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/sessions/{session_id}/tasks/sdlc")
async def add_sdlc_tasks(
    session_id: str,
    api: QuantumPlannerAPI = Depends(get_api),
    compliance: QuantumComplianceManager = Depends(get_compliance),
    _: None = Depends(rate_limit_check)
):
    """Add standard SDLC tasks to a session."""
    response = api.add_sdlc_tasks(session_id)

    # Log compliance event
    compliance.log_compliance_event(
        event_type="sdlc_tasks_added",
        session_id=session_id,
        metadata={"tasks_added": response.get("tasks_added", 0)}
    )

    return response


# Planning endpoints
@app.post("/api/v1/sessions/{session_id}/plan")
async def create_plan(
    session_id: str,
    background_tasks: BackgroundTasks,
    api: QuantumPlannerAPI = Depends(get_api),
    compliance: QuantumComplianceManager = Depends(get_compliance),
    _: None = Depends(rate_limit_check)
):
    """Create a quantum-optimized execution plan."""
    try:
        response = api.create_plan(session_id)

        # Log compliance event in background
        background_tasks.add_task(
            compliance.log_compliance_event,
            event_type="plan_created",
            session_id=session_id,
            data_processed=True,
            metadata={
                "total_tasks": len(response.get("quantum_plan", {}).get("optimized_tasks", [])),
                "quantum_fidelity": response.get("quantum_plan", {}).get("quantum_fidelity", 0.0)
            }
        )

        return response

    except Exception as e:
        logger.error(f"Failed to create plan: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/sessions/{session_id}/export")
async def export_plan(
    session_id: str,
    request: ExportPlanRequest,
    api: QuantumPlannerAPI = Depends(get_api),
    compliance: QuantumComplianceManager = Depends(get_compliance),
    _: None = Depends(rate_limit_check)
):
    """Export quantum plan in specified format."""
    try:
        response = api.export_plan(
            session_id=session_id,
            format=request.format,
            output_path=request.output_path
        )

        # Log compliance event
        compliance.log_compliance_event(
            event_type="plan_exported",
            session_id=session_id,
            metadata={"format": request.format}
        )

        return response

    except Exception as e:
        logger.error(f"Failed to export plan: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Compliance endpoints
@app.post("/api/v1/compliance/consent")
async def record_consent(
    request: ConsentRequest,
    compliance: QuantumComplianceManager = Depends(get_compliance),
    _: None = Depends(rate_limit_check)
):
    """Record user consent."""
    try:
        consent_id = compliance.record_consent(
            user_id=request.user_id,
            purpose=request.purpose,
            consent_given=request.consent_given,
            data_types=request.data_types,
            retention_period=request.retention_period
        )

        return {"status": "success", "consent_id": consent_id}

    except Exception as e:
        logger.error(f"Failed to record consent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/compliance/consent/{user_id}/{consent_id}")
async def withdraw_consent(
    user_id: str,
    consent_id: str,
    compliance: QuantumComplianceManager = Depends(get_compliance),
    _: None = Depends(rate_limit_check)
):
    """Withdraw user consent."""
    success = compliance.withdraw_consent(user_id, consent_id)

    if success:
        return {"status": "success", "message": "Consent withdrawn successfully"}
    else:
        raise HTTPException(status_code=404, detail="Consent record not found")


@app.get("/api/v1/compliance/audit")
async def run_compliance_audit(
    compliance: QuantumComplianceManager = Depends(get_compliance),
    _: None = Depends(rate_limit_check)
):
    """Run comprehensive compliance audit."""
    return compliance.run_compliance_audit()


@app.get("/api/v1/compliance/dashboard")
async def get_compliance_dashboard(
    compliance: QuantumComplianceManager = Depends(get_compliance),
    _: None = Depends(rate_limit_check)
):
    """Get compliance dashboard data."""
    return compliance.get_compliance_dashboard()


@app.get("/api/v1/compliance/export/{user_id}")
async def export_user_data(
    user_id: str,
    compliance: QuantumComplianceManager = Depends(get_compliance),
    _: None = Depends(rate_limit_check)
):
    """Export user data for portability."""
    try:
        return compliance.export_user_data(user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/v1/compliance/user/{user_id}")
async def delete_user_data(
    user_id: str,
    compliance: QuantumComplianceManager = Depends(get_compliance),
    _: None = Depends(rate_limit_check)
):
    """Delete all user data (Right to Erasure)."""
    try:
        return compliance.delete_user_data(user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Security endpoints
@app.get("/api/v1/security/report/{session_id}")
async def get_security_report(
    session_id: str,
    api: QuantumPlannerAPI = Depends(get_api),
    _: None = Depends(rate_limit_check)
):
    """Get security report for a session."""
    session_status = api.get_session_status(session_id)

    if session_status["status"] != "success":
        raise HTTPException(status_code=404, detail="Session not found")

    # Get security report from the planner
    planner = api.planners.get(session_id)
    if planner:
        return planner.get_security_report()
    else:
        raise HTTPException(status_code=404, detail="Session not found")


# Performance and scaling endpoints
@app.get("/api/v1/performance/stats")
async def get_performance_stats(
    _: None = Depends(rate_limit_check)
):
    """Get system performance statistics."""
    if scaler:
        return scaler.get_performance_stats()
    else:
        raise HTTPException(status_code=503, detail="Scaler not initialized")


@app.get("/api/v1/performance/recovery")
async def get_recovery_stats(
    _: None = Depends(rate_limit_check)
):
    """Get recovery system statistics."""
    recovery_manager = get_recovery_manager()
    return recovery_manager.get_recovery_statistics()


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": time.time()
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "quantum_server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        workers=1,  # Use 1 worker for development
        reload=os.getenv("ENVIRONMENT") != "production",
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        access_log=True
    )
