"""OpenAPI documentation generation tools."""

from importlib.metadata import PackageNotFoundError, version

from .discovery import RouteDiscoverer, RouteInfo
from .documentator import APIDocumentator, DocumentationResult
from .graphql import GraphQLSchema
from .markdown import MarkdownGenerator
from .migration import MigrationGuideGenerator
from .playground import PlaygroundGenerator
from .schema import FieldInfo, SchemaInferer, SchemaInfo
from .spec import OpenAPISpecGenerator
from .templates import load_template
from .testsuite import TestSuiteGenerator
from .utils import echo
from .validator import SpecValidator

try:  # pragma: no cover - package may not be installed in dev
    __version__ = version("openapi_doc_generator")
except PackageNotFoundError:  # pragma: no cover - fallback for editable installs
    __version__ = "0.0.0"

from . import plugins  # noqa: F401 - register built-in plugins
from .auto_scaler import (
    IntelligentAutoScaler,
    ResourceLimits,
    ScalingRule,
    get_auto_scaler,
)
from .cli import main as cli_main
from .discovery import RoutePlugin, register_plugin

# Generation 3: Advanced Performance Optimization and Auto-Scaling
from .performance_optimizer import (
    AdvancedCache,
    ParallelProcessor,
    PerformanceOptimizer,
    get_optimizer,
    optimized,
)
from .performance_optimizer import (
    OptimizationConfig as PerfOptimizationConfig,
)
from .quantum_monitor import (
    HealthStatus,
    PerformanceMetrics,
    QuantumPlanningMonitor,
    get_monitor,
    monitor_operation,
)
from .quantum_optimizer import (
    AdaptiveQuantumScheduler,
    OptimizationConfig,
    OptimizedQuantumPlanner,
    ParallelQuantumProcessor,
    QuantumCache,
)
from .quantum_planner import QuantumTaskPlanner, integrate_with_existing_sdlc

# Quantum-inspired task planning components
from .quantum_scheduler import (
    QuantumInspiredScheduler,
    QuantumResourceAllocator,
    QuantumScheduleResult,
    QuantumTask,
    TaskState,
)
from .quantum_validator import (
    QuantumTaskValidator,
    ValidationIssue,
    ValidationLevel,
    validate_quantum_plan,
)

# Advanced Autonomous SDLC Components (Generation 1-3)
try:
    from .ai_documentation_agent import (
        AIDocumentationAgent,
        DocumentationContext,
        AIDocumentationResult,
        AdvancedReasoningEngine,
        create_ai_documentation_agent
    )
    from .autonomous_code_analyzer import (
        AutonomousCodeAnalyzer,
        CodeAnalysisResult,
        create_autonomous_analyzer
    )
    from .autonomous_reliability_engine import (
        AutonomousReliabilityEngine,
        FailureType,
        RecoveryStrategy,
        SystemMetrics,
        create_reliability_engine
    )
    from .advanced_security_guardian import (
        AdvancedSecurityGuardian,
        ThreatType,
        ThreatLevel,
        SecurityEvent,
        SecurityAction,
        create_security_guardian
    )
    from .quantum_performance_engine import (
        QuantumPerformanceEngine,
        OptimizationStrategy,
        PerformanceSnapshot,
        create_quantum_performance_engine
    )
    
    # Mark as successfully loaded
    AUTONOMOUS_SDLC_LOADED = True
    
except ImportError as e:
    # Autonomous modules optional for core functionality
    AUTONOMOUS_SDLC_LOADED = False

__all__ = [
    "echo",
    "RouteDiscoverer",
    "RouteInfo",
    "SchemaInferer",
    "SchemaInfo",
    "FieldInfo",
    "OpenAPISpecGenerator",
    "APIDocumentator",
    "DocumentationResult",
    "load_template",
    "MarkdownGenerator",
    "PlaygroundGenerator",
    "GraphQLSchema",
    "TestSuiteGenerator",
    "MigrationGuideGenerator",
    "SpecValidator",
    "register_plugin",
    "RoutePlugin",
    "cli_main",
    "__version__",
    # Quantum planning exports
    "QuantumInspiredScheduler",
    "QuantumResourceAllocator",
    "QuantumTask",
    "TaskState",
    "QuantumScheduleResult",
    "QuantumTaskPlanner",
    "integrate_with_existing_sdlc",
    "QuantumTaskValidator",
    "ValidationLevel",
    "ValidationIssue",
    "validate_quantum_plan",
    "QuantumPlanningMonitor",
    "PerformanceMetrics",
    "HealthStatus",
    "get_monitor",
    "monitor_operation",
    "OptimizedQuantumPlanner",
    "OptimizationConfig",
    "AdaptiveQuantumScheduler",
    "ParallelQuantumProcessor",
    "QuantumCache",
    # Generation 3: Performance optimization and auto-scaling exports
    "get_optimizer",
    "optimized",
    "PerformanceOptimizer",
    "PerfOptimizationConfig",
    "AdvancedCache",
    "ParallelProcessor",
    "get_auto_scaler",
    "IntelligentAutoScaler",
    "ResourceLimits",
    "ScalingRule",
    # Advanced Autonomous SDLC Research Innovation exports
    "AIDocumentationAgent",
    "DocumentationContext", 
    "AIDocumentationResult",
    "AdvancedReasoningEngine",
    "create_ai_documentation_agent",
    "AutonomousCodeAnalyzer",
    "CodeAnalysisResult",
    "create_autonomous_analyzer",
    "AutonomousReliabilityEngine",
    "FailureType",
    "RecoveryStrategy", 
    "SystemMetrics",
    "create_reliability_engine",
    "AdvancedSecurityGuardian",
    "ThreatType",
    "ThreatLevel",
    "SecurityEvent",
    "SecurityAction", 
    "create_security_guardian",
    "QuantumPerformanceEngine",
    "OptimizationStrategy",
    "PerformanceSnapshot",
    "create_quantum_performance_engine",
    "AUTONOMOUS_SDLC_LOADED"
]
