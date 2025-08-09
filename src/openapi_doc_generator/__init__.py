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
from .cli import main as cli_main
from .discovery import RoutePlugin, register_plugin
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

# Advanced Quantum SDLC Research Innovations (Generation 1-3)
# Temporarily disabled quantum imports to fix loading issues
# Will be re-enabled after core functionality is stable

# try:
#     from .quantum_hybrid_orchestrator import (
#         HybridQuantumClassicalOrchestrator,
#         HybridTask,
#         HybridTaskState,
#         CICDQuantumInterface,
#         CrossDomainEntanglementManager
#     )
# except ImportError:
#     pass  # Quantum modules optional for core functionality

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
    # Advanced Quantum SDLC Research Innovation exports (temporarily disabled)
    # "HybridQuantumClassicalOrchestrator",
    # ... quantum exports will be re-enabled after core stability
]
