"""OpenAPI documentation generation tools."""

from importlib.metadata import PackageNotFoundError, version

from .utils import echo
from .discovery import RouteDiscoverer, RouteInfo
from .schema import SchemaInferer, SchemaInfo, FieldInfo
from .spec import OpenAPISpecGenerator
from .documentator import APIDocumentator, DocumentationResult
from .templates import load_template
from .markdown import MarkdownGenerator
from .playground import PlaygroundGenerator
from .graphql import GraphQLSchema
from .testsuite import TestSuiteGenerator
from .migration import MigrationGuideGenerator
from .validator import SpecValidator

try:  # pragma: no cover - package may not be installed in dev
    __version__ = version("openapi_doc_generator")
except PackageNotFoundError:  # pragma: no cover - fallback for editable installs
    __version__ = "0.0.0"

from .cli import main as cli_main
from .discovery import register_plugin, RoutePlugin
from . import plugins  # noqa: F401 - register built-in plugins

# Quantum-inspired task planning components 
from .quantum_scheduler import (
    QuantumInspiredScheduler,
    QuantumResourceAllocator,
    QuantumTask,
    TaskState,
    QuantumScheduleResult
)
from .quantum_planner import QuantumTaskPlanner, integrate_with_existing_sdlc
from .quantum_validator import (
    QuantumTaskValidator,
    ValidationLevel,
    ValidationIssue,
    validate_quantum_plan
)
from .quantum_monitor import (
    QuantumPlanningMonitor,
    PerformanceMetrics,
    HealthStatus,
    get_monitor,
    monitor_operation
)
from .quantum_optimizer import (
    OptimizedQuantumPlanner,
    OptimizationConfig,
    AdaptiveQuantumScheduler,
    ParallelQuantumProcessor,
    QuantumCache
)

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
]
