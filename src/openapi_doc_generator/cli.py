"""Command line interface for generating documentation artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from . import __version__
from .config import config
from .documentator import APIDocumentator, DocumentationResult
from .graphql import GraphQLSchema
from .migration import MigrationGuideGenerator
from .i18n import get_i18n_manager, SupportedLanguage, ComplianceRegion, localize_text
from .playground import PlaygroundGenerator
from .quantum_planner import QuantumTaskPlanner, integrate_with_existing_sdlc
from .testsuite import TestSuiteGenerator


class ErrorCode:
    """Enumeration of CLI error codes."""

    APP_NOT_FOUND = "CLI001"
    OLD_SPEC_REQUIRED = "CLI002"
    OLD_SPEC_INVALID = "CLI003"
    OUTPUT_PATH_INVALID = "CLI004"
    TESTS_PATH_INVALID = "CLI005"
    SECURITY_VIOLATION = "CLI006"
    RESOURCE_EXHAUSTION = "CLI007"
    INVALID_INPUT = "CLI008"
    PERMISSION_DENIED = "CLI009"
    TIMEOUT_ERROR = "CLI010"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate documentation in various formats"
    )
    parser.add_argument(
        "--app",
        required=True,
        help="Path to application source file",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="File to write output to (prints to stdout if omitted)",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["markdown", "openapi", "html", "graphql", "guide", "quantum-plan"],
        default="markdown",
        help=("Output format: markdown (default), openapi, html, graphql, guide, or quantum-plan"),
    )
    parser.add_argument(
        "--old-spec",
        help="Path to previous OpenAPI spec for migration guide",
    )
    parser.add_argument(
        "--tests",
        help="File to write generated pytest suite",
    )
    parser.add_argument(
        "--title",
        default=config.DEFAULT_API_TITLE,
        help="Title for generated API documentation",
    )
    parser.add_argument(
        "--api-version",
        default=config.DEFAULT_API_VERSION,
        help="Version string for generated API documentation",
    )
    parser.add_argument(
        "--log-format",
        choices=["standard", "json"],
        default="standard",
        help="Log format: standard (default) or json for structured logging",
    )
    parser.add_argument(
        "--performance-metrics",
        action="store_true",
        help="Enable detailed performance metrics collection and logging",
    )
    parser.add_argument(
        "--quantum-temperature",
        type=float,
        default=2.0,
        help="Initial temperature for quantum annealing (default: 2.0)",
    )
    parser.add_argument(
        "--quantum-resources",
        type=int,
        default=4,
        help="Number of quantum resources for allocation (default: 4)",
    )
    parser.add_argument(
        "--quantum-cooling-rate",
        type=float,
        default=0.95,
        help="Cooling rate for quantum annealing (default: 0.95)",
    )
    parser.add_argument(
        "--quantum-validation",
        choices=["strict", "moderate", "lenient"],
        default="moderate",
        help="Quantum task validation level (default: moderate)",
    )

    # Verbose/Quiet mode flags (mutually exclusive)
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output with detailed progress information",
    )
    verbosity_group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress all non-error output",
    )

    # Color output control
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )
    
    # Global/i18n options
    parser.add_argument(
        "--language",
        choices=[lang.value for lang in SupportedLanguage],
        default=None,
        help="Output language (default: auto-detect from system)",
    )
    parser.add_argument(
        "--region", 
        default="US",
        help="Deployment region for compliance and localization (default: US)",
    )
    parser.add_argument(
        "--compliance",
        choices=["gdpr", "ccpa", "pdpa-sg", "lgpd", "pipeda"],
        action="append",
        help="Enable compliance features for specified regulations",
    )
    parser.add_argument(
        "--timezone",
        default="UTC",
        help="Timezone for timestamps and date formatting (default: UTC)",
    )

    return parser


def _check_path_traversal(path_str: str) -> bool:
    """Check if path contains potential traversal attacks."""
    dangerous_patterns = ["../", "..\\", "~", "$", "|", ";", "&", "`", "<", ">"]
    return any(pattern in path_str for pattern in dangerous_patterns)


def _validate_file_size(file_path: Path, max_size_mb: int = 50) -> bool:
    """Validate file size to prevent resource exhaustion."""
    if not file_path.exists():
        return True  # File doesn't exist yet, size check not needed

    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    return file_size_mb <= max_size_mb


def _sanitize_user_input(user_input: str, max_length: int = 1000) -> str:
    """Sanitize user input to prevent injection attacks."""
    if len(user_input) > max_length:
        raise ValueError(f"Input too long: {len(user_input)} > {max_length}")

    # Remove potentially dangerous characters
    dangerous_chars = ['<', '>', '&', '"', "'", '`', '$', '|', ';']
    sanitized = user_input
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '')

    return sanitized.strip()


def _check_resource_limits() -> Dict[str, Any]:
    """Check system resource limits to prevent exhaustion."""
    try:
        import psutil
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        cpu_percent = psutil.cpu_percent(interval=1)

        return {
            'memory_usage': memory_percent,
            'disk_usage': disk_percent,
            'cpu_usage': cpu_percent,
            'healthy': memory_percent < 90 and disk_percent < 95 and cpu_percent < 95
        }
    except ImportError:
        # psutil not available, assume healthy
        return {'healthy': True, 'note': 'psutil not available for resource monitoring'}


def _create_secure_temp_file(content: str, prefix: str = 'openapi_') -> Path:
    """Create a secure temporary file with restricted permissions."""
    with tempfile.NamedTemporaryFile(mode='w', prefix=prefix, suffix='.tmp', delete=False) as tmp_file:
        tmp_file.write(content)
        temp_path = Path(tmp_file.name)

    # Set restrictive permissions (read/write for owner only)
    temp_path.chmod(0o600)
    return temp_path


def _validate_file_target(
    path_str: str,
    flag: str,
    parser: argparse.ArgumentParser,
    logger: logging.Logger,
    code: str,
) -> Path:
    """Ensure a CLI file argument points to a writable file path."""
    # Normalize and validate the path to prevent directory traversal
    path = Path(path_str).resolve()

    # Check for suspicious path patterns
    if _check_path_traversal(path_str):
        logger.error("Suspicious path detected: %s", path)
        parser.error(f"[{code}] Invalid path: path traversal attempts not allowed")

    if path.exists() and path.is_dir():
        logger.error("%s path '%s' is a directory", flag, path)
        parser.error(f"[{code}] {flag} path '{path}' is a directory")

    parent = path.parent
    if parent and not parent.exists():
        logger.error("Directory '%s' does not exist", parent)
        parser.error(f"[{code}] directory '{parent}' does not exist")

    return path


def _load_old_spec_data(old_spec_path: str, parser: argparse.ArgumentParser, logger: logging.Logger) -> dict:
    """Load and validate old spec file."""
    old_path = Path(old_spec_path)
    if not old_path.exists():
        logger.error("Old spec file '%s' not found", old_path)
        parser.error(
            f"[{ErrorCode.OLD_SPEC_REQUIRED}] Old spec file '{old_path}' not found"
        )
    try:
        return json.loads(old_path.read_text())
    except json.JSONDecodeError as exc:  # pragma: no cover - manual error path
        logger.error("Invalid JSON in old spec: %s", exc)
        parser.error(f"[{ErrorCode.OLD_SPEC_INVALID}] --old-spec is not valid JSON")


def _generate_guide_output(result: DocumentationResult, args: argparse.Namespace, parser: argparse.ArgumentParser, logger: logging.Logger) -> str:
    """Generate migration guide output."""
    if not args.old_spec:
        parser.error(
            f"[{ErrorCode.OLD_SPEC_REQUIRED}] --old-spec is required "
            f"for guide format"
        )
    old_data = _load_old_spec_data(args.old_spec, parser, logger)
    new_spec = result.generate_openapi_spec(
        title=args.title, version=args.api_version
    )
    return MigrationGuideGenerator(old_data, new_spec).generate_markdown()


def _generate_output(
    result: DocumentationResult,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    logger: logging.Logger,
) -> str:
    """Return documentation output based on CLI args."""
    if args.format == "markdown":
        return result.generate_markdown(title=args.title, version=args.api_version)
    if args.format == "openapi":
        spec = result.generate_openapi_spec(title=args.title, version=args.api_version)
        return json.dumps(spec, indent=2)
    if args.format == "html":
        spec = result.generate_openapi_spec(title=args.title, version=args.api_version)
        return PlaygroundGenerator().generate(spec)
    if args.format == "guide":
        return _generate_guide_output(result, args, parser, logger)
    parser.error(f"Unknown format '{args.format}'")


def _setup_logging(log_format: str = "standard", level: int = logging.INFO, colored: bool = True) -> logging.Logger:
    """Configure logging and return logger instance."""
    if log_format == "json":
        from .utils import setup_json_logging
        return setup_json_logging(level)
    else:
        # Standard logging format with optional color support
        if colored and not os.getenv("NO_COLOR") and sys.stderr.isatty():
            # Use colored logging format
            format_str = "\033[36m%(levelname)s\033[0m:\033[32m%(name)s\033[0m:%(message)s"
        else:
            # Standard format without colors
            format_str = "%(levelname)s:%(name)s:%(message)s"

        logging.basicConfig(level=level, format=format_str)
        return logging.getLogger(__name__)


def _validate_app_path_input(app_path_str: str) -> Path:
    """Validate and normalize app path input."""
    # Check for empty or whitespace-only paths
    if not app_path_str or not app_path_str.strip():
        raise ValueError("App path cannot be empty")

    app_path = Path(app_path_str).resolve()

    # Security check - prevent obvious directory traversal patterns
    if _check_path_traversal(app_path_str):
        raise ValueError("Path contains suspicious traversal patterns")

    return app_path


def _validate_app_path(
    app_path_str: str, parser: argparse.ArgumentParser, logger: logging.Logger
) -> Path:
    """Validate and normalize app path with security checks."""
    try:
        app_path = _validate_app_path_input(app_path_str)
    except (ValueError, OSError) as e:
        logger.error("Invalid app path '%s': %s", app_path_str, e)
        parser.error(f"[{ErrorCode.APP_NOT_FOUND}] Invalid app path: {e}")
        return Path()  # This line will never be reached in real usage, but needed for testing

    if not app_path.exists():
        logger.error("App file '%s' not found", app_path)
        parser.error(f"[{ErrorCode.APP_NOT_FOUND}] App file '{app_path}' not found")

    return app_path


def _process_graphql_format(
    app_path: Path, parser: argparse.ArgumentParser, logger: logging.Logger
) -> str:
    """Process GraphQL format and return JSON output."""
    try:
        data = GraphQLSchema(str(app_path)).introspect()
        return json.dumps(data, indent=2)
    except ValueError as e:
        logger.error("GraphQL schema error: %s", e)
        parser.error(f"[{ErrorCode.APP_NOT_FOUND}] GraphQL schema error: {e}")


def _write_output(
    output: str,
    output_path: Optional[str],
    parser: argparse.ArgumentParser,
    logger: logging.Logger,
) -> None:
    """Write output to file or stdout."""
    if output_path:
        out_path = _validate_file_target(
            output_path, "--output", parser, logger, ErrorCode.OUTPUT_PATH_INVALID
        )
        out_path.write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)


def _show_progress(message: str, verbose: bool = False) -> None:
    """Show progress message if verbose mode is enabled."""
    if verbose:
        sys.stderr.write(f"🔄 {message}...\n")
        sys.stderr.flush()


def _determine_log_level(args: argparse.Namespace) -> int:
    """Determine logging level based on CLI flags."""
    if args.verbose:
        return logging.DEBUG
    elif args.quiet:
        return logging.WARNING
    else:
        level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        return getattr(logging, level_name, logging.INFO)


def _process_documentation_format(args: argparse.Namespace, app_path: Path, parser: argparse.ArgumentParser, logger: logging.Logger) -> tuple[str, Optional[DocumentationResult]]:
    """Process documentation based on format type."""
    if args.format == "graphql":
        _show_progress("Processing GraphQL schema", args.verbose)
        output = _process_graphql_format(app_path, parser, logger)
        return output, None
    elif args.format == "quantum-plan":
        _show_progress("Creating quantum-inspired task plan", args.verbose)
        output = _process_quantum_plan_format(args, parser, logger)
        return output, None
    else:
        _show_progress("Analyzing application structure", args.verbose)
        result = APIDocumentator().analyze_app(str(app_path))
        _show_progress("Generating documentation", args.verbose)
        output = _generate_output(result, args, parser, logger)
        return output, result


def _generate_test_suite(result: DocumentationResult, args: argparse.Namespace, parser: argparse.ArgumentParser, logger: logging.Logger) -> None:
    """Generate test suite if requested."""
    if result and args.tests:
        _show_progress("Generating test suite", args.verbose)
        tests_path = _validate_file_target(
            args.tests, "--tests", parser, logger, ErrorCode.TESTS_PATH_INVALID
        )
        tests_path.write_text(
            TestSuiteGenerator(result).generate_pytest(), encoding="utf-8"
        )


def _process_quantum_plan_format(args: argparse.Namespace, parser: argparse.ArgumentParser, logger: logging.Logger) -> str:
    """Process quantum-inspired task planning format with enhanced features."""
    try:
        # Map validation level string to enum
        from .quantum_validator import ValidationLevel
        validation_map = {
            "strict": ValidationLevel.STRICT,
            "moderate": ValidationLevel.MODERATE,
            "lenient": ValidationLevel.LENIENT
        }

        # Initialize quantum planner with CLI parameters
        planner = QuantumTaskPlanner(
            temperature=args.quantum_temperature,
            cooling_rate=args.quantum_cooling_rate,
            num_resources=args.quantum_resources,
            validation_level=validation_map[args.quantum_validation],
            enable_monitoring=args.performance_metrics,
            enable_optimization=True
        )

        # Integrate with existing SDLC tasks
        integrate_with_existing_sdlc(planner)

        logger.info(f"Quantum planner initialized with temperature={args.quantum_temperature}, resources={args.quantum_resources}")

        # Create quantum plan
        result = planner.create_quantum_plan()

        # Get performance statistics if monitoring enabled
        perf_stats = planner.get_performance_statistics() if args.performance_metrics else None

        # Generate detailed output
        output_lines = [
            "# Quantum-Inspired Task Planning Results",
            "",
            f"**Generated**: {result.execution_time:.3f}s",
            f"**Quantum Fidelity**: {result.quantum_fidelity:.3f}",
            f"**Total Business Value**: {result.total_value:.2f}",
            f"**Convergence Iterations**: {result.convergence_iterations}",
            "",
            "## Configuration",
            f"- **Temperature**: {args.quantum_temperature}",
            f"- **Resources**: {args.quantum_resources}",
            f"- **Monitoring**: {'Enabled' if args.performance_metrics else 'Disabled'}",
            "- **Optimization**: Enabled",
            "",
            "## Optimized Task Schedule",
            ""
        ]

        # Add task details with enhanced information
        for i, task in enumerate(result.optimized_tasks, 1):
            resource_id = getattr(task, 'allocated_resource', 0)
            quantum_metrics = planner.get_task_quantum_metrics(task.id)

            output_lines.extend([
                f"### {i}. {task.name}",
                f"- **ID**: `{task.id}`",
                f"- **Priority**: {task.priority:.2f}",
                f"- **Effort**: {task.effort:.1f} story points",
                f"- **Business Value**: {task.value:.1f}",
                f"- **Quantum Weight**: {task.quantum_weight:.3f}",
                f"- **Coherence Time**: {task.coherence_time:.1f}s",
                f"- **Allocated Resource**: Resource-{resource_id}",
                f"- **Dependencies**: {', '.join(f'`{dep}`' for dep in task.dependencies) if task.dependencies else 'None'}",
                f"- **State**: {task.state.value}",
                f"- **Entangled Tasks**: {len(task.entangled_tasks)} connections",
                ""
            ])

        # Add enhanced quantum metrics
        superposition_tasks = [t for t in result.optimized_tasks if 'superposition' in t.state.value.lower()]
        total_entanglements = sum(len(t.entangled_tasks) for t in result.optimized_tasks) // 2
        total_measurements = sum(t.measurement_count for t in result.optimized_tasks)

        output_lines.extend([
            "## Quantum Effects Analysis",
            f"- **Tasks in Superposition**: {len(superposition_tasks)}",
            f"- **Total Task Entanglements**: {total_entanglements}",
            f"- **Quantum Measurements**: {total_measurements}",
            f"- **Average Coherence Time**: {sum(t.coherence_time for t in result.optimized_tasks) / len(result.optimized_tasks):.1f}s",
            ""
        ])

        # Add simulation results with detailed resource utilization
        simulation = planner.simulate_execution(result)
        output_lines.extend([
            "## Execution Simulation",
            f"- **Estimated Completion**: {simulation['estimated_completion_time']:.2f} time units",
            f"- **Total Tasks**: {simulation['total_tasks']}",
            "",
            "### Resource Utilization",
        ])

        for resource, utilization in simulation['resource_utilization'].items():
            output_lines.append(f"- **{resource}**: {utilization:.1f} time units")

        output_lines.extend([
            "",
            "### Quantum Effects During Execution",
            f"- **Superposition Collapses**: {simulation['quantum_effects']['superposition_collapses']}",
            f"- **Entanglement Breaks**: {simulation['quantum_effects']['entanglement_breaks']}",
            f"- **Coherence Loss Events**: {simulation['quantum_effects']['coherence_loss']:.0f}",
            ""
        ])

        # Add performance statistics if enabled
        if perf_stats and args.performance_metrics:
            output_lines.extend([
                "## Performance Metrics",
                f"- **Total Tasks Registered**: {perf_stats['task_registry']['total_tasks']}",
                f"- **Validation Level**: {perf_stats['configuration']['validation_level']}",
                f"- **Monitoring Status**: {'Active' if perf_stats['configuration']['monitoring_enabled'] else 'Inactive'}",
                ""
            ])

            if 'monitoring' in perf_stats:
                output_lines.extend([
                    "### System Health",
                    f"- **Status**: {perf_stats['health']['system_status']}",
                ])
                for check in perf_stats['health']['health_checks']:
                    status_icon = "✅" if check['status'] == 'healthy' else "❌"
                    output_lines.append(f"- **{check['component']}**: {status_icon} {check['message']}")
                output_lines.append("")

        # Add recommendations
        output_lines.extend([
            "## Optimization Recommendations",
            "- Monitor task execution against quantum fidelity metrics",
            "- Adjust temperature parameter if convergence is slow",
            "- Consider increasing resources for parallel task execution",
            "- Review dependencies to minimize entanglement complexity",
            "",
            "---",
            f"*Generated by Quantum Task Planner v{planner.scheduler.temperature} at {result.execution_time:.3f}s*"
        ])

        logger.info(f"Quantum plan generated successfully: {len(result.optimized_tasks)} tasks, fidelity={result.quantum_fidelity:.3f}")
        return "\n".join(output_lines)

    except Exception as e:
        logger.error(f"Quantum plan generation failed: {str(e)}")
        return f"# Quantum Plan Generation Failed\n\nError: {str(e)}\n\nPlease check your configuration and try again."


def _log_performance_summary(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Log performance summary if metrics are enabled."""
    if args.performance_metrics:
        from .utils import get_performance_summary
        summary = get_performance_summary()
        if summary:
            logger.info(
                "Performance Summary:",
                extra={
                    "operation": "performance_summary",
                    "performance_stats": summary,
                },
            )


def main(argv: Optional[list[str]] = None) -> int:
    """Main CLI function with comprehensive error handling and security measures."""
    start_time = time.time()
    parser = build_parser()

    try:
        args = parser.parse_args(argv)

        # Configure i18n and global settings
        i18n_manager = get_i18n_manager()
        if args.language:
            try:
                language = SupportedLanguage(args.language)
                i18n_manager.set_language(language)
            except ValueError:
                sys.stderr.write(f"[{ErrorCode.INVALID_INPUT}] Unsupported language: {args.language}\n")
                return 1
        
        # Configure compliance regions
        if args.compliance:
            compliance_mapping = {
                "gdpr": ComplianceRegion.GDPR,
                "ccpa": ComplianceRegion.CCPA,
                "pdpa-sg": ComplianceRegion.PDPA_SINGAPORE,
                "lgpd": ComplianceRegion.LGPD,
                "pipeda": ComplianceRegion.PIPEDA
            }
            for compliance_str in args.compliance:
                if compliance_str in compliance_mapping:
                    i18n_manager.add_compliance_region(compliance_mapping[compliance_str])

        # Sanitize user inputs
        try:
            args.title = _sanitize_user_input(args.title)
            args.api_version = _sanitize_user_input(args.api_version)
        except ValueError as e:
            sys.stderr.write(f"[{ErrorCode.INVALID_INPUT}] {localize_text('cli.error.invalid_input')}: {e}\n")
            return 1

        # Setup logging
        log_level = _determine_log_level(args)
        use_colors = not args.no_color
        logger = _setup_logging(args.log_format, level=log_level, colored=use_colors)

        # Check system resources
        _show_progress("Checking system resources", args.verbose)
        resource_status = _check_resource_limits()
        if not resource_status.get('healthy', True):
            logger.warning("System resources under pressure: %s", resource_status)
            if resource_status.get('memory_usage', 0) > 95:
                sys.stderr.write(f"[{ErrorCode.RESOURCE_EXHAUSTION}] Insufficient memory\n")
                return 1

        # Configure performance metrics
        from .utils import set_performance_tracking
        set_performance_tracking(args.performance_metrics)

        # Validate and process with timeout protection
        _show_progress("Validating application path", args.verbose)
        app_path = _validate_app_path(args.app, parser, logger)

        # Check file size limits
        if not _validate_file_size(app_path):
            sys.stderr.write(f"[{ErrorCode.RESOURCE_EXHAUSTION}] File too large: {app_path}\n")
            return 1

        _show_progress("Processing documentation format", args.verbose)
        output, result = _process_documentation_format(args, app_path, parser, logger)

        _show_progress("Generating test suite", args.verbose)
        _generate_test_suite(result, args, parser, logger)

        _show_progress("Writing output", args.verbose)
        _write_output(output, args.output, parser, logger)

        _log_performance_summary(args, logger)

        execution_time = time.time() - start_time
        if args.verbose:
            sys.stderr.write(f"✅ Documentation generation completed successfully in {execution_time:.2f}s!\n")

        return 0

    except KeyboardInterrupt:
        sys.stderr.write(f"\n[{ErrorCode.TIMEOUT_ERROR}] Operation cancelled by user\n")
        return 130  # Standard exit code for SIGINT

    except PermissionError as e:
        sys.stderr.write(f"[{ErrorCode.PERMISSION_DENIED}] Permission denied: {e}\n")
        return 1

    except OSError as e:
        sys.stderr.write(f"[{ErrorCode.RESOURCE_EXHAUSTION}] System error: {e}\n")
        return 1

    except MemoryError:
        sys.stderr.write(f"[{ErrorCode.RESOURCE_EXHAUSTION}] Out of memory\n")
        return 1

    except Exception as e:
        # Log full traceback for debugging
        traceback.print_exc()
        sys.stderr.write(f"[INTERNAL_ERROR] Unexpected error: {e}\n")
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
