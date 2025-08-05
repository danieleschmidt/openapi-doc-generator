"""Command line interface for generating documentation artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from .documentator import APIDocumentator, DocumentationResult
from .playground import PlaygroundGenerator
from .graphql import GraphQLSchema
from .testsuite import TestSuiteGenerator
from .migration import MigrationGuideGenerator
from .config import config
from .quantum_planner import QuantumTaskPlanner, integrate_with_existing_sdlc
from . import __version__


class ErrorCode:
    """Enumeration of CLI error codes."""

    APP_NOT_FOUND = "CLI001"
    OLD_SPEC_REQUIRED = "CLI002"
    OLD_SPEC_INVALID = "CLI003"
    OUTPUT_PATH_INVALID = "CLI004"
    TESTS_PATH_INVALID = "CLI005"


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
    
    return parser


def _check_path_traversal(path_str: str) -> bool:
    """Check for path traversal patterns."""
    return ".." in path_str or (path_str.startswith("/") and "/../" in path_str)


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
    """Process quantum-inspired task planning format."""
    # Initialize quantum planner with CLI parameters
    planner = QuantumTaskPlanner(
        temperature=args.quantum_temperature,
        num_resources=args.quantum_resources
    )
    
    # Integrate with existing SDLC tasks
    integrate_with_existing_sdlc(planner)
    
    # Create quantum plan
    result = planner.create_quantum_plan()
    
    # Generate detailed output
    output_lines = [
        "# Quantum-Inspired Task Planning Results",
        "",
        f"**Quantum Fidelity**: {result.quantum_fidelity:.3f}",
        f"**Total Value**: {result.total_value:.2f}",
        f"**Execution Time**: {result.execution_time:.3f}s",
        f"**Convergence Iterations**: {result.convergence_iterations}",
        "",
        "## Optimized Task Schedule",
        ""
    ]
    
    for i, task in enumerate(result.optimized_tasks, 1):
        resource_id = getattr(task, 'allocated_resource', 0)
        output_lines.extend([
            f"### {i}. {task.name}",
            f"- **ID**: {task.id}",
            f"- **Priority**: {task.priority:.2f}",
            f"- **Effort**: {task.effort:.1f}",
            f"- **Value**: {task.value:.1f}",
            f"- **Quantum Weight**: {task.quantum_weight:.3f}",
            f"- **Allocated Resource**: Resource-{resource_id}",
            f"- **Dependencies**: {', '.join(task.dependencies) if task.dependencies else 'None'}",
            f"- **Entangled Tasks**: {len(task.entangled_tasks)}",
            ""
        ])
    
    # Add quantum metrics
    output_lines.extend([
        "## Quantum Effects Summary",
        f"- **Superposition States Created**: {len([t for t in result.optimized_tasks if 'super' in t.id])}",
        f"- **Task Entanglements**: {sum(len(t.entangled_tasks) for t in result.optimized_tasks) // 2}",
        f"- **Measurement Collapses**: {sum(t.measurement_count for t in result.optimized_tasks)}",
        ""
    ])
    
    # Add simulation results
    simulation = planner.simulate_execution(result)
    output_lines.extend([
        "## Execution Simulation",
        f"- **Estimated Completion Time**: {simulation['estimated_completion_time']:.2f} time units",
        f"- **Total Tasks**: {simulation['total_tasks']}",
        f"- **Quantum Effects**:",
        f"  - Superposition Collapses: {simulation['quantum_effects']['superposition_collapses']}",
        f"  - Entanglement Breaks: {simulation['quantum_effects']['entanglement_breaks']}",
        f"  - Coherence Loss Events: {simulation['quantum_effects']['coherence_loss']:.0f}",
        ""
    ])
    
    return "\n".join(output_lines)


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
    parser = build_parser()
    args = parser.parse_args(argv)
    
    # Setup logging
    log_level = _determine_log_level(args)
    use_colors = not args.no_color
    logger = _setup_logging(args.log_format, level=log_level, colored=use_colors)

    # Configure performance metrics
    from .utils import set_performance_tracking
    set_performance_tracking(args.performance_metrics)

    # Validate and process
    _show_progress("Validating application path", args.verbose)
    app_path = _validate_app_path(args.app, parser, logger)

    output, result = _process_documentation_format(args, app_path, parser, logger)
    _generate_test_suite(result, args, parser, logger)

    _show_progress("Writing output", args.verbose)
    _write_output(output, args.output, parser, logger)

    _log_performance_summary(args, logger)

    if args.verbose:
        sys.stderr.write("✅ Documentation generation completed successfully!\n")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
