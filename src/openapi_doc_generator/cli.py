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
        choices=["markdown", "openapi", "html", "graphql", "guide"],
        default="markdown",
        help=("Output format: markdown (default), openapi, html, graphql, or guide"),
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


def _validate_file_target(
    path_str: str,
    flag: str,
    parser: argparse.ArgumentParser,
    logger: logging.Logger,
    code: str,
) -> Path:
    """Ensure a CLI file argument points to a writable file path."""
    path = Path(path_str).resolve()
    
    _check_path_security(path_str, code, logger, parser)
    _validate_path_not_directory(path, flag, code, logger, parser)
    _ensure_parent_directory_exists(path, code, logger, parser)
    
    return path


def _check_path_security(
    path_str: str, code: str, logger: logging.Logger, parser: argparse.ArgumentParser
) -> None:
    """Check for suspicious path patterns to prevent directory traversal."""
    if ".." in path_str or (path_str.startswith("/") and "/../" in path_str):
        logger.error("Suspicious path detected: %s", path_str)
        parser.error(f"[{code}] Invalid path: path traversal attempts not allowed")


def _validate_path_not_directory(
    path: Path, flag: str, code: str, logger: logging.Logger, parser: argparse.ArgumentParser
) -> None:
    """Ensure the path is not an existing directory."""
    if path.exists() and path.is_dir():
        logger.error("%s path '%s' is a directory", flag, path)
        parser.error(f"[{code}] {flag} path '{path}' is a directory")


def _ensure_parent_directory_exists(
    path: Path, code: str, logger: logging.Logger, parser: argparse.ArgumentParser
) -> None:
    """Verify that the parent directory exists."""
    parent = path.parent
    if parent and not parent.exists():
        logger.error("Directory '%s' does not exist", parent)
        parser.error(f"[{code}] directory '{parent}' does not exist")


def _generate_output(
    result: DocumentationResult,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    logger: logging.Logger,
) -> str:
    """Return documentation output based on CLI args."""
    format_handlers = {
        "markdown": _generate_markdown_output,
        "openapi": _generate_openapi_output,
        "html": _generate_html_output,
        "guide": _generate_guide_output,
    }
    
    handler = format_handlers.get(args.format)
    if not handler:
        parser.error(f"Unknown format '{args.format}'")
    
    return handler(result, args, parser, logger)


def _generate_markdown_output(
    result: DocumentationResult, args: argparse.Namespace, 
    parser: argparse.ArgumentParser, logger: logging.Logger
) -> str:
    """Generate markdown format output."""
    return result.generate_markdown(title=args.title, version=args.api_version)


def _generate_openapi_output(
    result: DocumentationResult, args: argparse.Namespace,
    parser: argparse.ArgumentParser, logger: logging.Logger
) -> str:
    """Generate OpenAPI JSON format output."""
    spec = result.generate_openapi_spec(title=args.title, version=args.api_version)
    return json.dumps(spec, indent=2)


def _generate_html_output(
    result: DocumentationResult, args: argparse.Namespace,
    parser: argparse.ArgumentParser, logger: logging.Logger
) -> str:
    """Generate HTML playground format output."""
    spec = result.generate_openapi_spec(title=args.title, version=args.api_version)
    return PlaygroundGenerator().generate(spec)


def _generate_guide_output(
    result: DocumentationResult, args: argparse.Namespace,
    parser: argparse.ArgumentParser, logger: logging.Logger
) -> str:
    """Generate migration guide format output."""
    _validate_old_spec_required(args, parser)
    old_data = _load_old_spec_data(args.old_spec, parser, logger)
    new_spec = result.generate_openapi_spec(title=args.title, version=args.api_version)
    return MigrationGuideGenerator(old_data, new_spec).generate_markdown()


def _validate_old_spec_required(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Ensure old spec is provided for guide format."""
    if not args.old_spec:
        parser.error(
            f"[{ErrorCode.OLD_SPEC_REQUIRED}] --old-spec is required for guide format"
        )


def _load_old_spec_data(old_spec_path: str, parser: argparse.ArgumentParser, logger: logging.Logger) -> dict:
    """Load and validate old specification data."""
    old_path = Path(old_spec_path)
    if not old_path.exists():
        logger.error("Old spec file '%s' not found", old_path)
        parser.error(f"[{ErrorCode.OLD_SPEC_REQUIRED}] Old spec file '{old_path}' not found")
    
    try:
        return json.loads(old_path.read_text())
    except json.JSONDecodeError as exc:  # pragma: no cover - manual error path
        logger.error("Invalid JSON in old spec: %s", exc)
        parser.error(f"[{ErrorCode.OLD_SPEC_INVALID}] --old-spec is not valid JSON")


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


def _validate_app_path(
    app_path_str: str, parser: argparse.ArgumentParser, logger: logging.Logger
) -> Path:
    """Validate and normalize app path with security checks."""
    app_path = _parse_and_validate_app_path(app_path_str, parser, logger)
    if app_path is not None:  # Only check existence if path was successfully parsed
        _ensure_app_path_exists(app_path, parser, logger)
    return app_path


def _parse_and_validate_app_path(
    app_path_str: str, parser: argparse.ArgumentParser, logger: logging.Logger
) -> Path:
    """Parse and validate app path string with security checks."""
    try:
        _validate_app_path_not_empty(app_path_str)
        app_path = Path(app_path_str).resolve()
        _check_app_path_security(app_path_str)
        return app_path
    except (ValueError, OSError) as e:
        logger.error("Invalid app path '%s': %s", app_path_str, e)
        parser.error(f"[{ErrorCode.APP_NOT_FOUND}] Invalid app path: {e}")
        return None  # Never reached due to parser.error, but needed for type checker


def _validate_app_path_not_empty(app_path_str: str) -> None:
    """Ensure app path is not empty or whitespace-only."""
    if not app_path_str or not app_path_str.strip():
        raise ValueError("App path cannot be empty")


def _check_app_path_security(app_path_str: str) -> None:
    """Check for directory traversal patterns in app path."""
    if ".." in app_path_str or (app_path_str.startswith("/") and "/../" in app_path_str):
        raise ValueError("Path contains suspicious traversal patterns")


def _ensure_app_path_exists(app_path: Path, parser: argparse.ArgumentParser, logger: logging.Logger) -> None:
    """Verify that the app path exists."""
    if not app_path.exists():
        logger.error("App file '%s' not found", app_path)
        parser.error(f"[{ErrorCode.APP_NOT_FOUND}] App file '{app_path}' not found")


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


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    
    logger = _setup_main_logging(args)
    _configure_performance_tracking(args)
    app_path = _validate_and_prepare_app(args, parser, logger)
    
    output, result = _process_documentation_format(args, app_path, parser, logger)
    _handle_test_generation(args, result, parser, logger)
    _write_and_finalize_output(args, output, parser, logger)
    
    return 0


def _setup_main_logging(args: argparse.Namespace) -> logging.Logger:
    """Setup logging configuration based on CLI arguments."""
    log_level = _determine_log_level(args)
    use_colors = not args.no_color
    return _setup_logging(args.log_format, level=log_level, colored=use_colors)


def _determine_log_level(args: argparse.Namespace) -> int:
    """Determine appropriate logging level from arguments."""
    if args.verbose:
        return logging.DEBUG
    elif args.quiet:
        return logging.WARNING
    else:
        level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        return getattr(logging, level_name, logging.INFO)


def _configure_performance_tracking(args: argparse.Namespace) -> None:
    """Configure performance tracking if enabled."""
    from .utils import set_performance_tracking
    set_performance_tracking(args.performance_metrics)


def _validate_and_prepare_app(args: argparse.Namespace, parser: argparse.ArgumentParser, logger: logging.Logger) -> Path:
    """Validate application path and prepare for processing."""
    _show_progress("Validating application path", args.verbose)
    return _validate_app_path(args.app, parser, logger)


def _process_documentation_format(args: argparse.Namespace, app_path: Path, parser: argparse.ArgumentParser, logger: logging.Logger) -> tuple[str, Optional['DocumentationResult']]:
    """Process documentation based on format type."""
    if args.format == "graphql":
        _show_progress("Processing GraphQL schema", args.verbose)
        return _process_graphql_format(app_path, parser, logger), None
    else:
        _show_progress("Analyzing application structure", args.verbose)
        result = APIDocumentator().analyze_app(str(app_path))
        _show_progress("Generating documentation", args.verbose)
        output = _generate_output(result, args, parser, logger)
        return output, result


def _handle_test_generation(args: argparse.Namespace, result: Optional['DocumentationResult'], parser: argparse.ArgumentParser, logger: logging.Logger) -> None:
    """Generate test suite if requested and possible."""
    if result and args.tests:
        _show_progress("Generating test suite", args.verbose)
        tests_path = _validate_file_target(
            args.tests, "--tests", parser, logger, ErrorCode.TESTS_PATH_INVALID
        )
        tests_path.write_text(
            TestSuiteGenerator(result).generate_pytest(), encoding="utf-8"
        )


def _write_and_finalize_output(args: argparse.Namespace, output: str, parser: argparse.ArgumentParser, logger: logging.Logger) -> None:
    """Write output and log performance summary."""
    _show_progress("Writing output", args.verbose)
    _write_output(output, args.output, parser, logger)
    
    _log_performance_summary_if_enabled(args, logger)
    _show_completion_message_if_verbose(args)


def _log_performance_summary_if_enabled(args: argparse.Namespace, logger: logging.Logger) -> None:
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


def _show_completion_message_if_verbose(args: argparse.Namespace) -> None:
    """Show completion message if verbose mode is enabled."""
    if args.verbose:
        sys.stderr.write("✅ Documentation generation completed successfully!\n")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
