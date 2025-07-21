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
    return parser


def _validate_file_target(
    path_str: str, 
    flag: str, 
    parser: argparse.ArgumentParser, 
    logger: logging.Logger, 
    code: str
) -> Path:
    """Ensure a CLI file argument points to a writable file path."""
    # Normalize and validate the path to prevent directory traversal
    path = Path(path_str).resolve()
    
    # Check for suspicious path patterns
    if ".." in path_str or (path_str.startswith("/") and "/../" in path_str):
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
        if not args.old_spec:
            parser.error(f"[{ErrorCode.OLD_SPEC_REQUIRED}] --old-spec is required for guide format")
        old_path = Path(args.old_spec)
        if not old_path.exists():
            logger.error("Old spec file '%s' not found", old_path)
            parser.error(f"[{ErrorCode.OLD_SPEC_REQUIRED}] Old spec file '{old_path}' not found")
        try:
            old_data = json.loads(old_path.read_text())
        except json.JSONDecodeError as exc:  # pragma: no cover - manual error path
            logger.error("Invalid JSON in old spec: %s", exc)
            parser.error(f"[{ErrorCode.OLD_SPEC_INVALID}] --old-spec is not valid JSON")
        new_spec = result.generate_openapi_spec(
            title=args.title, version=args.api_version
        )
        return MigrationGuideGenerator(old_data, new_spec).generate_markdown()
    parser.error(f"Unknown format '{args.format}'")


def _setup_logging() -> logging.Logger:
    """Configure logging and return logger instance."""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")
    return logging.getLogger(__name__)


def _validate_app_path(app_path_str: str, parser: argparse.ArgumentParser, logger: logging.Logger) -> Path:
    """Validate and normalize app path with security checks."""
    try:
        # Check for empty or whitespace-only paths
        if not app_path_str or not app_path_str.strip():
            raise ValueError("App path cannot be empty")
        app_path = Path(app_path_str).resolve()
        # Security check - prevent obvious directory traversal patterns
        if ".." in app_path_str or app_path_str.startswith("/") and "/../" in app_path_str:
            raise ValueError("Path contains suspicious traversal patterns")
    except (ValueError, OSError) as e:
        logger.error("Invalid app path '%s': %s", app_path_str, e)
        parser.error(f"[{ErrorCode.APP_NOT_FOUND}] Invalid app path: {e}")
    
    if not app_path.exists():
        logger.error("App file '%s' not found", app_path)
        parser.error(f"[{ErrorCode.APP_NOT_FOUND}] App file '{app_path}' not found")
    
    return app_path


def _process_graphql_format(app_path: Path, parser: argparse.ArgumentParser, logger: logging.Logger) -> str:
    """Process GraphQL format and return JSON output."""
    try:
        data = GraphQLSchema(str(app_path)).introspect()
        return json.dumps(data, indent=2)
    except ValueError as e:
        logger.error("GraphQL schema error: %s", e)
        parser.error(f"[{ErrorCode.APP_NOT_FOUND}] GraphQL schema error: {e}")


def _write_output(output: str, output_path: Optional[str], parser: argparse.ArgumentParser, logger: logging.Logger) -> None:
    """Write output to file or stdout."""
    if output_path:
        out_path = _validate_file_target(
            output_path, "--output", parser, logger, ErrorCode.OUTPUT_PATH_INVALID
        )
        out_path.write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)


def main(argv: Optional[list[str]] = None) -> int:
    logger = _setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    app_path = _validate_app_path(args.app, parser, logger)

    if args.format == "graphql":
        output = _process_graphql_format(app_path, parser, logger)
        result = None
    else:
        result = APIDocumentator().analyze_app(str(app_path))
        output = _generate_output(result, args, parser, logger)

    if result and args.tests:
        tests_path = _validate_file_target(
            args.tests, "--tests", parser, logger, ErrorCode.TESTS_PATH_INVALID
        )
        tests_path.write_text(
            TestSuiteGenerator(result).generate_pytest(), encoding="utf-8"
        )

    _write_output(output, args.output, parser, logger)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
