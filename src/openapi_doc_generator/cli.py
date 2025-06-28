"""Command line interface for generating documentation artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import json

from .documentator import APIDocumentator
from .playground import PlaygroundGenerator
from .graphql import GraphQLSchema
from .testsuite import TestSuiteGenerator
from .migration import MigrationGuideGenerator
from . import __version__


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
        help=(
            "Output format: markdown (default), openapi, html, graphql, or guide"
        ),
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
        default="API",
        help="Title for generated API documentation",
    )
    parser.add_argument(
        "--api-version",
        default="1.0.0",
        help="Version string for generated API documentation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    app_path = Path(args.app)
    if not app_path.exists():
        parser.error(f"App file '{app_path}' not found")

    if args.format == "graphql":
        data = GraphQLSchema(str(app_path)).introspect()
        output = json.dumps(data, indent=2)
        result = None
    else:
        result = APIDocumentator().analyze_app(str(app_path))

        if args.format == "markdown":
            output = result.generate_markdown(title=args.title, version=args.api_version)
        elif args.format == "openapi":
            spec = result.generate_openapi_spec(title=args.title, version=args.api_version)
            output = json.dumps(spec, indent=2)
        elif args.format == "html":
            spec = result.generate_openapi_spec(title=args.title, version=args.api_version)
            output = PlaygroundGenerator().generate(spec)
        elif args.format == "guide":
            if not args.old_spec:
                parser.error("--old-spec is required for guide format")
            old_data = json.loads(Path(args.old_spec).read_text())
            new_spec = result.generate_openapi_spec(title=args.title, version=args.api_version)
            output = MigrationGuideGenerator(
                old_data, new_spec
            ).generate_markdown()
        else:  # pragma: no cover - argparse enforces choices
            parser.error(f"Unknown format '{args.format}'")

    if result and args.tests:
        Path(args.tests).write_text(
            TestSuiteGenerator(result).generate_pytest(),
            encoding="utf-8",
        )

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
