"""Command line interface for generating markdown docs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .documentator import APIDocumentator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate markdown API docs")
    parser.add_argument(
        "--app",
        required=True,
        help="Path to application source file",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="File to write markdown to (prints to stdout if omitted)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    app_path = Path(args.app)
    if not app_path.exists():
        parser.error(f"App file '{app_path}' not found")

    result = APIDocumentator().analyze_app(str(app_path))
    markdown = result.generate_markdown()

    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")
    else:
        sys.stdout.write(markdown)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
