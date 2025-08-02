#!/usr/bin/env python3
"""
Development task runner for OpenAPI Doc Generator.
Provides common development tasks in a unified interface.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], cwd: Path = None) -> int:
    """Run a command and return exit code."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd or Path.cwd())
    return result.returncode


def test(args):
    """Run test suite."""
    cmd = ["pytest", "tests/"]
    if args.verbose:
        cmd.append("-v")
    if args.coverage:
        cmd = ["coverage", "run", "-m"] + cmd
    return run_command(cmd)


def lint(args):
    """Run code linting."""
    exit_code = 0
    
    # Run ruff
    exit_code |= run_command(["ruff", "check", "src/", "tests/"])
    
    # Run mypy
    exit_code |= run_command(["mypy", "src/"])
    
    # Run bandit security check
    exit_code |= run_command(["bandit", "-r", "src/", "-f", "json", "-o", "security_results.json"])
    
    return exit_code


def format_code(args):
    """Format code with black and ruff."""
    exit_code = 0
    exit_code |= run_command(["black", "src/", "tests/"])
    exit_code |= run_command(["ruff", "check", "--fix", "src/", "tests/"])
    return exit_code


def build(args):
    """Build the project."""
    exit_code = 0
    
    # Clean previous builds
    exit_code |= run_command(["rm", "-rf", "dist/", "build/", "*.egg-info"])
    
    # Build package
    exit_code |= run_command(["python", "-m", "build"])
    
    return exit_code


def clean(args):
    """Clean build artifacts and cache."""
    artifacts = [
        "dist/", "build/", "*.egg-info", "__pycache__/", ".pytest_cache/",
        ".coverage", "htmlcov/", ".mypy_cache/", ".ruff_cache/"
    ]
    
    for pattern in artifacts:
        run_command(["find", ".", "-name", pattern, "-exec", "rm", "-rf", "{}", "+"])
    
    return 0


def security_check(args):
    """Run security checks."""
    exit_code = 0
    
    # Run safety check
    exit_code |= run_command(["safety", "check"])
    
    # Run pip-audit
    exit_code |= run_command(["pip-audit"])
    
    # Run bandit
    exit_code |= run_command(["python", "scripts/security_scan.py"])
    
    return exit_code


def dev_server(args):
    """Start development server with file watching."""
    print("Starting development mode...")
    print("Use examples/app.py for testing")
    
    # Run example generation
    cmd = [
        "python", "-m", "openapi_doc_generator.cli",
        "--app", "examples/app.py",
        "--format", "openapi",
        "--output", "dev_output.json",
        "--verbose"
    ]
    
    if args.performance:
        cmd.extend(["--performance-metrics", "--log-format", "json"])
    
    return run_command(cmd)


def docker_build(args):
    """Build Docker image."""
    tag = args.tag or "openapi-doc-generator:dev"
    cmd = ["docker", "build", "-t", tag, "."]
    return run_command(cmd)


def main():
    parser = argparse.ArgumentParser(description="Development task runner")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    test_parser.add_argument("-c", "--coverage", action="store_true", help="Run with coverage")
    test_parser.set_defaults(func=test)
    
    # Lint command
    lint_parser = subparsers.add_parser("lint", help="Run linting")
    lint_parser.set_defaults(func=lint)
    
    # Format command
    format_parser = subparsers.add_parser("format", help="Format code")
    format_parser.set_defaults(func=format_code)
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build package")
    build_parser.set_defaults(func=build)
    
    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean artifacts")
    clean_parser.set_defaults(func=clean)
    
    # Security command
    security_parser = subparsers.add_parser("security", help="Security checks")
    security_parser.set_defaults(func=security_check)
    
    # Dev server command
    dev_parser = subparsers.add_parser("dev", help="Development server")
    dev_parser.add_argument("-p", "--performance", action="store_true", help="Enable performance metrics")
    dev_parser.set_defaults(func=dev_server)
    
    # Docker build command
    docker_parser = subparsers.add_parser("docker", help="Build Docker image")
    docker_parser.add_argument("-t", "--tag", help="Docker tag")
    docker_parser.set_defaults(func=docker_build)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())