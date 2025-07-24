"""Comprehensive CLI test coverage to improve cli.py from 24% to >90%."""

import pytest
import argparse
import logging
from unittest.mock import MagicMock

from openapi_doc_generator.cli import (
    build_parser,
    _validate_file_target,
    _generate_output,
    _setup_logging,
    _validate_app_path,
    _process_graphql_format,
    _write_output,
    main
)


class TestCLIParserFunctions:
    """Test CLI parser and argument validation functions."""

    def test_build_parser_creates_valid_parser(self):
        """Test that build_parser creates a properly configured ArgumentParser."""
        parser = build_parser()
        assert isinstance(parser, argparse.ArgumentParser)
        
        # Test that all expected arguments are present
        help_text = parser.format_help()
        assert "--app" in help_text
        assert "--version" in help_text
        assert "--output" in help_text
        assert "--format" in help_text
        assert "--old-spec" in help_text
        
    def test_build_parser_default_values(self):
        """Test default argument values."""
        parser = build_parser()
        
        # Test with minimal arguments
        args = parser.parse_args(["--app", "test.py"])
        assert args.app == "test.py"
        assert args.format == "markdown"
        assert args.output is None
        assert args.performance_metrics is False
        
    def test_build_parser_format_choices(self):
        """Test that format argument accepts valid choices."""
        parser = build_parser()
        
        valid_formats = ["markdown", "openapi", "html", "graphql", "guide"]
        for fmt in valid_formats:
            args = parser.parse_args(["--app", "test.py", "--format", fmt])
            assert args.format == fmt
            
    def test_build_parser_invalid_format(self):
        """Test that invalid format raises error."""
        parser = build_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args(["--app", "test.py", "--format", "invalid"])


class TestCLIValidationFunctions:
    """Test CLI validation helper functions."""

    def test_validate_file_target_valid_path(self, tmp_path):
        """Test _validate_file_target with valid output path."""
        output_file = tmp_path / "output.md"
        parser = MagicMock()
        logger = MagicMock()
        
        # Should not raise any exceptions
        result = _validate_file_target(str(output_file), "--output", parser, logger, "CLI004")
        assert result == output_file.resolve()
        parser.error.assert_not_called()
        
    def test_validate_file_target_invalid_directory(self):
        """Test _validate_file_target with invalid directory."""
        parser = MagicMock()
        logger = MagicMock()
        invalid_path = "/nonexistent/directory/file.md"
        
        _validate_file_target(invalid_path, "--output", parser, logger, "CLI004")
        parser.error.assert_called_once()
        
    def test_validate_file_target_existing_directory_as_file(self, tmp_path):
        """Test _validate_file_target when target is existing directory."""
        parser = MagicMock()
        logger = MagicMock()
        
        _validate_file_target(str(tmp_path), "--output", parser, logger, "CLI004")
        parser.error.assert_called_once()
        
    def test_validate_app_path_valid_file(self, tmp_path):
        """Test _validate_app_path with valid Python file."""
        app_file = tmp_path / "app.py"
        app_file.write_text("print('hello')")
        
        parser = MagicMock()
        logger = MagicMock()
        
        result = _validate_app_path(str(app_file), parser, logger)
        assert result == app_file
        parser.error.assert_not_called()
        
    def test_validate_app_path_nonexistent_file(self):
        """Test _validate_app_path with nonexistent file."""
        parser = MagicMock()
        logger = MagicMock()
        
        _validate_app_path("/nonexistent/app.py", parser, logger)
        parser.error.assert_called_once()
        
    def test_validate_app_path_directory_instead_of_file(self, tmp_path):
        """Test _validate_app_path when path is directory (should succeed)."""
        parser = MagicMock()
        logger = MagicMock()
        
        # The function allows directories - it only checks existence
        result = _validate_app_path(str(tmp_path), parser, logger)
        assert result == tmp_path.resolve()
        parser.error.assert_not_called()
        
    def test_validate_app_path_path_traversal_attempt(self):
        """Test _validate_app_path blocks path traversal attempts."""
        parser = MagicMock()
        logger = MagicMock()
        
        _validate_app_path("../../../etc/passwd", parser, logger)
        parser.error.assert_called_once()


class TestCLIOutputFunctions:
    """Test CLI output generation and writing functions."""

    def test_generate_output_markdown_format(self, tmp_path):
        """Test _generate_output with markdown format."""
        from openapi_doc_generator.documentator import DocumentationResult
        from openapi_doc_generator.discovery import RouteInfo
        
        # Create mock documentation result
        routes = [RouteInfo(path="/test", methods=["GET"], name="test")]
        result = DocumentationResult(routes=routes, schemas=[])
        
        args = MagicMock()
        args.format = "markdown"
        args.title = "Test API"
        args.api_version = "1.0.0"
        
        parser = MagicMock()
        logger = MagicMock()
        
        output = _generate_output(result, args, parser, logger)
        assert isinstance(output, str)
        assert "# Test API" in output
        
    def test_generate_output_openapi_format(self, tmp_path):
        """Test _generate_output with OpenAPI format."""
        from openapi_doc_generator.documentator import DocumentationResult
        from openapi_doc_generator.discovery import RouteInfo
        
        routes = [RouteInfo(path="/api/test", methods=["GET"], name="test")]
        result = DocumentationResult(routes=routes, schemas=[])
        
        args = MagicMock()
        args.format = "openapi"
        args.title = "Test API"
        args.api_version = "1.0.0"
        
        parser = MagicMock()
        logger = MagicMock()
        
        output = _generate_output(result, args, parser, logger)
        assert isinstance(output, str)
        # Should be valid JSON
        import json
        parsed = json.loads(output)
        assert parsed["info"]["title"] == "Test API"
        assert parsed["info"]["version"] == "1.0.0"
        
    def test_generate_output_html_format(self, tmp_path):
        """Test _generate_output with HTML playground format."""
        from openapi_doc_generator.documentator import DocumentationResult
        from openapi_doc_generator.discovery import RouteInfo
        
        routes = [RouteInfo(path="/test", methods=["GET"], name="test")]
        result = DocumentationResult(routes=routes, schemas=[])
        
        args = MagicMock()
        args.format = "html"
        args.title = "Test API"
        args.api_version = "1.0.0"
        
        parser = MagicMock()
        logger = MagicMock()
        
        output = _generate_output(result, args, parser, logger)
        assert isinstance(output, str)
        assert "<html" in output.lower()
        
    def test_generate_output_guide_format_missing_old_spec(self, tmp_path):
        """Test _generate_output with guide format but missing old-spec."""
        from openapi_doc_generator.documentator import DocumentationResult
        from openapi_doc_generator.discovery import RouteInfo
        
        routes = [RouteInfo(path="/test", methods=["GET"], name="test")]
        result = DocumentationResult(routes=routes, schemas=[])
        
        args = MagicMock()
        args.format = "guide"
        args.old_spec = None
        
        parser = MagicMock()
        parser.error.side_effect = SystemExit(2)
        logger = MagicMock()
        
        with pytest.raises(SystemExit):
            _generate_output(result, args, parser, logger)
        parser.error.assert_called_once()
        
    def test_generate_output_guide_format_nonexistent_old_spec(self, tmp_path):
        """Test _generate_output with guide format but nonexistent old-spec file."""
        from openapi_doc_generator.documentator import DocumentationResult
        from openapi_doc_generator.discovery import RouteInfo
        
        routes = [RouteInfo(path="/test", methods=["GET"], name="test")]
        result = DocumentationResult(routes=routes, schemas=[])
        
        args = MagicMock()
        args.format = "guide"
        args.old_spec = "/nonexistent/spec.json"
        args.title = "Test API"
        args.api_version = "1.0.0"
        
        parser = MagicMock()
        parser.error.side_effect = SystemExit(2)
        logger = MagicMock()
        
        with pytest.raises(SystemExit):
            _generate_output(result, args, parser, logger)
        parser.error.assert_called_once()
        
    def test_generate_output_unknown_format(self, tmp_path):
        """Test _generate_output with unknown format."""
        from openapi_doc_generator.documentator import DocumentationResult
        from openapi_doc_generator.discovery import RouteInfo
        
        routes = [RouteInfo(path="/test", methods=["GET"], name="test")]
        result = DocumentationResult(routes=routes, schemas=[])
        
        args = MagicMock()
        args.format = "unknown"
        
        parser = MagicMock()
        parser.error.side_effect = SystemExit(2)
        logger = MagicMock()
        
        with pytest.raises(SystemExit):
            _generate_output(result, args, parser, logger)
        parser.error.assert_called_once()
        
    def test_process_graphql_format_valid_schema(self, tmp_path):
        """Test _process_graphql_format with valid GraphQL schema."""
        schema_file = tmp_path / "schema.graphql"
        schema_file.write_text("""
type Query {
    hello: String
}
""")
        
        parser = MagicMock()
        logger = MagicMock()
        
        result = _process_graphql_format(schema_file, parser, logger)
        assert isinstance(result, str)
        # Should be valid JSON containing schema introspection
        import json
        parsed = json.loads(result)
        assert "__schema" in parsed
        
    def test_write_output_to_file(self, tmp_path):
        """Test _write_output writes content to file."""
        output_file = tmp_path / "output.txt"
        content = "Test content"
        parser = MagicMock()
        logger = MagicMock()
        
        _write_output(content, str(output_file), parser, logger)
        
        assert output_file.exists()
        assert output_file.read_text() == content
        
    def test_write_output_to_stdout(self, capsys):
        """Test _write_output prints to stdout when no file specified."""
        content = "Test stdout content"
        parser = MagicMock()
        logger = MagicMock()
        
        _write_output(content, None, parser, logger)
        
        captured = capsys.readouterr()
        assert content in captured.out


class TestCLILoggingFunctions:
    """Test CLI logging setup functions."""

    def test_setup_logging_standard_format(self):
        """Test _setup_logging with standard format."""
        logger = _setup_logging("standard")
        assert isinstance(logger, logging.Logger)
        # Logger name will vary depending on context
        assert "openapi_doc_generator" in logger.name
        
    def test_setup_logging_json_format(self):
        """Test _setup_logging with JSON format."""
        logger = _setup_logging("json")
        assert isinstance(logger, logging.Logger)
        # Logger name will vary depending on context
        assert "openapi_doc_generator" in logger.name


class TestCLIMainFunction:
    """Test main CLI entry point function."""

    def test_main_with_valid_markdown_args(self, tmp_path):
        """Test main function with valid markdown arguments."""
        app_file = tmp_path / "app.py"
        app_file.write_text("""
from flask import Flask
app = Flask(__name__)

@app.route('/hello')
def hello():
    return 'Hello World'
""")
        
        result = main(["--app", str(app_file), "--format", "markdown"])
        assert result == 0
        
    def test_main_with_output_file(self, tmp_path):
        """Test main function writing to output file."""
        app_file = tmp_path / "app.py"
        app_file.write_text("""
from flask import Flask
app = Flask(__name__)

@app.route('/test')
def test():
    return 'test'
""")
        
        output_file = tmp_path / "output.md"
        
        result = main([
            "--app", str(app_file),
            "--format", "markdown",
            "--output", str(output_file)
        ])
        
        assert result == 0
        assert output_file.exists()
        content = output_file.read_text()
        assert "# API" in content
        
    def test_main_with_custom_title_and_version(self, tmp_path):
        """Test main function with custom API title and version."""
        app_file = tmp_path / "app.py"
        app_file.write_text("""
from flask import Flask
app = Flask(__name__)

@app.route('/custom')
def custom():
    return 'custom'
""")
        
        output_file = tmp_path / "output.md"
        
        result = main([
            "--app", str(app_file),
            "--format", "markdown",
            "--output", str(output_file),
            "--title", "Custom API",
            "--api-version", "2.0.0"
        ])
        
        assert result == 0
        content = output_file.read_text()
        assert "# Custom API" in content
        
    def test_main_with_invalid_app_path(self):
        """Test main function with invalid app path."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--app", "/nonexistent/app.py"])
        assert exc_info.value.code == 2  # argparse error code
        
    def test_main_with_tests_argument(self, tmp_path):
        """Test main function with tests argument."""
        app_file = tmp_path / "app.py"
        app_file.write_text("""
from flask import Flask
app = Flask(__name__)

@app.route('/test')
def test():
    return 'test'
""")
        
        tests_file = tmp_path / "test_generated.py"
        
        result = main([
            "--app", str(app_file),
            "--tests", str(tests_file)
        ])
        
        assert result == 0
        assert tests_file.exists()
        test_content = tests_file.read_text()
        assert "def test_" in test_content
        assert "requests." in test_content
        
    def test_main_with_graphql_format(self, tmp_path):
        """Test main function with GraphQL format."""
        schema_file = tmp_path / "schema.graphql" 
        schema_file.write_text("""
type Query {
    hello: String
}
""")
        
        result = main([
            "--app", str(schema_file),
            "--format", "graphql"
        ])
        
        assert result == 0
        
    def test_main_with_performance_metrics(self, tmp_path):
        """Test main function with performance metrics enabled."""
        app_file = tmp_path / "app.py"
        app_file.write_text("""
from flask import Flask
app = Flask(__name__)

@app.route('/test')
def test():
    return 'test'
""")
        
        result = main([
            "--app", str(app_file),
            "--performance-metrics"
        ])
        
        assert result == 0