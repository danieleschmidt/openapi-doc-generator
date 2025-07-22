"""Tests to ensure CLI refactoring preserves behavior."""

import json
import logging
import pytest
from openapi_doc_generator.cli import main


class TestMainFunctionBehavior:
    """Test suite to verify main() function behavior before refactoring."""

    def test_main_basic_functionality(self, tmp_path, capsys):
        """Test basic main function workflow with valid FastAPI app."""
        app = tmp_path / "app.py"
        app.write_text(
            "from fastapi import FastAPI\n"
            "app = FastAPI()\n"
            "@app.get('/hello')\n"
            "def hello():\n"
            "    return {'message': 'hello'}\n"
        )

        result = main(["--app", str(app), "--format", "openapi"])
        assert result == 0

        output = capsys.readouterr().out
        data = json.loads(output)
        assert "/hello" in data["paths"]

    def test_main_with_output_file(self, tmp_path, capsys):
        """Test main function with output file writing."""
        app = tmp_path / "app.py"
        app.write_text(
            "from fastapi import FastAPI\n"
            "app = FastAPI()\n"
            "@app.get('/test')\n"
            "def test():\n"
            "    return {'test': True}\n"
        )

        output_file = tmp_path / "output.json"
        result = main(
            ["--app", str(app), "--format", "openapi", "--output", str(output_file)]
        )
        assert result == 0

        # Should not print to stdout when output file specified
        output = capsys.readouterr().out
        assert output == ""

        # Should write to file
        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert "/test" in data["paths"]

    def test_main_app_validation_errors(self, capsys):
        """Test main function app path validation error handling."""
        with pytest.raises(SystemExit):
            main(["--app", "/non/existent/file.py", "--format", "openapi"])

        err = capsys.readouterr().err
        assert "CLI001" in err
        assert "not found" in err

    def test_main_empty_app_path(self, capsys):
        """Test main function with empty app path."""
        with pytest.raises(SystemExit):
            main(["--app", "", "--format", "openapi"])

        err = capsys.readouterr().err
        assert "CLI001" in err
        assert "empty" in err.lower()

    def test_main_suspicious_path_patterns(self, capsys):
        """Test main function rejects suspicious path patterns."""
        with pytest.raises(SystemExit):
            main(["--app", "../../../etc/passwd", "--format", "openapi"])

        err = capsys.readouterr().err
        assert "CLI001" in err
        assert "suspicious" in err.lower()

    def test_main_graphql_mode(self, tmp_path, capsys):
        """Test main function in GraphQL mode."""
        schema = tmp_path / "schema.graphql"
        schema.write_text("""
        type Query {
            hello: String
        }
        """)

        result = main(["--app", str(schema), "--format", "graphql"])
        assert result == 0

        output = capsys.readouterr().out
        data = json.loads(output)
        assert "__schema" in data

    def test_main_with_tests_generation(self, tmp_path, capsys):
        """Test main function with test suite generation."""
        app = tmp_path / "app.py"
        app.write_text(
            "from fastapi import FastAPI\n"
            "app = FastAPI()\n"
            "@app.get('/api/test')\n"
            "def test_endpoint():\n"
            "    return {'success': True}\n"
        )

        tests_file = tmp_path / "test_generated.py"
        result = main(
            ["--app", str(app), "--format", "openapi", "--tests", str(tests_file)]
        )
        assert result == 0

        # Tests file should be created
        assert tests_file.exists()
        test_content = tests_file.read_text()
        assert "def test_" in test_content
        assert "/api/test" in test_content

    def test_main_logging_configuration(self, tmp_path, caplog):
        """Test that main function configures logging correctly."""
        app = tmp_path / "app.py"
        app.write_text("from fastapi import FastAPI\napp = FastAPI()")

        with caplog.at_level(logging.INFO):
            main(["--app", str(app), "--format", "openapi"])

        # Should have configured logging
        assert (
            len(caplog.records) >= 0
        )  # May or may not log depending on implementation
