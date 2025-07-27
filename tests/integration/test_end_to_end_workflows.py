"""End-to-end integration tests for complete workflows."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from openapi_doc_generator.cli import main as cli_main
from openapi_doc_generator.documentator import APIDocumentator


@pytest.mark.integration
class TestEndToEndWorkflows:
    """Test complete workflows from CLI to output generation."""

    def test_flask_app_complete_workflow(self, sample_flask_app, temp_dir):
        """Test complete workflow with Flask application."""
        # Create test Flask app file
        app_file = temp_dir / "test_app.py"
        app_file.write_text(sample_flask_app)
        
        # Generate OpenAPI spec
        documentator = APIDocumentator()
        result = documentator.analyze_app(str(app_file))
        
        # Verify result contains expected data
        assert result is not None
        
        # Generate OpenAPI specification
        spec = result.generate_openapi_spec()
        assert "openapi" in spec
        assert "paths" in spec
        
        # Generate markdown documentation
        markdown = result.generate_markdown()
        assert "# API Documentation" in markdown or "# " in markdown

    def test_fastapi_app_complete_workflow(self, sample_fastapi_app, temp_dir):
        """Test complete workflow with FastAPI application."""
        # Create test FastAPI app file
        app_file = temp_dir / "test_app.py"
        app_file.write_text(sample_fastapi_app)
        
        # Generate documentation
        documentator = APIDocumentator()
        result = documentator.analyze_app(str(app_file))
        
        # Verify result
        assert result is not None
        
        # Test OpenAPI generation
        spec = result.generate_openapi_spec()
        assert spec["info"]["title"] == "Test API"
        assert spec["info"]["version"] == "1.0.0"

    @pytest.mark.slow
    def test_cli_integration_openapi_output(self, sample_flask_app, temp_dir):
        """Test CLI integration with OpenAPI output."""
        # Create test app
        app_file = temp_dir / "test_app.py"
        output_file = temp_dir / "openapi.json"
        app_file.write_text(sample_flask_app)
        
        # Mock CLI arguments
        with patch('sys.argv', [
            'openapi-doc-generator',
            '--app', str(app_file),
            '--format', 'openapi',
            '--output', str(output_file)
        ]):
            try:
                cli_main()
            except SystemExit as e:
                # CLI should exit successfully
                assert e.code == 0
        
        # Verify output file was created
        assert output_file.exists()
        
        # Verify output content
        with open(output_file) as f:
            spec = json.load(f)
        
        assert "openapi" in spec
        assert "paths" in spec

    @pytest.mark.slow
    def test_cli_integration_markdown_output(self, sample_flask_app, temp_dir):
        """Test CLI integration with Markdown output."""
        # Create test app
        app_file = temp_dir / "test_app.py"
        output_file = temp_dir / "api.md"
        app_file.write_text(sample_flask_app)
        
        # Mock CLI arguments
        with patch('sys.argv', [
            'openapi-doc-generator',
            '--app', str(app_file),
            '--format', 'markdown',
            '--output', str(output_file)
        ]):
            try:
                cli_main()
            except SystemExit as e:
                assert e.code == 0
        
        # Verify output file was created
        assert output_file.exists()
        
        # Verify markdown content
        content = output_file.read_text()
        assert "#" in content  # Should contain headers

    def test_plugin_loading_integration(self, temp_dir):
        """Test that plugins are loaded correctly during integration."""
        from openapi_doc_generator.discovery import RouteDiscovery
        
        discovery = RouteDiscovery()
        
        # Verify plugins are loaded
        assert hasattr(discovery, '_plugins')
        
        # Test plugin detection with different frameworks
        flask_file = temp_dir / "flask_app.py"
        flask_file.write_text('''
from flask import Flask
app = Flask(__name__)

@app.route("/test")
def test():
    return "test"
''')
        
        # Should detect Flask framework
        result = discovery.discover_routes(str(flask_file))
        assert result is not None

    def test_error_handling_integration(self, temp_dir):
        """Test error handling in integration scenarios."""
        # Test with non-existent file
        documentator = APIDocumentator()
        
        with pytest.raises(FileNotFoundError):
            documentator.analyze_app("/non/existent/file.py")
        
        # Test with invalid Python file
        invalid_file = temp_dir / "invalid.py"
        invalid_file.write_text("invalid python syntax {{{")
        
        with pytest.raises(SyntaxError):
            documentator.analyze_app(str(invalid_file))

    @pytest.mark.performance
    def test_performance_integration(self, sample_flask_app, temp_dir, performance_timer):
        """Test performance of complete workflow."""
        # Create test app
        app_file = temp_dir / "test_app.py"
        app_file.write_text(sample_flask_app)
        
        # Time the complete workflow
        performance_timer.start()
        
        documentator = APIDocumentator()
        result = documentator.analyze_app(str(app_file))
        spec = result.generate_openapi_spec()
        markdown = result.generate_markdown()
        
        performance_timer.stop()
        
        # Verify performance is reasonable (adjust threshold as needed)
        assert performance_timer.elapsed < 5.0  # Should complete in under 5 seconds
        
        # Verify outputs are valid
        assert spec is not None
        assert markdown is not None