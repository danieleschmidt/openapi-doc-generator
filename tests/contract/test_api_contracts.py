"""Contract tests for API compatibility."""

import json
import tempfile
from pathlib import Path

import pytest
from pact import Consumer, Provider

from openapi_doc_generator.cli import main


class TestAPIContracts:
    """Test API contracts using Pact."""
    
    @pytest.fixture
    def pact(self):
        """Create a Pact consumer."""
        pact = Consumer("doc-generator-client").has_pact_with(
            Provider("api-service"), 
            port=1234
        )
        pact.start()
        yield pact
        pact.stop()
    
    @pytest.mark.contract
    def test_openapi_spec_contract(self, pact):
        """Test that generated OpenAPI specs follow expected contract."""
        expected_response = {
            "openapi": "3.0.0",
            "info": {
                "title": "Test API",
                "version": "1.0.0"
            },
            "paths": {}
        }
        
        (pact
         .given("A valid Flask application exists")
         .upon_receiving("A request for OpenAPI specification")
         .with_request("GET", "/openapi.json")
         .will_respond_with(200, body=expected_response))
        
        with pact:
            # Test actual generation
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                
                app_file = tmpdir_path / "test_app.py"
                app_file.write_text("""
from flask import Flask
app = Flask(__name__)

@app.route("/test")
def test():
    return {"message": "test"}
""")
                
                output_file = tmpdir_path / "openapi.json"
                result = main([
                    "--app", str(app_file),
                    "--format", "openapi", 
                    "--title", "Test API",
                    "--output", str(output_file)
                ])
                
                assert result == 0
                
                with open(output_file) as f:
                    spec = json.load(f)
                
                # Verify contract compliance
                assert spec["openapi"] == "3.0.0"
                assert spec["info"]["title"] == "Test API"
                assert spec["info"]["version"] == "1.0.0"
                assert "paths" in spec
    
    @pytest.mark.contract 
    def test_markdown_format_contract(self):
        """Test markdown output format contract."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            app_file = tmpdir_path / "test_app.py"
            app_file.write_text("""
from flask import Flask
app = Flask(__name__)

@app.route("/users", methods=["GET", "POST"])
def users():
    '''Manage users.'''
    return {"users": []}
""")
            
            output_file = tmpdir_path / "api.md"
            result = main([
                "--app", str(app_file),
                "--format", "markdown",
                "--output", str(output_file)
            ])
            
            assert result == 0
            
            content = output_file.read_text()
            
            # Contract requirements for markdown
            assert content.startswith("# API Documentation")
            assert "## Endpoints" in content
            assert "/users" in content
            assert "GET" in content
            assert "POST" in content
            assert "Manage users" in content
    
    @pytest.mark.contract
    def test_cli_error_contract(self):
        """Test CLI error handling contract."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Test with non-existent file
            result = main([
                "--app", str(tmpdir_path / "nonexistent.py"),
                "--format", "openapi",
                "--output", str(tmpdir_path / "output.json")
            ])
            
            # Should return specific error code for missing app file
            assert result == 1  # CLI001 error code
    
    @pytest.mark.contract
    def test_plugin_interface_contract(self):
        """Test plugin interface contract."""
        from openapi_doc_generator.plugins.base import BasePlugin
        
        # Verify plugin interface contract
        required_methods = ["discover_routes", "get_framework_name"]
        
        for method_name in required_methods:
            assert hasattr(BasePlugin, method_name)
            assert callable(getattr(BasePlugin, method_name))
    
    @pytest.mark.contract
    def test_performance_metrics_contract(self):
        """Test performance metrics output contract."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            app_file = tmpdir_path / "test_app.py"
            app_file.write_text("""
from flask import Flask
app = Flask(__name__)

@app.route("/test")
def test():
    return {"test": True}
""")
            
            output_file = tmpdir_path / "openapi.json"
            
            # Capture performance metrics
            import io
            import sys
            from contextlib import redirect_stderr
            
            stderr_capture = io.StringIO()
            
            with redirect_stderr(stderr_capture):
                result = main([
                    "--app", str(app_file),
                    "--format", "openapi",
                    "--output", str(output_file),
                    "--performance-metrics",
                    "--log-format", "json"
                ])
            
            assert result == 0
            
            # Verify performance metrics contract
            stderr_output = stderr_capture.getvalue()
            lines = [line for line in stderr_output.split('\n') if line.strip()]
            
            # Should have JSON-formatted performance logs
            performance_logs = []
            for line in lines:
                try:
                    log_entry = json.loads(line)
                    if "Performance:" in log_entry.get("message", ""):
                        performance_logs.append(log_entry)
                except json.JSONDecodeError:
                    continue
            
            assert len(performance_logs) > 0
            
            # Verify log structure contract
            for log in performance_logs:
                assert "timestamp" in log
                assert "level" in log
                assert "logger" in log
                assert "message" in log
                assert "duration_ms" in log