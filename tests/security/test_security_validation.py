"""Security validation tests."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.mark.security
class TestSecurityValidation:
    """Security-focused tests for the application."""

    def test_input_validation_file_paths(self, temp_dir):
        """Test input validation for file paths."""
        from openapi_doc_generator.documentator import APIDocumentator
        
        documentator = APIDocumentator()
        
        # Test directory traversal attempts
        with pytest.raises(FileNotFoundError):
            documentator.analyze_app("../../../etc/passwd")
        
        with pytest.raises(FileNotFoundError):
            documentator.analyze_app("..\\..\\..\\windows\\system32\\config\\sam")
        
        # Test null bytes
        with pytest.raises(ValueError):
            documentator.analyze_app("test\x00.py")

    def test_no_arbitrary_code_execution(self, temp_dir):
        """Ensure no arbitrary code execution during analysis."""
        # Create a file with potentially dangerous code
        malicious_file = temp_dir / "malicious.py"
        malicious_file.write_text('''
import os
import subprocess

# This should not be executed during analysis
os.system("echo 'SECURITY_BREACH' > /tmp/security_test")
subprocess.run(["touch", "/tmp/security_breach"])

from flask import Flask
app = Flask(__name__)

@app.route("/test")
def test():
    return "test"
''')
        
        from openapi_doc_generator.documentator import APIDocumentator
        
        # Analyze the file - this should not execute the malicious code
        documentator = APIDocumentator()
        result = documentator.analyze_app(str(malicious_file))
        
        # Verify the malicious code was not executed
        assert not Path("/tmp/security_test").exists()
        assert not Path("/tmp/security_breach").exists()
        
        # But legitimate analysis should work
        assert result is not None

    def test_secrets_not_in_output(self, temp_dir):
        """Ensure secrets are not included in generated documentation."""
        # Create app with secrets
        app_with_secrets = temp_dir / "app_with_secrets.py"
        app_with_secrets.write_text('''
from flask import Flask

app = Flask(__name__)

# These should not appear in documentation
API_KEY = "sk-1234567890abcdef"
DATABASE_URL = "postgresql://user:password@localhost/db"
SECRET_TOKEN = "supersecrettoken123"

@app.route("/api/data")
def get_data():
    """Get data from API."""
    # Use the secret in code
    headers = {"Authorization": f"Bearer {API_KEY}"}
    return {"message": "data"}
''')
        
        from openapi_doc_generator.documentator import APIDocumentator
        
        documentator = APIDocumentator()
        result = documentator.analyze_app(str(app_with_secrets))
        
        # Generate outputs
        spec = result.generate_openapi_spec()
        markdown = result.generate_markdown()
        
        # Verify secrets don't appear in outputs
        spec_str = json.dumps(spec)
        assert "sk-1234567890abcdef" not in spec_str
        assert "password" not in spec_str.lower()
        assert "supersecrettoken123" not in spec_str
        
        assert "sk-1234567890abcdef" not in markdown
        assert "password" not in markdown.lower()
        assert "supersecrettoken123" not in markdown

    def test_dependency_security_scanning(self):
        """Test that dependencies are scanned for vulnerabilities."""
        # This would typically use safety or pip-audit
        # Here we just verify the tools are available
        try:
            result = subprocess.run(
                ["safety", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            assert result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Safety not available for testing")

    def test_file_size_limits(self, temp_dir):
        """Test that extremely large files are handled safely."""
        # Create a very large file (but not so large it fills disk)
        large_file = temp_dir / "large_app.py"
        
        # Write a moderately large file (1MB)
        content = "# Large file test\n" + "# Comment line\n" * 50000
        content += '''
from flask import Flask
app = Flask(__name__)

@app.route("/test")
def test():
    return "test"
'''
        
        large_file.write_text(content)
        
        from openapi_doc_generator.documentator import APIDocumentator
        
        documentator = APIDocumentator()
        
        # Should handle large files without crashing
        result = documentator.analyze_app(str(large_file))
        assert result is not None

    def test_malformed_syntax_handling(self, temp_dir):
        """Test handling of malformed Python syntax."""
        malformed_file = temp_dir / "malformed.py"
        malformed_file.write_text('''
# This file has syntax errors
from flask import Flask

app = Flask(__name__

@app.route("/test"  # Missing closing parenthesis
def test():
    return "test
''')
        
        from openapi_doc_generator.documentator import APIDocumentator
        
        documentator = APIDocumentator()
        
        # Should raise SyntaxError, not crash
        with pytest.raises(SyntaxError):
            documentator.analyze_app(str(malformed_file))

    def test_output_sanitization(self, temp_dir):
        """Test that output is properly sanitized."""
        # Create app with potentially dangerous docstrings
        dangerous_app = temp_dir / "dangerous_docs.py"
        dangerous_app.write_text('''
from flask import Flask

app = Flask(__name__)

@app.route("/api/test")
def test():
    """
    This endpoint is safe.
    <script>alert('XSS')</script>
    <img src="x" onerror="alert('XSS')">
    javascript:alert('XSS')
    """
    return {"message": "test"}
''')
        
        from openapi_doc_generator.documentator import APIDocumentator
        
        documentator = APIDocumentator()
        result = documentator.analyze_app(str(dangerous_app))
        
        # Generate outputs
        spec = result.generate_openapi_spec()
        markdown = result.generate_markdown()
        
        # Verify dangerous content is sanitized or escaped
        spec_str = json.dumps(spec)
        
        # XSS attempts should not be present as executable code
        assert "<script>" not in spec_str
        assert "javascript:" not in spec_str
        assert "onerror=" not in spec_str
        
        # Same for markdown output
        assert "<script>" not in markdown
        assert "javascript:" not in markdown

    def test_environment_isolation(self, temp_dir):
        """Test that the analysis runs in isolated environment."""
        # Create app that tries to modify environment
        env_app = temp_dir / "env_app.py"
        env_app.write_text('''
import os
import sys

# Try to modify environment
os.environ["MALICIOUS_VAR"] = "compromised"
sys.path.insert(0, "/malicious/path")

from flask import Flask

app = Flask(__name__)

@app.route("/test")
def test():
    return "test"
''')
        
        # Store original environment
        original_env = dict(os.environ)
        original_path = sys.path.copy()
        
        from openapi_doc_generator.documentator import APIDocumentator
        
        documentator = APIDocumentator()
        result = documentator.analyze_app(str(env_app))
        
        # Verify environment wasn't permanently modified
        assert "MALICIOUS_VAR" not in os.environ
        assert "/malicious/path" not in sys.path
        
        # Verify analysis still worked
        assert result is not None

    @pytest.mark.slow
    def test_denial_of_service_protection(self, temp_dir):
        """Test protection against DoS attacks via malicious input."""
        # Create file with deeply nested structures that could cause stack overflow
        dos_file = temp_dir / "dos_app.py"
        
        # Generate deeply nested function calls
        nested_content = "from flask import Flask\napp = Flask(__name__)\n\n"
        
        # Create many nested decorators (but not infinite)
        for i in range(100):  # Reasonable limit for testing
            nested_content += f"@app.route('/route_{i}')\n"
        
        nested_content += "def nested_route():\n    return 'test'\n"
        
        dos_file.write_text(nested_content)
        
        from openapi_doc_generator.documentator import APIDocumentator
        
        documentator = APIDocumentator()
        
        # Should handle without infinite recursion or excessive memory usage
        result = documentator.analyze_app(str(dos_file))
        assert result is not None

    def test_temporary_file_cleanup(self, temp_dir):
        """Test that temporary files are properly cleaned up."""
        import tempfile
        
        initial_temp_files = len(list(Path(tempfile.gettempdir()).glob("*")))
        
        app_file = temp_dir / "temp_test.py"
        app_file.write_text('''
from flask import Flask
app = Flask(__name__)

@app.route("/test")
def test():
    return "test"
''')
        
        from openapi_doc_generator.documentator import APIDocumentator
        
        # Run analysis multiple times
        for _ in range(5):
            documentator = APIDocumentator()
            result = documentator.analyze_app(str(app_file))
            spec = result.generate_openapi_spec()
        
        final_temp_files = len(list(Path(tempfile.gettempdir()).glob("*")))
        
        # Should not have significantly more temp files
        assert final_temp_files <= initial_temp_files + 5  # Allow some tolerance