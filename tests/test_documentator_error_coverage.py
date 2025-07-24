"""Tests for documentator.py error handling and edge cases to improve coverage."""

import pytest
from unittest.mock import patch, MagicMock

from openapi_doc_generator.documentator import APIDocumentator


class TestDocumentatorErrorHandling:
    """Test error handling and edge cases in APIDocumentator."""

    def test_analyze_app_handles_schema_file_not_found(self, tmp_path):
        """Test that analyze_app gracefully handles when no schema files are found."""
        # Create a simple app file
        app_file = tmp_path / "app.py"
        app_file.write_text("""
from flask import Flask
app = Flask(__name__)

@app.route("/test")
def test_route():
    return "test"
""")
        
        # Mock SchemaInferer to raise FileNotFoundError
        with patch('openapi_doc_generator.documentator.SchemaInferer') as mock_schema_inferer:
            mock_instance = MagicMock()
            mock_instance.infer.side_effect = FileNotFoundError("No Python files found")
            mock_schema_inferer.return_value = mock_instance
            
            # This should not raise an exception, but handle it gracefully
            documentator = APIDocumentator()
            result = documentator.analyze_app(str(app_file))
            
            # Should return result with empty schemas list
            assert result is not None
            assert result.schemas == []
            assert len(result.routes) >= 0  # May have routes from route discovery

    def test_analyze_app_with_nonexistent_file(self):
        """Test analyze_app with completely nonexistent app file."""
        documentator = APIDocumentator()
        
        # This should raise an exception since the app file doesn't exist
        with pytest.raises(FileNotFoundError):
            documentator.analyze_app("/nonexistent/app.py")

    def test_analyze_app_logs_schema_not_found_message(self, tmp_path, caplog):
        """Test that appropriate log message is created when schemas not found."""
        import logging
        caplog.set_level(logging.INFO)  # Ensure INFO level logs are captured
        
        app_file = tmp_path / "app.py"
        app_file.write_text("""
from flask import Flask
app = Flask(__name__)

@app.route("/test")
def test():
    return "test"
""")
        
        with patch('openapi_doc_generator.documentator.SchemaInferer') as mock_schema_inferer:
            mock_instance = MagicMock()
            mock_instance.infer.side_effect = FileNotFoundError("No models found")
            mock_schema_inferer.return_value = mock_instance
            
            documentator = APIDocumentator()
            result = documentator.analyze_app(str(app_file))
            
            # Check that appropriate log message was created
            assert "No models found in" in caplog.text
            assert str(app_file) in caplog.text  
            assert result.schemas == []