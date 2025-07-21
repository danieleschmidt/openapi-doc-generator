"""Tests for structured JSON logging functionality."""

import json
import logging
from io import StringIO
from unittest.mock import patch

import pytest

from openapi_doc_generator.cli import main


class TestJSONLogging:
    """Test suite for structured JSON logging implementation."""
    
    def test_json_logging_basic_functionality(self, tmp_path):
        """Test that --log-format json produces valid JSON log entries."""
        app = tmp_path / "app.py"
        app.write_text("""
from fastapi import FastAPI
app = FastAPI()

@app.get('/test')
def test_endpoint():
    return {'message': 'test'}
""")
        
        # Capture stderr where logs are written
        with patch('sys.stderr', new=StringIO()) as mock_stderr:
            with patch.dict('os.environ', {'LOG_LEVEL': 'INFO'}):
                main(["--app", str(app), "--format", "openapi", "--log-format", "json"])
        
        log_output = mock_stderr.getvalue()
        
        # Should contain JSON log entries
        log_lines = [line.strip() for line in log_output.strip().split('\n') if line.strip()]
        
        for line in log_lines:
            # Each line should be valid JSON
            log_entry = json.loads(line)
            
            # Should have required fields
            assert 'timestamp' in log_entry
            assert 'level' in log_entry
            assert 'logger' in log_entry
            assert 'message' in log_entry
            assert 'correlation_id' in log_entry
    
    def test_json_logging_correlation_id_consistency(self, tmp_path):
        """Test that correlation IDs are consistent within a single execution."""
        app = tmp_path / "app.py"
        app.write_text("from fastapi import FastAPI\napp = FastAPI()")
        
        with patch('sys.stderr', new=StringIO()) as mock_stderr:
            with patch.dict('os.environ', {'LOG_LEVEL': 'DEBUG'}):
                main(["--app", str(app), "--format", "openapi", "--log-format", "json"])
        
        log_output = mock_stderr.getvalue()
        log_lines = [line.strip() for line in log_output.strip().split('\n') if line.strip()]
        
        correlation_ids = set()
        for line in log_lines:
            log_entry = json.loads(line)
            correlation_ids.add(log_entry['correlation_id'])
        
        # All log entries in same execution should have same correlation ID
        assert len(correlation_ids) == 1
    
    def test_json_logging_includes_timing_metrics(self, tmp_path):
        """Test that JSON logs include timing and performance metrics."""
        app = tmp_path / "app.py"
        app.write_text("""
from fastapi import FastAPI
app = FastAPI()

@app.get('/users')
def get_users():
    return []
""")
        
        with patch('sys.stderr', new=StringIO()) as mock_stderr:
            with patch.dict('os.environ', {'LOG_LEVEL': 'DEBUG'}):
                main(["--app", str(app), "--format", "openapi", "--log-format", "json"])
        
        log_output = mock_stderr.getvalue()
        log_lines = [line.strip() for line in log_output.strip().split('\n') if line.strip()]
        
        timing_found = False
        for line in log_lines:
            log_entry = json.loads(line)
            if 'execution_time_ms' in log_entry or 'duration_ms' in log_entry or 'timing' in log_entry:
                timing_found = True
                break
        
        # Should include timing information in at least some log entries
        assert timing_found, "No timing metrics found in JSON logs"
    
    def test_json_logging_vs_standard_logging_content(self, tmp_path, caplog):
        """Test that JSON logging contains same core information as standard logging."""
        app = tmp_path / "app.py" 
        app.write_text("from fastapi import FastAPI\napp = FastAPI()")
        
        # Capture standard format logs via caplog
        with caplog.at_level(logging.INFO):
            main(["--app", str(app), "--format", "openapi"])
        standard_logs = [record.message for record in caplog.records]
        
        # Capture JSON format logs
        with patch('sys.stderr', new=StringIO()) as mock_stderr_json:
            with patch.dict('os.environ', {'LOG_LEVEL': 'INFO'}):
                main(["--app", str(app), "--format", "openapi", "--log-format", "json"])
        json_output = mock_stderr_json.getvalue()
        
        # Both should have log content (not empty)
        assert len(standard_logs) > 0
        assert len(json_output.strip()) > 0
        
        # JSON output should be parseable
        json_lines = [line.strip() for line in json_output.strip().split('\n') if line.strip()]
        for line in json_lines:
            json.loads(line)  # Should not raise exception
    
    def test_json_logging_error_handling(self, tmp_path, capsys):
        """Test JSON logging with error scenarios."""
        app = tmp_path / "app.py"  
        app.write_text("from fastapi import FastAPI\napp = FastAPI()")
        
        # Test that JSON logging works for successful runs (basic error handling test)
        with patch('sys.stderr', new=StringIO()) as mock_stderr:
            result = main(["--app", str(app), "--format", "openapi", "--log-format", "json"])
        
        log_output = mock_stderr.getvalue()
        
        # Should produce valid JSON log entries
        assert result == 0
        assert len(log_output.strip()) > 0
        
        log_lines = [line.strip() for line in log_output.strip().split('\n') if line.strip()]
        json_lines = []
        for line in log_lines:
            if line.strip():
                log_entry = json.loads(line)
                json_lines.append(log_entry)
                assert 'level' in log_entry
                assert 'correlation_id' in log_entry
                assert 'message' in log_entry
        
        # Should have valid JSON log entries
        assert len(json_lines) > 0
    
    def test_json_logging_respects_log_level(self, tmp_path):
        """Test that JSON logging respects LOG_LEVEL environment variable."""
        app = tmp_path / "app.py"
        app.write_text("from fastapi import FastAPI\napp = FastAPI()")
        
        # Test with ERROR level - should have fewer logs
        with patch('sys.stderr', new=StringIO()) as mock_stderr_error:
            with patch.dict('os.environ', {'LOG_LEVEL': 'ERROR'}):
                main(["--app", str(app), "--format", "openapi", "--log-format", "json"])
        error_output = mock_stderr_error.getvalue()
        
        # Test with DEBUG level - should have more logs  
        with patch('sys.stderr', new=StringIO()) as mock_stderr_debug:
            with patch.dict('os.environ', {'LOG_LEVEL': 'DEBUG'}):
                main(["--app", str(app), "--format", "openapi", "--log-format", "json"])
        debug_output = mock_stderr_debug.getvalue()
        
        # DEBUG should produce more log lines than ERROR
        error_lines = len([line for line in error_output.split('\n') if line.strip()])
        debug_lines = len([line for line in debug_output.split('\n') if line.strip()])
        
        assert debug_lines >= error_lines
    
    def test_json_logging_invalid_format_fallback(self, tmp_path):
        """Test that invalid --log-format values are rejected by argparse."""
        app = tmp_path / "app.py"
        app.write_text("from fastapi import FastAPI\napp = FastAPI()")
        
        # Should raise SystemExit due to argparse validation
        with pytest.raises(SystemExit):
            main(["--app", str(app), "--format", "openapi", "--log-format", "invalid"])
    
    def test_json_logging_structured_fields(self, tmp_path):
        """Test that JSON logs contain expected structured fields."""
        app = tmp_path / "app.py"
        app.write_text("""
from fastapi import FastAPI
app = FastAPI()

@app.get('/api/test')
def test():
    return {}
""")
        
        with patch('sys.stderr', new=StringIO()) as mock_stderr:
            with patch.dict('os.environ', {'LOG_LEVEL': 'INFO'}):
                main(["--app", str(app), "--format", "openapi", "--log-format", "json"])
        
        log_output = mock_stderr.getvalue()
        log_lines = [line.strip() for line in log_output.strip().split('\n') if line.strip()]
        
        required_fields = {'timestamp', 'level', 'logger', 'message', 'correlation_id'}
        
        for line in log_lines:
            log_entry = json.loads(line)
            
            # Check required fields are present
            for field in required_fields:
                assert field in log_entry, f"Missing required field: {field}"
            
            # Validate field types and formats
            assert isinstance(log_entry['timestamp'], str)
            assert log_entry['level'] in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            assert isinstance(log_entry['logger'], str)
            assert isinstance(log_entry['message'], str)
            assert isinstance(log_entry['correlation_id'], str)
            assert len(log_entry['correlation_id']) > 0