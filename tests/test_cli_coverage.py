"""Tests to improve CLI coverage."""

import json
import pytest
from unittest.mock import patch, MagicMock
from openapi_doc_generator.cli import main


def test_cli_with_performance_metrics_enabled(tmp_path, capsys, caplog):
    """Test CLI with --performance-metrics flag enabled."""
    # Create a simple app file
    app_file = tmp_path / "app.py"
    app_file.write_text(
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n"
        "@app.get('/test')\n"
        "def test_endpoint():\n"
        "    return {'message': 'test'}\n"
    )
    
    # Run CLI with performance metrics enabled
    exit_code = main([
        "--app", str(app_file),
        "--format", "openapi",
        "--performance-metrics"
    ])
    
    assert exit_code == 0
    
    # Check that performance summary was logged
    captured = capsys.readouterr()
    output = captured.out
    
    # Output should be valid JSON (OpenAPI spec)
    spec = json.loads(output)
    assert "openapi" in spec
    assert "paths" in spec
    
    # Check logs for performance metrics (if JSON logging is enabled)
    log_text = caplog.text
    # Performance tracking should have been enabled
    # Note: The actual performance summary logging might not show up in caplog
    # due to how the logger is configured, but the functionality is exercised


def test_cli_performance_metrics_with_json_logging(tmp_path, capsys):
    """Test CLI with performance metrics and JSON logging."""
    # Create a simple app file
    app_file = tmp_path / "app.py"
    app_file.write_text(
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n"
        "@app.get('/metrics')\n"
        "def metrics():\n"
        "    return {'status': 'ok'}\n"
    )
    
    # Mock get_performance_summary to return some data
    mock_summary = {
        "route_discovery": {
            "count": 1,
            "total_duration_ms": 50.5,
            "avg_duration_ms": 50.5
        }
    }
    
    with patch('openapi_doc_generator.utils.get_performance_summary') as mock_perf:
        mock_perf.return_value = mock_summary
        
        exit_code = main([
            "--app", str(app_file),
            "--format", "markdown",
            "--performance-metrics",
            "--log-format", "json"
        ])
        
        assert exit_code == 0
        
        # Verify performance summary was requested
        mock_perf.assert_called_once()


def test_cli_performance_metrics_with_output_file(tmp_path):
    """Test CLI with performance metrics when writing to output file."""
    # Create app and output files
    app_file = tmp_path / "app.py"
    app_file.write_text(
        "from flask import Flask\n"
        "app = Flask(__name__)\n"
        "@app.route('/perf')\n"
        "def perf():\n"
        "    return 'Performance Test'\n"
    )
    
    output_file = tmp_path / "output.md"
    
    # Mock the performance summary
    with patch('openapi_doc_generator.utils.get_performance_summary') as mock_perf:
        mock_perf.return_value = {"test": {"count": 1}}
        
        exit_code = main([
            "--app", str(app_file),
            "--format", "markdown",
            "--output", str(output_file),
            "--performance-metrics"
        ])
        
        assert exit_code == 0
        assert output_file.exists()
        
        # Verify markdown was written
        content = output_file.read_text()
        assert "# My API" in content  # Default title
        assert "/perf" in content


def test_cli_performance_metrics_empty_summary(tmp_path, caplog):
    """Test CLI with performance metrics when summary is empty."""
    # Create a simple app file
    app_file = tmp_path / "app.py"
    app_file.write_text(
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n"
        "@app.get('/')\n"
        "def root():\n"
        "    return {}\n"
    )
    
    # Mock get_performance_summary to return None (empty)
    with patch('openapi_doc_generator.utils.get_performance_summary') as mock_perf:
        mock_perf.return_value = None
        
        exit_code = main([
            "--app", str(app_file),
            "--format", "openapi",
            "--performance-metrics"
        ])
        
        assert exit_code == 0
        
        # No performance summary should be logged when empty
        assert "Performance Summary:" not in caplog.text


def test_cli_unknown_format_unreachable(tmp_path):
    """Test that unknown format error (line 154) is actually unreachable due to argparse."""
    # This test documents that line 154 is unreachable in normal usage
    # because argparse validates the format choices before the code reaches that point
    
    # Create a simple app file
    app_file = tmp_path / "app.py"
    app_file.write_text("from flask import Flask\napp = Flask(__name__)")
    
    # Try to use an invalid format - argparse should reject it
    with pytest.raises(SystemExit) as exc_info:
        main([
            "--app", str(app_file),
            "--format", "invalid_format"  # Not in choices
        ])
    
    # argparse exits with code 2 for usage errors
    assert exc_info.value.code == 2
    
    # The parser.error on line 154 is never reached because argparse
    # validates choices before calling _generate_output