"""Tests for performance metrics collection and reporting."""

import json
import time
import unittest.mock
from pathlib import Path
import pytest

from openapi_doc_generator.discovery import RouteDiscoverer
from openapi_doc_generator.utils import (
    setup_json_logging,
    reset_correlation_id,
)


class TestPerformanceMetrics:
    """Test suite for performance metrics functionality."""

    def setup_method(self):
        """Reset state before each test."""
        from openapi_doc_generator.utils import set_performance_tracking, clear_performance_stats
        reset_correlation_id()
        clear_performance_stats()
        set_performance_tracking(True)  # Ensure performance tracking is enabled for tests

    def test_performance_decorator_measures_execution_time(self):
        """Test that performance decorator accurately measures execution time."""
        from openapi_doc_generator.utils import measure_performance

        @measure_performance("test_function")
        def slow_function():
            time.sleep(0.1)  # 100ms delay
            return "result"

        with unittest.mock.patch("logging.getLogger") as mock_logger:
            mock_logger_instance = unittest.mock.MagicMock()
            mock_logger.return_value = mock_logger_instance

            result = slow_function()

            assert result == "result"
            mock_logger_instance.info.assert_called_once()
            call_args = mock_logger_instance.info.call_args

            # Check that timing information was logged
            assert "extra" in call_args.kwargs
            extra = call_args.kwargs["extra"]
            assert "duration_ms" in extra
            duration = extra["duration_ms"]
            assert 95 <= duration <= 150  # Allow some variance for test timing

    def test_performance_decorator_tracks_memory_usage(self):
        """Test that performance decorator tracks memory usage."""
        from openapi_doc_generator.utils import measure_performance

        @measure_performance("memory_test")
        def memory_allocating_function():
            # Allocate some memory
            large_list = [i for i in range(10000)]
            return len(large_list)

        with unittest.mock.patch("logging.getLogger") as mock_logger:
            mock_logger_instance = unittest.mock.MagicMock()
            mock_logger.return_value = mock_logger_instance

            result = memory_allocating_function()

            assert result == 10000
            mock_logger_instance.info.assert_called_once()
            call_args = mock_logger_instance.info.call_args

            # Check that memory information was logged
            assert "extra" in call_args.kwargs
            extra = call_args.kwargs["extra"]
            assert "memory_peak_mb" in extra
            assert extra["memory_peak_mb"] > 0

    def test_performance_decorator_handles_exceptions(self):
        """Test that performance decorator properly handles exceptions."""
        from openapi_doc_generator.utils import measure_performance

        @measure_performance("exception_test")
        def failing_function():
            raise ValueError("Test exception")

        with unittest.mock.patch("logging.getLogger") as mock_logger:
            mock_logger_instance = unittest.mock.MagicMock()
            mock_logger.return_value = mock_logger_instance

            with pytest.raises(ValueError, match="Test exception"):
                failing_function()

            # Should still log performance data even when exception occurs
            mock_logger_instance.info.assert_called_once()
            call_args = mock_logger_instance.info.call_args
            assert "extra" in call_args.kwargs
            extra = call_args.kwargs["extra"]
            assert "duration_ms" in extra

    def test_route_discovery_performance_tracking(self):
        """Test that route discovery tracks performance metrics."""
        # Create a temporary FastAPI file
        import tempfile

        fastapi_code = '''
from fastapi import FastAPI

app = FastAPI()

@app.get("/users")
def get_users():
    """Get all users."""
    return []

@app.post("/users")
def create_user():
    """Create a new user."""
    return {}
'''

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp_file:
            tmp_file.write(fastapi_code)
            tmp_file.flush()

            try:
                with unittest.mock.patch("logging.getLogger") as mock_logger:
                    mock_logger_instance = unittest.mock.MagicMock()
                    mock_logger.return_value = mock_logger_instance

                    discoverer = RouteDiscoverer(tmp_file.name)
                    routes = discoverer.discover()

                    assert len(routes) == 2
                    assert routes[0].path == "/users"
                    assert routes[0].methods == ["GET"]

                    # Check that performance metrics were logged
                    performance_calls = [
                        call
                        for call in mock_logger_instance.info.call_args_list
                        if call.kwargs.get("extra", {}).get("duration_ms") is not None
                    ]
                    assert len(performance_calls) > 0

                    # Verify metrics structure
                    perf_call = performance_calls[0]
                    extra = perf_call.kwargs["extra"]
                    assert "duration_ms" in extra
                    assert "operation" in extra

                    # Check for route count in any info call
                    route_calls = [
                        call
                        for call in mock_logger_instance.info.call_args_list
                        if call.kwargs.get("extra", {}).get("route_count") is not None
                    ]
                    if route_calls:
                        route_extra = route_calls[0].kwargs["extra"]
                        assert route_extra["route_count"] == 2

            finally:
                Path(tmp_file.name).unlink()

    def test_ast_cache_performance_metrics(self):
        """Test that AST caching performance is tracked."""
        from openapi_doc_generator.utils import get_cached_ast, _clear_ast_cache

        # Clear cache to start fresh
        _clear_ast_cache()

        source_code = """
def hello():
    return "world"
"""

        with unittest.mock.patch("logging.getLogger") as mock_logger:
            mock_logger_instance = unittest.mock.MagicMock()
            mock_logger.return_value = mock_logger_instance

            # First call - should be a cache miss
            ast1 = get_cached_ast(source_code, "test.py")
            assert ast1 is not None

            # Second call - should be a cache hit
            ast2 = get_cached_ast(source_code, "test.py")
            assert ast2 is ast1  # Same object due to caching

            # Verify cache metrics were logged
            cache_calls = [
                call
                for call in mock_logger_instance.debug.call_args_list
                if "cache" in str(call).lower()
            ]
            assert len(cache_calls) >= 1

    def test_performance_metrics_json_format(self):
        """Test that performance metrics are properly formatted in JSON logs."""
        setup_json_logging()

        from openapi_doc_generator.utils import measure_performance

        @measure_performance("json_test")
        def test_function():
            return "test"

        import io
        import sys

        # Capture stderr to check JSON output
        captured_stderr = io.StringIO()
        with unittest.mock.patch.object(sys, "stderr", captured_stderr):
            test_function()

        # Parse the JSON output
        stderr_output = captured_stderr.getvalue().strip()
        if stderr_output:
            log_lines = stderr_output.split("\n")
            json_logs = []
            for line in log_lines:
                if line.strip():
                    try:
                        json_logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

            # Find performance-related log entry
            perf_logs = [
                log for log in json_logs if "duration_ms" in log or "operation" in log
            ]
            assert len(perf_logs) > 0

            perf_log = perf_logs[0]
            assert "timestamp" in perf_log
            assert "correlation_id" in perf_log
            assert "duration_ms" in perf_log or "execution_time_ms" in perf_log

    def test_framework_detection_performance_tracking(self):
        """Test that framework detection performance is measured."""
        import tempfile

        # Test with multiple frameworks to ensure detection performance is tracked
        test_cases = [
            ("fastapi", "from fastapi import FastAPI\napp = FastAPI()"),
            ("flask", "from flask import Flask\napp = Flask(__name__)"),
            ("django", "from django.urls import path"),
            ("express", "const express = require('express')"),
        ]

        with unittest.mock.patch("logging.getLogger") as mock_logger:
            mock_logger_instance = unittest.mock.MagicMock()
            mock_logger.return_value = mock_logger_instance

            for framework, code in test_cases:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".py", delete=False
                ) as tmp_file:
                    tmp_file.write(code)
                    tmp_file.flush()

                    try:
                        discoverer = RouteDiscoverer(tmp_file.name)
                        # Framework detection happens during initialization and discovery
                        discoverer.discover()

                    except ValueError:
                        # Some frameworks may not have discoverable routes in minimal examples
                        pass
                    finally:
                        Path(tmp_file.name).unlink()

            # Verify that framework detection performance was logged
            detection_calls = [
                call
                for call in mock_logger_instance.info.call_args_list
                if call.kwargs.get("extra", {}).get("operation")
                == "framework_detection"
            ]
            assert len(detection_calls) > 0

    def test_performance_metrics_aggregation(self):
        """Test that performance metrics can be aggregated across operations."""
        from openapi_doc_generator.utils import (
            measure_performance,
            get_performance_summary,
        )

        @measure_performance("operation_a")
        def operation_a():
            time.sleep(0.05)  # 50ms
            return "a"

        @measure_performance("operation_b")
        def operation_b():
            time.sleep(0.03)  # 30ms
            return "b"

        # Execute operations multiple times
        for _ in range(3):
            operation_a()
            operation_b()

        # Get performance summary
        summary = get_performance_summary()

        assert "operation_a" in summary
        assert "operation_b" in summary

        # Check aggregated metrics
        op_a_stats = summary["operation_a"]
        assert op_a_stats["count"] == 3
        assert op_a_stats["total_duration_ms"] >= 150  # 3 * 50ms
        assert op_a_stats["avg_duration_ms"] >= 50

        op_b_stats = summary["operation_b"]
        assert op_b_stats["count"] == 3
        assert op_b_stats["total_duration_ms"] >= 90  # 3 * 30ms
        assert op_b_stats["avg_duration_ms"] >= 30

    def test_performance_metrics_disabled_by_default(self):
        """Test that performance metrics collection can be disabled."""
        from openapi_doc_generator.utils import (
            measure_performance,
            set_performance_tracking,
        )

        # Disable performance tracking
        set_performance_tracking(False)

        @measure_performance("disabled_test")
        def test_function():
            return "test"

        with unittest.mock.patch("logging.getLogger") as mock_logger:
            mock_logger_instance = unittest.mock.MagicMock()
            mock_logger.return_value = mock_logger_instance

            result = test_function()

            assert result == "test"
            # No performance logs should be emitted when disabled
            perf_calls = [
                call
                for call in mock_logger_instance.info.call_args_list
                if call.kwargs.get("extra", {}).get("duration_ms") is not None
            ]
            assert len(perf_calls) == 0

        # Re-enable for other tests
        set_performance_tracking(True)

    def test_memory_tracking_accuracy(self):
        """Test that memory tracking provides reasonable measurements."""
        from openapi_doc_generator.utils import measure_performance
        import tracemalloc

        # Ensure tracemalloc is available
        if not hasattr(tracemalloc, "start"):
            pytest.skip("tracemalloc not available")

        @measure_performance("memory_accuracy_test")
        def allocate_known_memory():
            # Allocate approximately 1MB of memory
            data = bytearray(1024 * 1024)  # 1MB
            return len(data)

        with unittest.mock.patch("logging.getLogger") as mock_logger:
            mock_logger_instance = unittest.mock.MagicMock()
            mock_logger.return_value = mock_logger_instance

            result = allocate_known_memory()

            assert result == 1024 * 1024
            mock_logger_instance.info.assert_called_once()
            call_args = mock_logger_instance.info.call_args

            # Memory usage should be reported and be reasonable
            extra = call_args.kwargs.get("extra", {})
            if "memory_peak_mb" in extra:
                memory_mb = extra["memory_peak_mb"]
                assert 0.5 <= memory_mb <= 10.0  # Reasonable range, allowing for test overhead

    def test_correlation_id_consistency_in_metrics(self):
        """Test that all metrics within a session share the same correlation ID."""
        from openapi_doc_generator.utils import measure_performance

        correlation_ids = []

        @measure_performance("correlation_test_1")
        def func1():
            return "1"

        @measure_performance("correlation_test_2")
        def func2():
            return "2"

        with unittest.mock.patch("logging.getLogger") as mock_logger:
            mock_logger_instance = unittest.mock.MagicMock()
            mock_logger.return_value = mock_logger_instance

            func1()
            func2()

            # Extract correlation IDs from all calls
            for call in mock_logger_instance.info.call_args_list:
                extra = call.kwargs.get("extra", {})
                if "correlation_id" in extra:
                    correlation_ids.append(extra["correlation_id"])

            # All correlation IDs should be the same within a session
            assert len(set(correlation_ids)) == 1
            assert len(correlation_ids) >= 2
