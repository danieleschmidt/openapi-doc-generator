"""Final tests to achieve 100% coverage for remaining utils lines."""

import logging
import tracemalloc
from unittest.mock import Mock, patch, MagicMock
import pytest

from openapi_doc_generator.utils import (
    _clear_ast_cache,
    get_performance_summary,
    clear_performance_stats,
    set_performance_tracking,
    measure_performance,
    JSONFormatter,
    get_cached_ast
)


class TestRemainingUtilsCoverage:
    """Test remaining uncovered lines in utils module."""
    
    def test_clear_ast_cache_function(self):
        """Test _clear_ast_cache function (line 91)."""
        # First, populate the cache
        source_code = "print('test')"
        filename = "test.py"
        get_cached_ast(source_code, filename)
        
        # Now clear the cache - this should trigger line 91
        _clear_ast_cache()
        
        # Verify cache was cleared by checking cache stats
        from openapi_doc_generator.utils import _parse_ast_cached
        cache_info = _parse_ast_cached.cache_info()
        assert cache_info.currsize == 0
    
    def test_json_formatter_with_duration_ms(self):
        """Test JSON formatter with duration_ms attribute (line 129)."""
        formatter = JSONFormatter()
        
        # Create log record with duration_ms attribute
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        # Add duration_ms attribute to trigger line 129
        record.duration_ms = 1234.56
        
        formatted = formatter.format(record)
        
        # Verify duration_ms is included
        import json
        log_data = json.loads(formatted)
        assert 'duration_ms' in log_data
        assert log_data['duration_ms'] == 1234.56
    
    def test_get_performance_summary(self):
        """Test get_performance_summary function (line 193)."""
        # Clear any existing stats
        clear_performance_stats()
        
        # Enable performance tracking and run a test function
        set_performance_tracking(True)
        
        @measure_performance("test_operation")
        def test_function():
            return "result"
        
        # Execute function to generate stats
        result = test_function()
        assert result == "result"
        
        # Now get performance summary - this should trigger line 193
        summary = get_performance_summary()
        
        # Verify we got a dict with performance data
        assert isinstance(summary, dict)
        assert "test_operation" in summary
        assert "count" in summary["test_operation"]
        assert "total_duration_ms" in summary["test_operation"]
    
    def test_clear_performance_stats(self):
        """Test clear_performance_stats function (line 199)."""
        # First, generate some performance stats
        set_performance_tracking(True)
        
        @measure_performance("test_operation")
        def test_function():
            return "result"
        
        test_function()
        
        # Verify we have stats
        summary_before = get_performance_summary()
        assert len(summary_before) > 0
        
        # Clear performance stats - this should trigger line 199
        clear_performance_stats()
        
        # Verify stats are cleared
        summary_after = get_performance_summary()
        assert len(summary_after) == 0
    
    def test_performance_tracking_disabled_path(self):
        """Test measure_performance when tracking is disabled (line 216)."""
        # Disable performance tracking
        set_performance_tracking(False)
        
        @measure_performance("test_operation")
        def test_function():
            return "result"
        
        # This should trigger line 216 (early return when tracking disabled)
        result = test_function()
        assert result == "result"
        
        # Verify no performance stats were recorded
        summary = get_performance_summary()
        assert "test_operation" not in summary
    
    @patch('openapi_doc_generator.utils.tracemalloc')
    def test_tracemalloc_start_call(self, mock_tracemalloc):
        """Test tracemalloc.start() call (line 225)."""
        # Setup mock to simulate tracemalloc not being traced initially
        mock_tracemalloc.is_tracing.return_value = False
        mock_tracemalloc.get_traced_memory.return_value = (1000, 2000)
        
        # Enable performance tracking
        set_performance_tracking(True)
        
        @measure_performance("test_operation")
        def test_function():
            return "result"
        
        # This should trigger line 225: tracemalloc.start()
        result = test_function()
        assert result == "result"
        
        # Verify tracemalloc.start() was called
        mock_tracemalloc.start.assert_called_once()
    
    @patch('openapi_doc_generator.utils.tracemalloc')
    def test_memory_peak_calculation(self, mock_tracemalloc):
        """Test memory peak calculation (line 245)."""
        # Setup mock for successful memory tracking
        mock_tracemalloc.is_tracing.return_value = True
        mock_tracemalloc.get_traced_memory.side_effect = [
            (1000, 1000),  # Initial memory
            (2000, 3000),  # Final memory (2000 current, 3000 peak)
        ]
        
        # Enable performance tracking
        set_performance_tracking(True)
        
        @measure_performance("test_operation")
        def test_function():
            return "result"
        
        # This should trigger line 245: memory_peak calculation
        result = test_function()
        assert result == "result"
        
        # Verify memory calculations were performed
        assert mock_tracemalloc.get_traced_memory.call_count == 2
    
    @patch('openapi_doc_generator.utils.tracemalloc')
    @patch('openapi_doc_generator.utils.logging')
    def test_memory_peak_logging(self, mock_logging, mock_tracemalloc):
        """Test memory peak logging (line 278)."""
        # Setup mocks
        mock_logger = Mock()
        mock_logging.getLogger.return_value = mock_logger
        
        mock_tracemalloc.is_tracing.return_value = True
        mock_tracemalloc.get_traced_memory.side_effect = [
            (1000, 1000),  # Initial memory
            (2000, 3000),  # Final memory
        ]
        
        # Enable performance tracking
        set_performance_tracking(True)
        
        @measure_performance("test_operation")
        def test_function():
            return "result"
        
        # This should trigger line 278: memory_peak logging
        result = test_function()
        assert result == "result"
        
        # Verify logging was called with memory peak information
        mock_logger.info.assert_called()
        call_kwargs = mock_logger.info.call_args[1]
        assert 'extra' in call_kwargs
        assert 'memory_peak_mb' in call_kwargs['extra']