"""Tests for achieving 100% coverage of utils module error paths and edge cases."""

import logging
from unittest.mock import Mock, patch

from openapi_doc_generator.utils import (
    get_correlation_id,
    reset_correlation_id,
    JSONFormatter,
    measure_performance,
)


class TestCorrelationIdGeneration:
    """Test coverage for correlation ID generation edge cases."""

    def test_correlation_id_generation_when_none(self):
        """Test correlation ID generation when _correlation_id is None (line 107)."""
        # Reset correlation ID to None to trigger generation
        reset_correlation_id()
        
        # Mock uuid.uuid4 to ensure we can test the generation path
        with patch('openapi_doc_generator.utils.uuid.uuid4') as mock_uuid:
            mock_uuid.return_value = Mock()
            mock_uuid.return_value.__str__ = Mock(return_value='12345678-1234-5678-9012-123456789012')
            
            # This should trigger line 107: _correlation_id = str(uuid.uuid4())[:8]
            correlation_id = get_correlation_id()
            
            # Verify that uuid.uuid4 was called and ID was generated
            mock_uuid.assert_called_once()
            assert correlation_id == '12345678'

    def test_correlation_id_reuse_when_already_set(self):
        """Test that correlation ID is reused when already set."""
        # Set a correlation ID first
        reset_correlation_id()
        first_id = get_correlation_id()
        
        # Getting it again should return the same ID
        second_id = get_correlation_id()
        assert first_id == second_id


class TestJSONFormatterEdgeCases:
    """Test JSON formatter edge cases for complete coverage."""

    def test_json_formatter_correlation_id_generation(self):
        """Test JSON formatter correlation ID generation when None (line 107)."""
        # Reset correlation ID to trigger generation
        from openapi_doc_generator.utils import reset_correlation_id
        reset_correlation_id()
        
        formatter = JSONFormatter()
        
        # Create a log record
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        with patch('openapi_doc_generator.utils.uuid.uuid4') as mock_uuid:
            mock_uuid.return_value = Mock()
            mock_uuid.return_value.__str__ = Mock(return_value='abcd1234-5678-9012-3456-789012345678')
            
            # This should trigger line 107: _correlation_id = str(uuid.uuid4())[:8]
            formatted = formatter.format(record)
            
            # Verify that uuid.uuid4 was called and correlation ID was set
            mock_uuid.assert_called_once()
            
            # Verify the correlation ID is in the output
            import json
            log_data = json.loads(formatted)
            assert log_data['correlation_id'] == 'abcd1234'

    def test_json_formatter_with_timing_attribute(self):
        """Test JSON formatter when record has timing attribute (line 132)."""
        formatter = JSONFormatter()
        
        # Create a log record with timing attribute
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        # Add timing attribute to trigger line 132
        record.timing = {'duration': 1.5, 'memory_peak': 2048}
        
        formatted = formatter.format(record)
        
        # Verify timing information is included
        import json
        log_data = json.loads(formatted)
        assert 'timing' in log_data
        assert log_data['timing']['duration'] == 1.5
        assert log_data['timing']['memory_peak'] == 2048

    def test_json_formatter_with_exception_info(self):
        """Test JSON formatter when record has exception info (line 136)."""
        formatter = JSONFormatter()
        
        # Create exception info properly
        try:
            raise ValueError("Test exception")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        
        # Create log record with exception info
        record = logging.LogRecord(
            name='test',
            level=logging.ERROR,
            pathname='',
            lineno=0,
            msg='Error occurred',
            args=(),
            exc_info=exc_info
        )
        
        formatted = formatter.format(record)
        
        # Verify exception information is included
        import json
        log_data = json.loads(formatted)
        assert 'exception' in log_data
        assert 'ValueError: Test exception' in log_data['exception']

    def test_json_formatter_without_timing_or_exception(self):
        """Test JSON formatter with normal record (no timing or exception)."""
        formatter = JSONFormatter()
        
        # Create a normal log record
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg='Normal message',
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        
        # Verify basic structure without timing or exception
        import json
        log_data = json.loads(formatted)
        assert 'message' in log_data
        assert 'level' in log_data
        assert 'timing' not in log_data
        assert 'exception' not in log_data


class TestMemoryTrackingErrorHandling:
    """Test memory tracking error handling edge cases."""

    @patch('openapi_doc_generator.utils.tracemalloc')
    def test_memory_tracking_value_error(self, mock_tracemalloc):
        """Test memory tracking with ValueError (lines 248-251)."""
        # Enable performance tracking first
        from openapi_doc_generator.utils import set_performance_tracking
        set_performance_tracking(True)
        
        # Mock tracemalloc to work initially, then raise ValueError on cleanup
        mock_tracemalloc.is_tracing.return_value = True
        mock_tracemalloc.get_traced_memory.side_effect = [
            (1000, 2000),  # Initial call succeeds  
            ValueError("Tracemalloc error")  # Second call (cleanup) fails
        ]
        
        @measure_performance("test_operation")
        def test_function():
            return "test"
        
        # This should trigger the ValueError exception handling on lines 248-251
        result = test_function()
        assert result == "test"

    @patch('openapi_doc_generator.utils.tracemalloc')
    def test_memory_tracking_attribute_error(self, mock_tracemalloc):
        """Test memory tracking with AttributeError (lines 248-251)."""
        # Enable performance tracking first
        from openapi_doc_generator.utils import set_performance_tracking
        set_performance_tracking(True)
        
        # Mock tracemalloc to work initially, then raise AttributeError on cleanup
        mock_tracemalloc.is_tracing.return_value = True
        mock_tracemalloc.get_traced_memory.side_effect = [
            (1000, 2000),  # Initial call succeeds
            AttributeError("Missing attribute")  # Second call (cleanup) fails
        ]
        
        @measure_performance("test_operation")
        def test_function():
            return "test"
        
        # This should trigger the AttributeError exception handling on lines 248-251
        result = test_function()
        assert result == "test"

    @patch('openapi_doc_generator.utils.tracemalloc')
    def test_memory_tracking_os_error(self, mock_tracemalloc):
        """Test memory tracking with OSError (lines 248-251)."""
        # Enable performance tracking first
        from openapi_doc_generator.utils import set_performance_tracking
        set_performance_tracking(True)
        
        # Mock tracemalloc to work initially, then raise OSError on cleanup
        mock_tracemalloc.is_tracing.return_value = True
        mock_tracemalloc.get_traced_memory.side_effect = [
            (1000, 2000),  # Initial call succeeds
            OSError("System error")  # Second call (cleanup) fails
        ]
        
        @measure_performance("test_operation")
        def test_function():
            return "test"
        
        # This should trigger the OSError exception handling on lines 248-251
        result = test_function()
        assert result == "test"

    @patch('openapi_doc_generator.utils.tracemalloc')
    @patch('openapi_doc_generator.utils.logging')
    def test_memory_tracking_error_logging(self, mock_logging, mock_tracemalloc):
        """Test that memory tracking errors are properly logged."""
        # Enable performance tracking first
        from openapi_doc_generator.utils import set_performance_tracking
        set_performance_tracking(True)
        
        # Setup mock logger
        mock_logger = Mock()
        mock_logging.getLogger.return_value = mock_logger
        
        # Mock tracemalloc to work initially, then raise ValueError on cleanup
        mock_tracemalloc.is_tracing.return_value = True
        error_instance = ValueError("Test error")
        mock_tracemalloc.get_traced_memory.side_effect = [
            (1000, 2000),  # Initial call succeeds
            error_instance  # Second call (cleanup) fails
        ]
        
        @measure_performance("test_operation")
        def test_function():
            return "test"
        
        # Execute function to trigger error handling
        result = test_function()
        
        # Verify that debug logging was called with the error
        mock_logger.debug.assert_called_with("Memory tracking failed: %s", error_instance)
        assert result == "test"