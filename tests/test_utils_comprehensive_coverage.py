"""Comprehensive tests to achieve 100% coverage for utils module."""

import ast
import hashlib
import logging
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock
from openapi_doc_generator.utils import (
    get_cached_ast, 
    setup_json_logging,
    _parse_ast_cached,
    reset_correlation_id,
    get_correlation_id
)


class TestASTCachingCoverage:
    """Test AST caching functionality for complete coverage."""
    
    def test_get_cached_ast_cache_hit(self):
        """Test AST caching with cache hit scenario (lines 52-70)."""
        source_code = "print('hello world')"
        filename = "test.py"
        
        # Clear any existing cache state
        _parse_ast_cached.cache_clear()
        
        with patch('openapi_doc_generator.utils.logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            # First call should be a cache miss
            ast1 = get_cached_ast(source_code, filename)
            assert isinstance(ast1, ast.Module)
            
            # Second call should be a cache hit and trigger the debug logging
            ast2 = get_cached_ast(source_code, filename)
            assert isinstance(ast2, ast.Module)
            
            # Verify cache hit logging was called
            mock_logger.debug.assert_called()
            call_args = mock_logger.debug.call_args
            assert "AST cache hit" in call_args[0][0]
            assert call_args[1]['extra']['cache_hit'] is True
    
    def test_get_cached_ast_cache_miss(self):
        """Test AST caching with cache miss scenario (lines 71-85)."""
        source_code = "print('hello world')"
        filename = "test.py"
        
        # Clear cache to ensure miss
        _parse_ast_cached.cache_clear()
        
        with patch('openapi_doc_generator.utils.logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            # First call should be a cache miss
            ast_result = get_cached_ast(source_code, filename)
            assert isinstance(ast_result, ast.Module)
            
            # Verify cache miss logging was called
            mock_logger.debug.assert_called()
            call_args = mock_logger.debug.call_args
            assert "AST cache miss" in call_args[0][0]
            assert call_args[1]['extra']['cache_hit'] is False
    
    def test_get_cached_ast_source_hashing(self):
        """Test that different source code creates different cache keys (lines 48-49)."""
        source1 = "print('hello')"
        source2 = "print('world')"
        filename = "test.py"
        
        # Clear cache
        _parse_ast_cached.cache_clear()
        
        # These should create different cache entries
        ast1 = get_cached_ast(source1, filename)
        ast2 = get_cached_ast(source2, filename)
        
        # Both should be valid AST modules
        assert isinstance(ast1, ast.Module)
        assert isinstance(ast2, ast.Module)
        
        # Verify cache has 2 entries (different source hashes)
        cache_info = _parse_ast_cached.cache_info()
        assert cache_info.currsize >= 2
    
    def test_parse_ast_cached_syntax_error(self):
        """Test _parse_ast_cached with invalid syntax."""
        source_hash = "test_hash"
        invalid_source = "print('unclosed string"
        filename = "test.py"
        
        # This should raise SyntaxError
        with pytest.raises(SyntaxError):
            _parse_ast_cached(source_hash, invalid_source, filename)


class TestStructuredLoggingCoverage:
    """Test structured logging setup for complete coverage."""
    
    def test_setup_json_logging_basic(self):
        """Test basic structured logging setup (lines 146-160)."""
        # Reset correlation ID and execution time to trigger new values
        reset_correlation_id()
        
        with patch('openapi_doc_generator.utils.uuid.uuid4') as mock_uuid:
            mock_uuid.return_value = Mock()
            mock_uuid.return_value.__str__ = Mock(return_value='test-correlation-id-12345')
            
            with patch('openapi_doc_generator.utils.time.time') as mock_time:
                mock_time.return_value = 1234567890.123
                
                with patch('openapi_doc_generator.utils.logging') as mock_logging:
                    mock_root_logger = Mock()
                    mock_root_logger.handlers = []
                    mock_logging.getLogger.return_value = mock_root_logger
                    mock_logging.StreamHandler.return_value = Mock()
                    
                    # This should trigger lines 146-160
                    logger = setup_json_logging()
                    
                    # Verify that logging was configured
                    mock_logging.basicConfig.assert_called_once()
                    mock_logging.StreamHandler.assert_called_once()
    
    def test_setup_json_logging_with_existing_handlers(self):
        """Test structured logging setup with existing handlers (lines 149-152)."""
        reset_correlation_id()
        
        with patch('openapi_doc_generator.utils.logging') as mock_logging:
            # Create mock handlers to be removed
            mock_handler1 = Mock()
            mock_handler2 = Mock()
            mock_root_logger = Mock()
            mock_root_logger.handlers = [mock_handler1, mock_handler2]
            
            mock_logging.getLogger.return_value = mock_root_logger
            mock_logging.StreamHandler.return_value = Mock()
            
            # This should trigger handler removal logic
            logger = setup_json_logging()
            
            # Verify handlers were removed
            assert mock_root_logger.removeHandler.call_count == 2
            mock_root_logger.removeHandler.assert_any_call(mock_handler1)
            mock_root_logger.removeHandler.assert_any_call(mock_handler2)
    
    def test_setup_json_logging_custom_level(self):
        """Test structured logging setup with custom log level."""
        reset_correlation_id()
        
        with patch('openapi_doc_generator.utils.logging') as mock_logging:
            mock_root_logger = Mock()
            mock_root_logger.handlers = []
            mock_logging.getLogger.return_value = mock_root_logger
            mock_logging.StreamHandler.return_value = Mock()
            
            # Test with custom level
            logger = setup_json_logging(level=logging.ERROR)
            
            # Verify basicConfig was called with custom level
            mock_logging.basicConfig.assert_called_once()
            call_kwargs = mock_logging.basicConfig.call_args[1]
            assert call_kwargs['level'] == logging.ERROR
    
    def test_correlation_id_and_execution_time_reset(self):
        """Test that correlation ID and execution time are reset (lines 146-147)."""
        reset_correlation_id()
        
        with patch('openapi_doc_generator.utils.uuid.uuid4') as mock_uuid, \
             patch('openapi_doc_generator.utils.time.time') as mock_time:
            
            mock_uuid.return_value = Mock()
            mock_uuid.return_value.__str__ = Mock(return_value='new-correlation-id')
            mock_time.return_value = 9876543210.456
            
            with patch('openapi_doc_generator.utils.logging') as mock_logging:
                mock_root_logger = Mock()
                mock_root_logger.handlers = []
                mock_logging.getLogger.return_value = mock_root_logger
                
                # This should trigger correlation ID and time reset
                setup_json_logging()
                
                # Verify new correlation ID was generated
                mock_uuid.assert_called_once()
                mock_time.assert_called_once()
                
                # Verify the new correlation ID is accessible
                correlation_id = get_correlation_id()
                assert correlation_id == 'new-corr'  # First 8 chars