"""Tests for refactored plugin loading methods in discovery.py."""

import logging
from unittest.mock import Mock, patch
from openapi_doc_generator.discovery import get_plugins, _load_single_plugin, register_plugin, _PLUGINS


class TestPluginLoadingRefactoring:
    """Test the refactored plugin loading functionality."""
    
    def setup_method(self):
        """Clear plugins before each test."""
        self._original_plugins = _PLUGINS.copy()
        _PLUGINS.clear()
    
    def teardown_method(self):
        """Clear plugins after each test to ensure test isolation."""
        _PLUGINS.clear()
    
    def test_load_single_plugin_success(self):
        """Test successful plugin loading."""
        # Create a mock entry point and plugin class
        mock_ep = Mock()
        mock_ep.name = "test_plugin"
        
        mock_plugin_cls = Mock()
        mock_plugin_instance = Mock()
        mock_plugin_cls.return_value = mock_plugin_instance
        mock_ep.load.return_value = mock_plugin_cls
        
        # Test successful loading
        _load_single_plugin(mock_ep)
        
        # Verify plugin was loaded and registered
        mock_ep.load.assert_called_once()
        mock_plugin_cls.assert_called_once()
        assert mock_plugin_instance in _PLUGINS
    
    def test_load_single_plugin_import_error(self):
        """Test plugin loading with ImportError."""
        mock_ep = Mock()
        mock_ep.name = "failing_plugin"
        mock_ep.load.side_effect = ImportError("Module not found")
        
        with patch('openapi_doc_generator.discovery.logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            # This should handle the ImportError gracefully
            _load_single_plugin(mock_ep)
            
            # Verify error was logged
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args[0]
            assert "Failed to import plugin" in call_args[0]
            assert "failing_plugin" in call_args[1]
    
    def test_load_single_plugin_module_not_found_error(self):
        """Test plugin loading with ModuleNotFoundError."""
        mock_ep = Mock()
        mock_ep.name = "missing_module_plugin"
        mock_ep.load.side_effect = ModuleNotFoundError("No module named 'missing'")
        
        with patch('openapi_doc_generator.discovery.logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            _load_single_plugin(mock_ep)
            
            # Verify import error was logged
            mock_logger.warning.assert_called_once()
            assert "Failed to import plugin" in mock_logger.warning.call_args[0][0]
    
    def test_load_single_plugin_attribute_error(self):
        """Test plugin loading with AttributeError."""
        mock_ep = Mock()
        mock_ep.name = "attribute_error_plugin"
        mock_ep.load.side_effect = AttributeError("'module' object has no attribute 'Plugin'")
        
        with patch('openapi_doc_generator.discovery.logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            _load_single_plugin(mock_ep)
            
            # Verify import error was logged
            mock_logger.warning.assert_called_once()
            assert "Failed to import plugin" in mock_logger.warning.call_args[0][0]
    
    def test_load_single_plugin_type_error(self):
        """Test plugin loading with TypeError during instantiation."""
        mock_ep = Mock()
        mock_ep.name = "type_error_plugin"
        
        mock_plugin_cls = Mock()
        mock_plugin_cls.side_effect = TypeError("Plugin() missing required argument")
        mock_ep.load.return_value = mock_plugin_cls
        
        with patch('openapi_doc_generator.discovery.logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            _load_single_plugin(mock_ep)
            
            # Verify instantiation error was logged
            mock_logger.warning.assert_called_once()
            assert "Failed to instantiate plugin" in mock_logger.warning.call_args[0][0]
    
    def test_load_single_plugin_value_error(self):
        """Test plugin loading with ValueError during instantiation."""
        mock_ep = Mock()
        mock_ep.name = "value_error_plugin"
        
        mock_plugin_cls = Mock()
        mock_plugin_cls.side_effect = ValueError("Invalid plugin configuration")
        mock_ep.load.return_value = mock_plugin_cls
        
        with patch('openapi_doc_generator.discovery.logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            _load_single_plugin(mock_ep)
            
            # Verify instantiation error was logged
            mock_logger.warning.assert_called_once()
            assert "Failed to instantiate plugin" in mock_logger.warning.call_args[0][0]
    
    def test_load_single_plugin_unexpected_error(self):
        """Test plugin loading with unexpected exception."""
        mock_ep = Mock()
        mock_ep.name = "unexpected_error_plugin"
        mock_ep.load.side_effect = RuntimeError("Unexpected error")
        
        with patch('openapi_doc_generator.discovery.logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            _load_single_plugin(mock_ep)
            
            # Verify unexpected error was logged with exception details
            mock_logger.exception.assert_called_once()
            assert "Unexpected plugin loading error" in mock_logger.exception.call_args[0][0]
    
    def test_get_plugins_calls_load_single_plugin(self):
        """Test that get_plugins calls _load_single_plugin for each entry point."""
        # Clear plugins to trigger loading
        _PLUGINS.clear()
        
        mock_ep1 = Mock()
        mock_ep1.name = "plugin1"
        mock_ep2 = Mock()
        mock_ep2.name = "plugin2"
        
        with patch('openapi_doc_generator.discovery.metadata.entry_points') as mock_entry_points, \
             patch('openapi_doc_generator.discovery._load_single_plugin') as mock_load_single, \
             patch('importlib.import_module'):
            
            mock_entry_points.return_value = [mock_ep1, mock_ep2]
            
            result_plugins = get_plugins()
            
            # Verify _load_single_plugin was called for each entry point
            assert result_plugins is not None  # Ensure plugins were loaded
            assert mock_load_single.call_count == 2
            mock_load_single.assert_any_call(mock_ep1)
            mock_load_single.assert_any_call(mock_ep2)
    
    def test_get_plugins_returns_existing_plugins(self):
        """Test that get_plugins returns existing plugins without reloading."""
        # Add some plugins manually
        mock_plugin1 = Mock()
        mock_plugin2 = Mock()
        register_plugin(mock_plugin1)
        register_plugin(mock_plugin2)
        
        with patch('openapi_doc_generator.discovery.metadata.entry_points') as mock_entry_points:
            # This should not be called since plugins already exist
            plugins = get_plugins()
            
            # Verify existing plugins are returned
            assert len(plugins) == 2
            assert mock_plugin1 in plugins
            assert mock_plugin2 in plugins
            
            # Verify entry points were not accessed
            mock_entry_points.assert_not_called()
    
    def test_refactored_plugin_loading_maintains_original_behavior(self, caplog):
        """Test that refactored plugin loading maintains the original behavior."""
        # Clear plugins to trigger loading
        _PLUGINS.clear()
        
        # Create mock entry points that simulate real plugin loading scenarios
        mock_successful_ep = Mock()
        mock_successful_ep.name = "successful_plugin"
        mock_plugin_cls = Mock()
        mock_plugin_instance = Mock()
        mock_plugin_cls.return_value = mock_plugin_instance
        mock_successful_ep.load.return_value = mock_plugin_cls
        
        mock_failing_ep = Mock()
        mock_failing_ep.name = "failing_plugin"
        mock_failing_ep.load.side_effect = ImportError("Plugin not found")
        
        with patch('openapi_doc_generator.discovery.metadata.entry_points') as mock_entry_points, \
             patch('importlib.import_module'):
            
            mock_entry_points.return_value = [mock_successful_ep, mock_failing_ep]
            
            # Capture log messages
            with caplog.at_level(logging.WARNING):
                plugins = get_plugins()
            
            # Verify successful plugin was loaded
            assert len(plugins) == 1
            assert mock_plugin_instance in plugins
            
            # Verify failed plugin was logged but didn't stop the process
            assert len(caplog.records) >= 1
            assert "Failed to import plugin" in caplog.text
            assert "failing_plugin" in caplog.text