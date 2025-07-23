"""Tests for achieving 100% coverage of discovery module abstract methods and edge cases."""

import ast
import os
import tempfile
import pytest
from openapi_doc_generator.discovery import RoutePlugin, RouteDiscoverer


class TestAbstractMethodCoverage:
    """Test coverage for abstract method instantiation attempts."""

    def test_route_plugin_cannot_be_instantiated_directly(self):
        """Test that RoutePlugin abstract class cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class RoutePlugin"):
            RoutePlugin()

    def test_abstract_detect_method_coverage(self):
        """Ensure detect method is properly abstract."""
        # This test covers the abstract method definition at line 33
        assert hasattr(RoutePlugin, 'detect')
        assert getattr(RoutePlugin.detect, '__isabstractmethod__', False)

    def test_abstract_discover_method_coverage(self):
        """Ensure discover method is properly abstract."""
        # This test covers the abstract method definition at line 38
        assert hasattr(RoutePlugin, 'discover')
        assert getattr(RoutePlugin.discover, '__isabstractmethod__', False)


class TestDjangoViewNameExtractionEdgeCases:
    """Test edge cases in Django view name extraction."""

    def test_extract_django_view_name_with_ast_name(self):
        """Test Django view name extraction when target is ast.Name (line 336)."""
        # Create a temporary test file for RouteDiscoverer
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("# Test file")
            temp_path = f.name
        
        try:
            discoverer = RouteDiscoverer(temp_path)
            
            # Create AST nodes that represent: path('test/', view_function)
            # where view_function is an ast.Name node
            view_name_node = ast.Name(id='test_view_function', ctx=ast.Load())
            args = [
                ast.Constant(value='test/'),  # Path pattern
                view_name_node,  # View function as ast.Name
            ]
            
            # Verify that the second arg is indeed an ast.Name
            assert isinstance(args[1], ast.Name)
            assert len(args) > 1
            
            result = discoverer._extract_django_view_name(args)
            
            # This should hit line 336: return target.id
            assert result == 'test_view_function'
        finally:
            os.unlink(temp_path)

    def test_extract_django_view_name_with_ast_attribute(self):
        """Test Django view name extraction when target is ast.Attribute."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("# Test file")
            temp_path = f.name
        
        try:
            discoverer = RouteDiscoverer(temp_path)
            
            # Create AST nodes that represent: path('test/', views.my_view)
            attr_node = ast.Attribute(
                value=ast.Name(id='views', ctx=ast.Load()),
                attr='my_view',
                ctx=ast.Load()
            )
            args = [
                ast.Constant(value='test/'),  # Path pattern
                attr_node,  # View function as ast.Attribute
            ]
            
            result = discoverer._extract_django_view_name(args)
            
            # This should hit line 334: return target.attr
            assert result == 'my_view'
        finally:
            os.unlink(temp_path)

    def test_extract_django_view_name_insufficient_args(self):
        """Test Django view name extraction with insufficient arguments."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("# Test file")
            temp_path = f.name
        
        try:
            discoverer = RouteDiscoverer(temp_path)
            
            # Only one argument (path pattern only)
            args = [ast.Constant(value='test/')]
            
            result = discoverer._extract_django_view_name(args)
            
            # Should return empty string when insufficient args
            assert result == ""
        finally:
            os.unlink(temp_path)

    def test_extract_django_view_name_unknown_target_type(self):
        """Test Django view name extraction with unknown target type."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("# Test file")
            temp_path = f.name
        
        try:
            discoverer = RouteDiscoverer(temp_path)
            
            # Create args with unknown target type (e.g., ast.Constant)
            args = [
                ast.Constant(value='test/'),  # Path pattern
                ast.Constant(value='not_a_view'),  # Invalid target type
            ]
            
            result = discoverer._extract_django_view_name(args)
            
            # Should return empty string for unknown types
            assert result == ""
        finally:
            os.unlink(temp_path)