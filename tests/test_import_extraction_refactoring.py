"""Tests for refactored import extraction methods in discovery.py."""

import ast
import tempfile
import os
from openapi_doc_generator.discovery import RouteDiscoverer


class TestImportExtractionRefactoring:
    """Test the refactored import extraction methods."""
    
    def _create_temp_file(self, content: str) -> str:
        """Helper to create temporary file with content."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(content)
            return f.name
    
    def test_process_import_node_with_direct_import(self):
        """Test _process_import_node with ast.Import nodes."""
        temp_file = self._create_temp_file("# test")
        try:
            discoverer = RouteDiscoverer(temp_file)
            imports = set()
            
            # Create ast.Import node: import sys, os
            import_node = ast.Import(names=[
                ast.alias(name='sys', asname=None),
                ast.alias(name='os', asname=None)
            ])
            
            # Test the refactored method
            discoverer._process_import_node(import_node, imports)
            
            assert 'sys' in imports
            assert 'os' in imports
            assert len(imports) == 2
            
        finally:
            os.unlink(temp_file)
    
    def test_process_import_node_with_from_import(self):
        """Test _process_import_node with ast.ImportFrom nodes."""
        temp_file = self._create_temp_file("# test")
        try:
            discoverer = RouteDiscoverer(temp_file)
            imports = set()
            
            # Create ast.ImportFrom node: from flask import Flask
            from_import_node = ast.ImportFrom(
                module='flask',
                names=[ast.alias(name='Flask', asname=None)],
                level=0
            )
            
            # Test the refactored method
            discoverer._process_import_node(from_import_node, imports)
            
            assert 'flask' in imports
            assert len(imports) == 1
            
        finally:
            os.unlink(temp_file)
    
    def test_process_import_node_with_other_node_types(self):
        """Test _process_import_node with non-import nodes."""
        temp_file = self._create_temp_file("# test")
        try:
            discoverer = RouteDiscoverer(temp_file)
            imports = set()
            
            # Create a non-import node (e.g., function definition)
            func_node = ast.FunctionDef(
                name='test_func',
                args=ast.arguments(
                    posonlyargs=[], args=[], vararg=None, kwonlyargs=[],
                    kw_defaults=[], kwarg=None, defaults=[]
                ),
                body=[],
                decorator_list=[],
                returns=None
            )
            
            # Test that non-import nodes are ignored
            discoverer._process_import_node(func_node, imports)
            
            assert len(imports) == 0
            
        finally:
            os.unlink(temp_file)
    
    def test_handle_direct_import(self):
        """Test _handle_direct_import method directly."""
        temp_file = self._create_temp_file("# test")
        try:
            discoverer = RouteDiscoverer(temp_file)
            imports = set()
            
            # Create ast.Import node with multiple aliases
            import_node = ast.Import(names=[
                ast.alias(name='json', asname=None),
                ast.alias(name='collections', asname='col')
            ])
            
            discoverer._handle_direct_import(import_node, imports)
            
            assert 'json' in imports
            assert 'collections' in imports
            assert len(imports) == 2
            
        finally:
            os.unlink(temp_file)
    
    def test_handle_from_import_with_module(self):
        """Test _handle_from_import with module name."""
        temp_file = self._create_temp_file("# test")
        try:
            discoverer = RouteDiscoverer(temp_file)
            imports = set()
            
            # Create ast.ImportFrom node with module
            from_import_node = ast.ImportFrom(
                module='django.urls',
                names=[ast.alias(name='path', asname=None)],
                level=0
            )
            
            discoverer._handle_from_import(from_import_node, imports)
            
            assert 'django.urls' in imports
            assert len(imports) == 1
            
        finally:
            os.unlink(temp_file)
    
    def test_handle_from_import_without_module(self):
        """Test _handle_from_import when module is None."""
        temp_file = self._create_temp_file("# test")
        try:
            discoverer = RouteDiscoverer(temp_file)
            imports = set()
            
            # Create ast.ImportFrom node without module (relative import)
            from_import_node = ast.ImportFrom(
                module=None,  # Relative import like "from . import something"
                names=[ast.alias(name='something', asname=None)],
                level=1
            )
            
            discoverer._handle_from_import(from_import_node, imports)
            
            # Should not add anything when module is None
            assert len(imports) == 0
            
        finally:
            os.unlink(temp_file)
    
    def test_extract_imports_from_ast_integration(self):
        """Test the refactored _extract_imports_from_ast method with real code."""
        temp_file = self._create_temp_file("# test")
        try:
            discoverer = RouteDiscoverer(temp_file)
            
            # Test with Python code containing various import types
            source_code = '''
import sys
import os, json
from flask import Flask, request
from django.urls import path
from . import relative_module
'''
            
            imports = discoverer._extract_imports_from_ast(source_code)
            
            assert imports is not None
            assert 'sys' in imports
            assert 'os' in imports
            assert 'json' in imports
            assert 'flask' in imports
            assert 'django.urls' in imports
            # Relative imports (module=None) should not be included
            assert 'relative_module' not in imports
            
        finally:
            os.unlink(temp_file)
    
    def test_extract_imports_from_ast_syntax_error(self):
        """Test _extract_imports_from_ast with syntax error."""
        temp_file = self._create_temp_file("# test")
        try:
            discoverer = RouteDiscoverer(temp_file)
            
            # Test with invalid Python syntax
            invalid_source = '''
import sys
def invalid_function(
    # Missing closing parenthesis
'''
            
            imports = discoverer._extract_imports_from_ast(invalid_source)
            
            # Should return None for syntax errors
            assert imports is None
            
        finally:
            os.unlink(temp_file)
    
    def test_refactored_methods_maintain_original_behavior(self):
        """Test that refactored methods maintain the original behavior."""
        temp_file = self._create_temp_file("# test")
        try:
            discoverer = RouteDiscoverer(temp_file)
            
            # Test with various import patterns that should be handled correctly
            test_cases = [
                # Simple imports
                ("import flask", {'flask'}),
                ("import django", {'django'}),
                ("import fastapi", {'fastapi'}),
                
                # Multiple imports
                ("import sys, os, json", {'sys', 'os', 'json'}),
                
                # From imports
                ("from flask import Flask", {'flask'}),
                ("from django.urls import path", {'django.urls'}),
                
                # Mixed imports
                ("import sys\nfrom flask import Flask", {'sys', 'flask'}),
                
                # Relative imports (should not be included when module is None)
                ("from . import something", set()),
                # Note: "from ..parent import module" actually has module="parent" with level=2
                ("from ..parent import module", {'parent'}),
            ]
            
            for source, expected_imports in test_cases:
                result = discoverer._extract_imports_from_ast(source)
                assert result == expected_imports, f"Failed for source: {source}"
                
        finally:
            os.unlink(temp_file)