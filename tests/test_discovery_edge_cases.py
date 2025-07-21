"""Tests for edge cases and error scenarios in route discovery."""

import pytest
from openapi_doc_generator.discovery import RouteDiscoverer


def test_ast_parsing_failure_fallback(tmp_path):
    """Test that malformed Python files fall back to string matching."""
    app = tmp_path / "malformed.py"
    # Write syntactically invalid Python that will cause ast.parse to fail
    app.write_text(
        "from fastapi import FastAPI\n"
        "def broken_syntax(\n"  # Missing closing parenthesis
        "    invalid python syntax here\n"
        "app = FastAPI()\n"
    )
    
    discoverer = RouteDiscoverer(str(app))
    # Should fall back to string detection and still detect FastAPI
    framework = discoverer._detect_framework(app.read_text())
    assert framework == "fastapi"


def test_import_from_without_module(tmp_path):
    """Test handling of 'from' imports without module name."""
    app = tmp_path / "test_app.py"
    app.write_text("""
from . import something
import os
from fastapi import FastAPI
""")
    discoverer = RouteDiscoverer(str(app))
    framework = discoverer._detect_framework(app.read_text())
    assert framework == "fastapi"


def test_detect_framework_fallback_django(tmp_path):
    """Test fallback detection for Django framework."""
    app = tmp_path / "test_app.py"
    app.write_text("""
# This file doesn't have proper imports but mentions django
def some_function():
    return "django code here"
""")
    discoverer = RouteDiscoverer(str(app))
    framework = discoverer._detect_framework_fallback(app.read_text())
    assert framework == "django"


def test_detect_framework_fallback_express(tmp_path):
    """Test fallback detection for Express framework."""
    app = tmp_path / "test_app.js"
    app.write_text("""
// JavaScript file with express reference
const app = require('express')();
app.get('/test', handler);
""")
    discoverer = RouteDiscoverer(str(app))
    framework = discoverer._detect_framework_fallback(app.read_text())
    assert framework == "express"


def test_detect_framework_fallback_flask(tmp_path):
    """Test fallback detection for Flask framework."""
    app = tmp_path / "test_app.py"
    app.write_text("""
# Python file with flask in comments
# This uses flask framework
def create_app():
    return app
""")
    discoverer = RouteDiscoverer(str(app))
    framework = discoverer._detect_framework_fallback(app.read_text())
    assert framework == "flask"


def test_detect_framework_fallback_none(tmp_path):
    """Test fallback detection returns None for unknown frameworks."""
    app = tmp_path / "test_app.py"
    app.write_text("""
# Generic Python file with no framework references
import os
import sys

def generic_function():
    return "hello world"
""")
    discoverer = RouteDiscoverer(str(app))
    framework = discoverer._detect_framework_fallback(app.read_text())
    assert framework is None


def test_ast_import_edge_cases(tmp_path):
    """Test AST parsing with various import patterns."""
    app = tmp_path / "complex_imports.py"
    app.write_text("""
import os
import sys as system
from typing import List
from fastapi import FastAPI, APIRouter
from flask import Flask  # This should detect fastapi first due to order
import django.core
""")
    
    discoverer = RouteDiscoverer(str(app))
    framework = discoverer._detect_framework(app.read_text())
    # Should detect fastapi since it comes first in the detection order
    assert framework == "fastapi"


def test_django_route_name_extraction(tmp_path):
    """Test Django route discovery with name extraction."""
    app = tmp_path / "django_app.py"
    app.write_text("""
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/users/', views.user_list, name='user_list'),
    path('api/user/<int:pk>/', views.user_detail),
]
""")
    
    routes = RouteDiscoverer(str(app)).discover()
    # Find route with name
    named_routes = [r for r in routes if r.name]
    assert len(named_routes) >= 1
    assert any(r.name == "user_list" for r in named_routes)


def test_abstract_plugin_methods():
    """Test that abstract methods of RoutePlugin cannot be instantiated."""
    from openapi_doc_generator.discovery import RoutePlugin
    
    # Should not be able to instantiate abstract base class
    with pytest.raises(TypeError):
        RoutePlugin()


def test_file_not_found_error():
    """Test that RouteDiscoverer raises FileNotFoundError for non-existent files."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        RouteDiscoverer("/non/existent/file.py")


def test_no_matching_plugins(tmp_path):
    """Test behavior when no plugins match the source file."""
    app = tmp_path / "unknown_framework.py"
    app.write_text("""
# Generic Python file with no recognizable framework
import os
import sys

def main():
    print("Hello, world!")
""")
    
    # Should raise ValueError when no framework is detected
    with pytest.raises(ValueError, match="Unable to determine framework"):
        RouteDiscoverer(str(app)).discover()


def test_detect_framework_fallback_to_string_matching(tmp_path):
    """Test that framework detection falls back to string matching when AST parsing works but no imports found."""
    app = tmp_path / "fallback_test.py"
    app.write_text("""
# Valid Python but no framework imports
# But content mentions fastapi in comments
def setup():
    # Using fastapi framework
    pass
""")
    
    discoverer = RouteDiscoverer(str(app))
    framework = discoverer._detect_framework(app.read_text())
    # Should fall back to string matching and detect fastapi
    assert framework == "fastapi"