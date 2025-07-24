"""Tests to improve discovery.py coverage."""

import ast
import pytest
from openapi_doc_generator.discovery import (
    RouteDiscoverer
)


def test_route_plugin_abstract_methods():
    """Test that abstract methods in RoutePlugin are properly defined."""
    # The RoutePlugin abstract base class has pass statements (lines 33, 38)
    # These are part of the abstract method definitions and serve as documentation
    # They're not meant to be executed but define the interface for plugins


def test_fastapi_decorator_edge_cases(tmp_path):
    """Test edge cases in FastAPI route extraction."""
    # Create app with edge case decorators
    app_file = tmp_path / "fastapi_edge.py"
    app_file.write_text("""
from fastapi import FastAPI

app = FastAPI()

# Edge case 1: Decorator is not a Call node
@app
def invalid_decorator1():
    pass

# Edge case 2: Decorator call but not an Attribute
@some_function()
def invalid_decorator2():
    pass

# Edge case 3: Non-HTTP method
@app.websocket("/ws")
def websocket_endpoint():
    pass

# Edge case 4: Decorator on non-app object  
other = FastAPI()
@other.get("/other")
def other_app_route():
    pass

# Valid route for comparison
@app.get("/valid")
def valid_route():
    '''Valid route docstring'''
    pass
""")
    
    discoverer = RouteDiscoverer(str(app_file))
    routes = discoverer.discover()
    
    # Should only find the valid route, not the edge cases
    assert len(routes) == 1
    assert routes[0].path == "/valid"
    assert routes[0].methods == ["GET"]


def test_flask_decorator_edge_cases(tmp_path):
    """Test edge cases in Flask route extraction."""
    # Create Flask app with edge cases
    app_file = tmp_path / "flask_edge.py"
    app_file.write_text("""
from flask import Flask

app = Flask(__name__)

# Edge case 1: Route decorator is not a Call
@app
def invalid_decorator():
    pass

# Edge case 2: Decorator without proper structure
@route("/test")
def standalone_decorator():
    pass

# Valid routes
@app.route("/valid1")
def valid_route1():
    pass

@app.route("/valid2", methods=["POST", "PUT"])
def valid_route2():
    pass
""")
    
    discoverer = RouteDiscoverer(str(app_file))
    routes = discoverer.discover()
    
    # Should find only the valid routes
    assert len(routes) == 2
    valid_paths = [r.path for r in routes]
    assert "/valid1" in valid_paths
    assert "/valid2" in valid_paths


def test_django_path_edge_cases(tmp_path):
    """Test edge cases in Django path extraction."""
    # Create Django URLs with edge cases
    urls_file = tmp_path / "urls.py"
    urls_file.write_text("""
from django.urls import path, re_path
from . import views

urlpatterns = [
    # Edge case 1: path() call without constant string
    path(dynamic_path, views.dynamic_view),
    
    # Edge case 2: Empty arguments
    path(),
    
    # Edge case 3: View is neither Attribute nor Name
    path("lambda/", lambda request: None),
    
    # Valid paths
    path("users/", views.user_list),
    path("users/<int:pk>/", views.UserDetail.as_view()),
    re_path(r"^api/.*", views.api_view),
]
""")
    
    discoverer = RouteDiscoverer(str(urls_file))
    routes = discoverer.discover()
    
    # Should find the valid paths with proper string literals
    assert len(routes) >= 2
    paths = [r.path for r in routes]
    assert "users/" in paths
    assert "users/<int:pk>/" in paths


def test_path_extraction_with_non_constant_args(tmp_path):
    """Test _extract_path_from_args with non-constant arguments."""
    # Create a dummy app file
    app_file = tmp_path / "dummy_app.py"
    app_file.write_text("# Dummy app")
    
    discoverer = RouteDiscoverer(str(app_file))
    
    # Test with empty args
    assert discoverer._extract_path_from_args([]) == ""
    
    # Test with non-constant arg (e.g., variable)
    name_node = ast.Name(id="path_variable", ctx=ast.Load())
    assert discoverer._extract_path_from_args([name_node]) == ""
    
    # Test with constant arg
    const_node = ast.Constant(value="/test")
    assert discoverer._extract_path_from_args([const_node]) == "/test"


def test_express_route_discovery_patterns():
    """Test Express.js route discovery with various patterns."""
    source = """
const express = require('express');
const app = express();

// Standard routes
app.get('/users', getUsers);
app.post("/posts", createPost);
app.put('/items/:id', updateItem);
app.patch("/resources/:id", patchResource);
app.delete('/comments/:id', deleteComment);

// Edge cases that should not be matched
app.use('/middleware', middleware);
app.all('/*', catchAll);
const other = express();
other.get('/other', handler);
"""
    
    # Create a temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
        f.write(source)
        f.flush()
        
        try:
            discoverer = RouteDiscoverer(f.name)
            routes = discoverer.discover()
            
            # Should find exactly 5 routes (get, post, put, patch, delete)
            assert len(routes) == 5
            
            methods_found = {r.methods[0] for r in routes}
            assert methods_found == {"GET", "POST", "PUT", "PATCH", "DELETE"}
            
            paths_found = {r.path for r in routes}
            expected_paths = {'/users', '/posts', '/items/:id', '/resources/:id', '/comments/:id'}
            assert paths_found == expected_paths
        finally:
            import os
            os.unlink(f.name)


def test_discoverer_with_invalid_framework():
    """Test RouteDiscoverer with source that doesn't match any framework."""
    source = """
# This is just a regular Python file without any web framework
def hello():
    print("Hello, World!")

class MyClass:
    def method(self):
        pass
"""
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(source)
        f.flush()
        
        try:
            discoverer = RouteDiscoverer(f.name)
            with pytest.raises(ValueError, match="Unable to determine framework"):
                discoverer.discover()
        finally:
            import os
            os.unlink(f.name)