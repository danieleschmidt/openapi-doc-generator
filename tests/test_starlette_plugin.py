"""Tests for Starlette route discovery plugin."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from openapi_doc_generator.plugins.starlette import StarlettePlugin
from openapi_doc_generator.discovery import RouteInfo


class TestStarlettePlugin:
    """Test Starlette plugin functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = StarlettePlugin()

    def test_detect_starlette_import(self):
        """Test detection of Starlette applications."""
        starlette_code = """
from starlette.applications import Starlette
from starlette.routing import Route

app = Starlette()
"""
        assert self.plugin.detect(starlette_code) is True

    def test_detect_starlette_lowercase(self):
        """Test detection works with lowercase starlette."""
        starlette_code = "import starlette"
        assert self.plugin.detect(starlette_code) is True

    def test_detect_no_starlette(self):
        """Test non-Starlette code is not detected."""
        other_code = """
from flask import Flask
app = Flask(__name__)
"""
        assert self.plugin.detect(other_code) is False

    def test_discover_basic_routes(self):
        """Test discovery of basic Starlette routes."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse

async def homepage(request):
    return JSONResponse({'message': 'Hello, world!'})

async def user_detail(request):
    user_id = request.path_params['user_id']
    return JSONResponse({'user_id': user_id})

routes = [
    Route('/', homepage, methods=['GET']),
    Route('/users/{user_id}', user_detail, methods=['GET', 'POST']),
    Route('/health', lambda r: JSONResponse({'status': 'ok'}), methods=['GET']),
]

app = Starlette(routes=routes)
""")
            f.flush()
            
            routes = self.plugin.discover(f.name)
            
        Path(f.name).unlink()  # Clean up
        
        assert len(routes) == 3
        
        # Check homepage route
        homepage_route = next(r for r in routes if r.path == '/')
        assert homepage_route.methods == ['GET']
        assert homepage_route.name == 'homepage'
        
        # Check user detail route
        user_route = next(r for r in routes if r.path == '/users/{user_id}')
        assert user_route.methods == ['GET', 'POST']
        assert user_route.name == 'user_detail'
        
        # Check health route
        health_route = next(r for r in routes if r.path == '/health')
        assert health_route.methods == ['GET']
        assert 'lambda' in health_route.name or health_route.name == 'health'

    def test_discover_mount_routes(self):
        """Test discovery of mounted Starlette routes."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse

async def api_v1_users(request):
    return JSONResponse({'users': []})

api_routes = [
    Route('/users', api_v1_users, methods=['GET']),
]

routes = [
    Mount('/api/v1', routes=api_routes),
]

app = Starlette(routes=routes)
""")
            f.flush()
            
            routes = self.plugin.discover(f.name)
            
        Path(f.name).unlink()  # Clean up
        
        assert len(routes) == 1
        route = routes[0]
        assert route.path == '/api/v1/users'
        assert route.methods == ['GET']
        assert route.name == 'api_v1_users'

    def test_discover_websocket_routes(self):
        """Test discovery of WebSocket routes."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute

async def websocket_endpoint(websocket):
    await websocket.accept()
    await websocket.send_text("Hello WebSocket!")
    await websocket.close()

routes = [
    WebSocketRoute('/ws', websocket_endpoint),
]

app = Starlette(routes=routes)
""")
            f.flush()
            
            routes = self.plugin.discover(f.name)
            
        Path(f.name).unlink()  # Clean up
        
        assert len(routes) == 1
        route = routes[0]
        assert route.path == '/ws'
        assert route.methods == ['WEBSOCKET']
        assert route.name == 'websocket_endpoint'

    def test_discover_no_routes(self):
        """Test handling of Starlette app with no routes."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
from starlette.applications import Starlette

app = Starlette()
""")
            f.flush()
            
            routes = self.plugin.discover(f.name)
            
        Path(f.name).unlink()  # Clean up
        
        assert routes == []

    def test_discover_file_not_found(self):
        """Test handling of non-existent files."""
        routes = self.plugin.discover('/nonexistent/file.py')
        assert routes == []

    def test_discover_invalid_python(self):
        """Test handling of invalid Python code."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("invalid python code {[}")
            f.flush()
            
            routes = self.plugin.discover(f.name)
            
        Path(f.name).unlink()  # Clean up
        
        assert routes == []

    def test_discover_route_with_docstring(self):
        """Test discovery of routes with docstrings."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse

async def get_users(request):
    '''Get all users from the system.'''
    return JSONResponse({'users': []})

routes = [
    Route('/users', get_users, methods=['GET']),
]

app = Starlette(routes=routes)
""")
            f.flush()
            
            routes = self.plugin.discover(f.name)
            
        Path(f.name).unlink()  # Clean up
        
        assert len(routes) == 1
        route = routes[0]
        assert route.docstring == 'Get all users from the system.'

    def test_discover_complex_nested_structure(self):
        """Test discovery with complex nested routing structures."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse

async def root_handler(request):
    return JSONResponse({'message': 'root'})

async def api_handler(request):
    return JSONResponse({'api': 'v1'})

async def admin_handler(request):
    return JSONResponse({'admin': True})

admin_routes = [
    Route('/dashboard', admin_handler, methods=['GET']),
]

api_routes = [
    Route('/status', api_handler, methods=['GET']),
    Mount('/admin', routes=admin_routes),
]

routes = [
    Route('/', root_handler, methods=['GET']),
    Mount('/api/v1', routes=api_routes),
]

app = Starlette(routes=routes)
""")
            f.flush()
            
            routes = self.plugin.discover(f.name)
            
        Path(f.name).unlink()  # Clean up
        
        assert len(routes) == 3
        
        paths = [r.path for r in routes]
        assert '/' in paths
        assert '/api/v1/status' in paths
        assert '/api/v1/admin/dashboard' in paths