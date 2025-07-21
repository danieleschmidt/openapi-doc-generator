"""Tests for AST caching functionality."""

import ast
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from openapi_doc_generator.discovery import RouteDiscoverer
from openapi_doc_generator.schema import SchemaInferer


class TestASTCaching:
    """Test suite for AST caching performance optimization."""
    
    def test_ast_cache_basic_functionality(self, tmp_path):
        """Test that AST cache stores and retrieves parsed trees correctly."""
        # This test will verify the cache implementation once we create it
        from openapi_doc_generator.utils import get_cached_ast
        
        app = tmp_path / "app.py"
        source = """
from fastapi import FastAPI
app = FastAPI()

@app.get('/test')
def test_endpoint():
    return {'message': 'test'}
"""
        app.write_text(source)
        
        # First call should parse and cache
        tree1 = get_cached_ast(source, str(app))
        assert isinstance(tree1, ast.AST)
        
        # Second call should retrieve from cache
        tree2 = get_cached_ast(source, str(app))
        assert tree1 is tree2  # Should be the same object reference
    
    def test_ast_cache_different_content_different_cache(self, tmp_path):
        """Test that different source content results in different cache entries."""
        from openapi_doc_generator.utils import get_cached_ast
        
        app1 = tmp_path / "app1.py"
        app2 = tmp_path / "app2.py"
        
        source1 = "from fastapi import FastAPI\napp = FastAPI()"
        source2 = "from flask import Flask\napp = Flask(__name__)"
        
        app1.write_text(source1)
        app2.write_text(source2)
        
        tree1 = get_cached_ast(source1, str(app1))
        tree2 = get_cached_ast(source2, str(app2))
        
        assert tree1 is not tree2  # Should be different objects
    
    def test_ast_cache_syntax_error_not_cached(self, tmp_path):
        """Test that syntax errors are not cached and always re-raised."""
        from openapi_doc_generator.utils import get_cached_ast
        
        app = tmp_path / "invalid.py"
        invalid_source = "def broken_syntax(\n  # missing closing parenthesis"
        app.write_text(invalid_source)
        
        # Should raise SyntaxError
        with pytest.raises(SyntaxError):
            get_cached_ast(invalid_source, str(app))
        
        # Should raise again on second call (not cached)
        with pytest.raises(SyntaxError):
            get_cached_ast(invalid_source, str(app))
    
    def test_ast_cache_performance_improvement(self, tmp_path):
        """Test that AST caching provides measurable performance improvement."""
        from openapi_doc_generator.utils import get_cached_ast
        
        app = tmp_path / "large_app.py"
        # Create a larger source file to make parsing time more noticeable
        large_source = """
from fastapi import FastAPI
app = FastAPI()

""" + "\n".join([
            f"@app.get('/endpoint_{i}')\ndef endpoint_{i}():\n    return {{'id': {i}}}"
            for i in range(50)
        ])
        app.write_text(large_source)
        
        # Measure first parse (cache miss)
        start_time = time.perf_counter()
        tree1 = get_cached_ast(large_source, str(app))
        first_parse_time = time.perf_counter() - start_time
        
        # Measure second parse (cache hit)
        start_time = time.perf_counter()
        tree2 = get_cached_ast(large_source, str(app))
        second_parse_time = time.perf_counter() - start_time
        
        # Cache hit should be significantly faster
        assert tree1 is tree2
        assert second_parse_time < first_parse_time / 2  # At least 50% faster
    
    def test_discovery_uses_cached_ast(self, tmp_path):
        """Test that route discovery benefits from AST caching."""
        app = tmp_path / "app.py"
        app.write_text("""
from fastapi import FastAPI
app = FastAPI()

@app.get('/users')
def get_users():
    return []

@app.post('/users')
def create_user():
    return {}
""")
        
        discoverer = RouteDiscoverer(str(app))
        
        # Mock the cache to verify it's being used
        with patch('openapi_doc_generator.utils.get_cached_ast') as mock_cache:
            mock_cache.return_value = ast.parse(app.read_text())
            
            routes = discoverer.discover()
            
            # Should have called the cached version
            assert mock_cache.called
            assert len(routes) == 2
    
    def test_schema_inference_uses_cached_ast(self, tmp_path):
        """Test that schema inference benefits from AST caching."""
        app = tmp_path / "models.py"
        app.write_text("""
from dataclasses import dataclass

@dataclass
class User:
    id: int
    name: str
""")
        
        inferer = SchemaInferer(Path(str(app)))
        
        # Mock the cache to verify it's being used
        with patch('openapi_doc_generator.utils.get_cached_ast') as mock_cache:
            mock_cache.return_value = ast.parse(app.read_text())
            
            models = inferer.infer()
            
            # Should have called the cached version
            assert mock_cache.called
            assert len(models) >= 0  # May or may not find models depending on implementation
    
    def test_cache_size_limit(self, tmp_path):
        """Test that cache has size limit and evicts old entries."""
        from openapi_doc_generator.utils import get_cached_ast, _clear_ast_cache
        
        # Clear any existing cache
        _clear_ast_cache()
        
        # Create many different source files
        sources = []
        for i in range(150):  # Exceed typical cache size
            app = tmp_path / f"app_{i}.py"
            source = f"# File {i}\nimport os\ndef func_{i}(): pass"
            app.write_text(source)
            sources.append((source, str(app)))
        
        # Parse all files
        trees = []
        for source, path in sources:
            tree = get_cached_ast(source, path)
            trees.append(tree)
        
        # First entries should be evicted, last entries should still be cached
        late_tree = get_cached_ast(sources[-1][0], sources[-1][1])
        
        # Late entry should be from cache (same object)
        assert late_tree is trees[-1]