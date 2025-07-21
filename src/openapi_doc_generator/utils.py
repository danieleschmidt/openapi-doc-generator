"""Utility functions for openapi-doc-generator."""

from __future__ import annotations

import ast
import hashlib
from functools import lru_cache


def echo(value: object | None = None) -> object | None:
    """Return the provided value unchanged."""
    return value


# LRU cache for parsed AST trees to avoid repeated parsing
@lru_cache(maxsize=128)
def _parse_ast_cached(source_hash: str, source: str, filename: str) -> ast.AST:
    """Internal cached AST parsing function.
    
    Uses source hash as cache key to ensure content changes invalidate cache.
    """
    return ast.parse(source, filename=filename)


def get_cached_ast(source: str, filename: str) -> ast.AST:
    """Get parsed AST with caching for performance optimization.
    
    Args:
        source: Python source code to parse
        filename: Filename for error reporting
        
    Returns:
        Parsed AST tree
        
    Raises:
        SyntaxError: If source code has syntax errors
    """
    # Create hash of source content for cache key
    source_hash = hashlib.sha256(source.encode('utf-8')).hexdigest()[:16]
    
    return _parse_ast_cached(source_hash, source, filename)


def _clear_ast_cache() -> None:
    """Clear the AST cache. Useful for testing."""
    _parse_ast_cached.cache_clear()
