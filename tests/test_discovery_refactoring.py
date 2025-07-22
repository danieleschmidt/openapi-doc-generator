"""Tests to ensure discovery refactoring preserves behavior."""

from openapi_doc_generator.discovery import RouteDiscoverer


class TestDetectFrameworkBehavior:
    """Test suite to verify _detect_framework behavior before refactoring."""

    def test_detect_fastapi_import(self, tmp_path):
        """Test detection of FastAPI through imports."""
        app = tmp_path / "app.py"
        app.write_text("""
from fastapi import FastAPI
import os

app = FastAPI()
""")

        discoverer = RouteDiscoverer(str(app))
        framework = discoverer._detect_framework(app.read_text())
        assert framework == "fastapi"

    def test_detect_flask_import(self, tmp_path):
        """Test detection of Flask through imports."""
        app = tmp_path / "app.py"
        app.write_text("""
from flask import Flask
import sys

app = Flask(__name__)
""")

        discoverer = RouteDiscoverer(str(app))
        framework = discoverer._detect_framework(app.read_text())
        assert framework == "flask"

    def test_detect_django_import(self, tmp_path):
        """Test detection of Django through imports."""
        app = tmp_path / "app.py"
        app.write_text("""
from django.urls import path
import django.conf

urlpatterns = []
""")

        discoverer = RouteDiscoverer(str(app))
        framework = discoverer._detect_framework(app.read_text())
        assert framework == "django"

    def test_detect_multiple_frameworks_fastapi_wins(self, tmp_path):
        """Test that FastAPI is detected first when multiple frameworks present."""
        app = tmp_path / "app.py"
        app.write_text("""
from fastapi import FastAPI
from flask import Flask
from django.urls import path

# Multiple frameworks in same file
""")

        discoverer = RouteDiscoverer(str(app))
        framework = discoverer._detect_framework(app.read_text())
        assert framework == "fastapi"

    def test_detect_syntax_error_fallback(self, tmp_path):
        """Test fallback to string matching when syntax error occurs."""
        app = tmp_path / "app.py"
        app.write_text("""
from fastapi import FastAPI
def broken_syntax(
    # Missing closing parenthesis
app = FastAPI()
""")

        discoverer = RouteDiscoverer(str(app))
        framework = discoverer._detect_framework(app.read_text())
        assert framework == "fastapi"  # Should fall back to string matching

    def test_detect_no_framework_imports_fallback(self, tmp_path):
        """Test fallback when no framework imports found."""
        app = tmp_path / "app.py"
        app.write_text("""
import os
import sys

# No framework imports, but mentions fastapi in comment
def setup():
    # This uses fastapi
    pass
""")

        discoverer = RouteDiscoverer(str(app))
        framework = discoverer._detect_framework(app.read_text())
        assert framework == "fastapi"  # Should fall back to string matching

    def test_detect_import_from_patterns(self, tmp_path):
        """Test detection with various import patterns."""
        app = tmp_path / "app.py"
        app.write_text("""
from . import something
from fastapi.responses import JSONResponse
import typing
""")

        discoverer = RouteDiscoverer(str(app))
        framework = discoverer._detect_framework(app.read_text())
        assert framework == "fastapi"

    def test_detect_no_imports_no_fallback_match(self, tmp_path):
        """Test when no imports and no fallback match."""
        app = tmp_path / "app.py"
        app.write_text("""
import os
import sys

def generic_function():
    return "hello world"
""")

        discoverer = RouteDiscoverer(str(app))
        framework = discoverer._detect_framework(app.read_text())
        assert framework is None
