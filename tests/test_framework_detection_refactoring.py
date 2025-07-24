"""Tests for refactored framework detection methods."""

import tempfile
import os
from openapi_doc_generator.discovery import RouteDiscoverer


class TestFrameworkDetectionRefactoring:
    """Test the refactored framework detection methods."""
    
    def _create_temp_file(self, content: str) -> str:
        """Helper to create temporary file with content."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(content)
            return f.name
    
    def test_detect_fastapi_patterns(self):
        """Test FastAPI pattern detection."""
        temp_file = self._create_temp_file("# test")
        try:
            discoverer = RouteDiscoverer(temp_file)
            
            # Test positive cases (these methods receive lowercased input)
            assert discoverer._detect_fastapi_patterns("fastapi import") is True
            assert discoverer._detect_fastapi_patterns("from fastapi import") is True
            assert discoverer._detect_fastapi_patterns("fastapi") is True
            
            # Test negative cases
            assert discoverer._detect_fastapi_patterns("flask import") is False
            assert discoverer._detect_fastapi_patterns("django") is False
            
        finally:
            os.unlink(temp_file)
    
    def test_detect_flask_patterns(self):
        """Test Flask pattern detection."""
        temp_file = self._create_temp_file("# test")
        try:
            discoverer = RouteDiscoverer(temp_file)
            
            # Test positive cases (these methods receive lowercased input)
            assert discoverer._detect_flask_patterns("flask import") is True
            assert discoverer._detect_flask_patterns("from flask import") is True
            assert discoverer._detect_flask_patterns("flask") is True
            
            # Test negative cases
            assert discoverer._detect_flask_patterns("fastapi import") is False
            assert discoverer._detect_flask_patterns("django") is False
            
        finally:
            os.unlink(temp_file)
    
    def test_detect_django_patterns(self):
        """Test Django pattern detection."""
        temp_file = self._create_temp_file("# test")
        try:
            discoverer = RouteDiscoverer(temp_file)
            
            # Test positive cases (these methods receive lowercased input)
            assert discoverer._detect_django_patterns("django import") is True
            assert discoverer._detect_django_patterns("from django import") is True
            assert discoverer._detect_django_patterns("django") is True
            
            # Test negative cases
            assert discoverer._detect_django_patterns("flask import") is False
            assert discoverer._detect_django_patterns("fastapi") is False
            
        finally:
            os.unlink(temp_file)
    
    def test_detect_express_patterns(self):
        """Test Express.js pattern detection."""
        temp_file = self._create_temp_file("# test")
        try:
            discoverer = RouteDiscoverer(temp_file)
            
            # Test positive cases (these methods receive lowercased input)
            assert discoverer._detect_express_patterns("express import") is True
            assert discoverer._detect_express_patterns("require('express')") is True
            assert discoverer._detect_express_patterns("express") is True
            
            # Test negative cases
            assert discoverer._detect_express_patterns("flask import") is False
            assert discoverer._detect_express_patterns("django") is False
            
        finally:
            os.unlink(temp_file)
    
    def test_detect_framework_fallback_with_refactored_methods(self):
        """Test that the refactored _detect_framework_fallback works correctly."""
        temp_file = self._create_temp_file("# test")
        try:
            discoverer = RouteDiscoverer(temp_file)
            
            # Test each framework detection
            assert discoverer._detect_framework_fallback("from fastapi import FastAPI") == "fastapi"
            assert discoverer._detect_framework_fallback("from flask import Flask") == "flask"
            assert discoverer._detect_framework_fallback("from django.urls import path") == "django"
            assert discoverer._detect_framework_fallback("const express = require('express')") == "express"
            
            # Test no framework detected
            assert discoverer._detect_framework_fallback("import random") is None
            
            # Test case insensitivity
            assert discoverer._detect_framework_fallback("FROM FASTAPI IMPORT") == "fastapi"
            
        finally:
            os.unlink(temp_file)
    
    def test_framework_detection_order_priority(self):
        """Test that framework detection follows the expected priority order."""
        temp_file = self._create_temp_file("# test")
        try:
            discoverer = RouteDiscoverer(temp_file)
            
            # Test that FastAPI has priority when multiple frameworks are mentioned
            mixed_source = "from fastapi import FastAPI; import flask"
            assert discoverer._detect_framework_fallback(mixed_source) == "fastapi"
            
            # Test Flask priority over Django
            mixed_source2 = "from flask import Flask; import django"
            assert discoverer._detect_framework_fallback(mixed_source2) == "flask"
            
        finally:
            os.unlink(temp_file)