"""End-to-end tests for complete documentation workflows."""

import json
import tempfile
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from openapi_doc_generator.cli import main


@pytest.mark.e2e
@pytest.mark.requires_docker
def test_complete_documentation_workflow():
    """Test complete workflow from app analysis to documentation generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create a simple test app
        app_file = tmpdir_path / "test_app.py"
        app_file.write_text("""
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/api/users", methods=["GET", "POST"])
def users():
    '''User management endpoint.'''
    if request.method == "GET":
        return jsonify({"users": []})
    return jsonify({"message": "User created"})

@app.route("/health")
def health():
    '''Health check endpoint.'''
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run()
""")
        
        # Test OpenAPI generation
        openapi_file = tmpdir_path / "openapi.json"
        result = main([
            "--app", str(app_file),
            "--format", "openapi",
            "--output", str(openapi_file)
        ])
        
        assert result == 0
        assert openapi_file.exists()
        
        # Validate OpenAPI spec
        with open(openapi_file) as f:
            spec = json.load(f)
        
        assert "openapi" in spec
        assert "paths" in spec
        assert "/api/users" in spec["paths"]
        assert "/health" in spec["paths"]
        
        # Test markdown generation
        markdown_file = tmpdir_path / "API.md"
        result = main([
            "--app", str(app_file),
            "--format", "markdown",
            "--output", str(markdown_file)
        ])
        
        assert result == 0
        assert markdown_file.exists()
        
        # Validate markdown content
        markdown_content = markdown_file.read_text()
        assert "# API Documentation" in markdown_content
        assert "/api/users" in markdown_content
        assert "/health" in markdown_content


@pytest.mark.e2e
@pytest.mark.requires_network
def test_playground_generation_and_validation():
    """Test HTML playground generation and validation with browser."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create test app
        app_file = tmpdir_path / "simple_app.py"
        app_file.write_text("""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    name: str
    email: str

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/users/")
def create_user(user: User):
    return {"user_id": 1, "user": user}
""")
        
        # Generate HTML playground
        html_file = tmpdir_path / "playground.html"
        result = main([
            "--app", str(app_file),
            "--format", "html",
            "--output", str(html_file)
        ])
        
        assert result == 0
        assert html_file.exists()
        
        # Validate HTML with browser
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(f"file://{html_file}")
            
            # Check for Swagger UI elements
            assert page.title() == "API Documentation"
            assert page.is_visible("[data-testid='swagger-ui']")
            
            # Check for endpoints
            assert page.is_visible("text=/users/")
            assert page.is_visible("text=Hello World")
            
            browser.close()


@pytest.mark.e2e
@pytest.mark.slow
def test_large_application_performance():
    """Test performance with a larger application."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create a larger test application
        app_file = tmpdir_path / "large_app.py"
        routes = []
        for i in range(100):
            routes.append(f'''
@app.route("/api/resource{i}", methods=["GET", "POST", "PUT", "DELETE"])
def resource{i}():
    """Resource {i} management endpoint."""
    return {{"resource_id": {i}}}
''')
        
        app_content = f"""
from flask import Flask
app = Flask(__name__)

{chr(10).join(routes)}

if __name__ == "__main__":
    app.run()
"""
        app_file.write_text(app_content)
        
        # Test performance
        import time
        start_time = time.time()
        
        result = main([
            "--app", str(app_file),
            "--format", "openapi",
            "--output", str(tmpdir_path / "large_openapi.json"),
            "--performance-metrics"
        ])
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        assert result == 0
        assert processing_time < 10.0  # Should complete within 10 seconds
        
        # Verify all routes were discovered
        with open(tmpdir_path / "large_openapi.json") as f:
            spec = json.load(f)
        
        # Should have discovered all 100 routes
        assert len(spec["paths"]) == 100