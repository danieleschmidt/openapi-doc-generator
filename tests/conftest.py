"""Pytest configuration and shared fixtures."""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import Mock

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_flask_app() -> str:
    """Sample Flask application code for testing."""
    return '''
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/api/users", methods=["GET", "POST"])
def users():
    """User management endpoint."""
    if request.method == "GET":
        return jsonify({"users": []})
    return jsonify({"message": "User created"}), 201

@app.route("/api/users/<int:user_id>", methods=["GET", "PUT", "DELETE"])
def user_detail(user_id: int):
    """Individual user operations."""
    return jsonify({"user_id": user_id})

@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)
'''


@pytest.fixture
def sample_fastapi_app() -> str:
    """Sample FastAPI application code for testing."""
    return '''
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Test API", version="1.0.0")

class User(BaseModel):
    id: int
    name: str
    email: str

@app.get("/api/users", response_model=List[User])
async def get_users():
    """Get all users."""
    return []

@app.post("/api/users", response_model=User)
async def create_user(user: User):
    """Create a new user."""
    return user

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
'''


@pytest.fixture
def sample_openapi_spec() -> Dict[str, Any]:
    """Sample OpenAPI specification for testing."""
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Test API",
            "version": "1.0.0"
        },
        "paths": {
            "/api/users": {
                "get": {
                    "summary": "Get all users",
                    "responses": {
                        "200": {
                            "description": "List of users",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {"$ref": "#/components/schemas/User"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "User": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "name": {"type": "string"},
                        "email": {"type": "string"}
                    },
                    "required": ["id", "name", "email"]
                }
            }
        }
    }


@pytest.fixture
def mock_file_system(temp_dir: Path):
    """Mock file system with sample application files."""
    flask_app = temp_dir / "flask_app.py"
    fastapi_app = temp_dir / "fastapi_app.py"
    
    flask_app.write_text('''
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/api/test")
def test():
    return jsonify({"message": "test"})
''')
    
    fastapi_app.write_text('''
from fastapi import FastAPI

app = FastAPI()

@app.get("/api/test")
def test():
    return {"message": "test"}
''')
    
    return {
        "flask_app": str(flask_app),
        "fastapi_app": str(fastapi_app),
        "temp_dir": temp_dir
    }


@pytest.fixture
def mock_ast_tree():
    """Mock AST tree for testing."""
    mock = Mock()
    mock.body = []
    return mock


@pytest.fixture(autouse=True)
def reset_caches():
    """Reset any global caches between tests."""
    # Clear any module-level caches if they exist
    yield
    # Cleanup after test


# Markers for different test types
pytest_plugins = ["pytest_cov"]


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "security: mark test as a security test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )


@pytest.fixture
def performance_timer():
    """Timer fixture for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.perf_counter()
        
        def stop(self):
            self.end_time = time.perf_counter()
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()


@pytest.fixture
def security_test_config():
    """Configuration for security testing."""
    return {
        "scan_timeout": 30,
        "vulnerability_threshold": "medium",
        "exclude_paths": ["tests/", "__pycache__/"],
        "include_patterns": ["*.py"]
    }