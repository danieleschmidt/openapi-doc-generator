"""Test data and mock objects for testing."""

import json
from pathlib import Path
from typing import Dict, Any, List

# Mock performance metrics data
PERFORMANCE_METRICS = {
    "framework_detection": {
        "duration_ms": 0.57,
        "framework": "flask",
        "confidence": 0.95
    },
    "route_discovery": {
        "duration_ms": 1.52,
        "route_count": 5,
        "routes": [
            {"path": "/api/users", "methods": ["GET", "POST"]},
            {"path": "/api/users/<int:user_id>", "methods": ["GET", "PUT", "DELETE"]},
            {"path": "/health", "methods": ["GET"]}
        ]
    },
    "schema_inference": {
        "duration_ms": 2.34,
        "schemas_found": 3,
        "cache_hits": 2,
        "cache_misses": 1
    },
    "document_generation": {
        "duration_ms": 5.67,
        "output_size_bytes": 12450,
        "format": "openapi"
    }
}

# Mock route discovery results
DISCOVERED_ROUTES = [
    {
        "path": "/api/users",
        "methods": ["GET", "POST"],
        "handler": "users",
        "docstring": "User management endpoint.",
        "parameters": [],
        "response_schemas": {
            "GET": {"type": "array", "items": {"$ref": "#/components/schemas/User"}},
            "POST": {"$ref": "#/components/schemas/User"}
        }
    },
    {
        "path": "/api/users/{user_id}",
        "methods": ["GET", "PUT", "DELETE"],
        "handler": "user_detail",
        "docstring": "Individual user operations.",
        "parameters": [
            {
                "name": "user_id",
                "in": "path",
                "required": True,
                "schema": {"type": "integer"}
            }
        ],
        "response_schemas": {
            "GET": {"$ref": "#/components/schemas/User"},
            "PUT": {"$ref": "#/components/schemas/User"},
            "DELETE": {"type": "object", "properties": {}}
        }
    },
    {
        "path": "/health",
        "methods": ["GET"],
        "handler": "health",
        "docstring": "Health check endpoint.",
        "parameters": [],
        "response_schemas": {
            "GET": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "example": "ok"}
                }
            }
        }
    }
]

# Mock schema definitions
INFERRED_SCHEMAS = {
    "User": {
        "type": "object",
        "properties": {
            "id": {"type": "integer", "example": 1},
            "name": {"type": "string", "example": "John Doe"},
            "email": {"type": "string", "format": "email", "example": "john@example.com"}
        },
        "required": ["id", "name", "email"]
    },
    "UserCreate": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "example": "John Doe"},
            "email": {"type": "string", "format": "email", "example": "john@example.com"}
        },
        "required": ["name", "email"]
    }
}

# Mock CLI output examples
CLI_OUTPUT_EXAMPLES = {
    "success": {
        "exit_code": 0,
        "stdout": "✅ Documentation generated successfully\n📄 Output: api-docs.json\n📊 Found 5 routes across 3 endpoints\n",
        "stderr": ""
    },
    "file_not_found": {
        "exit_code": 1,
        "stdout": "",
        "stderr": "❌ Error CLI001: App file not found at ./nonexistent.py\n"
    },
    "invalid_format": {
        "exit_code": 1,
        "stdout": "",
        "stderr": "❌ Error: Invalid output format 'invalid'. Supported: openapi, markdown, html, tests, guide\n"
    },
    "with_performance": {
        "exit_code": 0,
        "stdout": "✅ Documentation generated successfully\n📊 Performance Summary:\n  Framework Detection: 0.57ms\n  Route Discovery: 1.52ms (5 routes)\n  Schema Inference: 2.34ms (3 schemas)\n  Document Generation: 5.67ms\n📄 Output: api-docs.json\n",
        "stderr": ""
    }
}

# Test file paths and configurations
TEST_FILES = {
    "valid_flask_app": "tests/fixtures/apps/flask_app.py",
    "valid_fastapi_app": "tests/fixtures/apps/fastapi_app.py",
    "invalid_python_file": "tests/fixtures/apps/invalid_syntax.py",
    "empty_file": "tests/fixtures/apps/empty.py",
    "config_file": "tests/fixtures/config/test_config.yaml",
    "openapi_spec": "tests/fixtures/specs/valid_openapi.json",
    "invalid_openapi_spec": "tests/fixtures/specs/invalid_openapi.json"
}

# Mock configuration data
CONFIG_DATA = {
    "frameworks": {
        "fastapi": {
            "include_internal": False,
            "example_generation": True
        },
        "flask": {
            "restful_support": True,
            "blueprint_discovery": True
        }
    },
    "output": {
        "formats": ["openapi", "markdown"],
        "include_examples": True,
        "authentication_docs": True
    },
    "performance": {
        "enable_caching": True,
        "cache_ttl": 3600,
        "max_memory_usage": "512MB"
    }
}

# Security scan mock results
SECURITY_SCAN_RESULTS = {
    "vulnerabilities": [],
    "bandit_results": {
        "errors": [],
        "generated_at": "2025-01-01T00:00:00Z",
        "metrics": {
            "loc": 1234,
            "nosec": 0
        },
        "results": []
    },
    "safety_results": {
        "scanned": 25,
        "vulnerabilities": []
    }
}

def create_temp_app_file(content: str, temp_dir: Path, filename: str = "app.py") -> Path:
    """Create a temporary application file for testing."""
    app_file = temp_dir / filename
    app_file.write_text(content)
    return app_file

def create_temp_config_file(config: Dict[str, Any], temp_dir: Path, filename: str = "config.yaml") -> Path:
    """Create a temporary configuration file for testing."""
    import yaml
    config_file = temp_dir / filename
    config_file.write_text(yaml.dump(config))
    return config_file

def create_temp_openapi_spec(spec: Dict[str, Any], temp_dir: Path, filename: str = "openapi.json") -> Path:
    """Create a temporary OpenAPI specification file for testing."""
    spec_file = temp_dir / filename
    spec_file.write_text(json.dumps(spec, indent=2))
    return spec_file