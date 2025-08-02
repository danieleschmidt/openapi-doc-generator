# User Guide

## Getting Started

### Installation

#### Using pip
```bash
pip install openapi-doc-generator
```

#### Using Docker
```bash
docker pull ghcr.io/danieleschmidt/openapi-doc-generator:latest
```

#### From Source
```bash
git clone https://github.com/danieleschmidt/openapi-doc-generator.git
cd openapi-doc-generator
pip install -e .
```

## Basic Usage

### Generate OpenAPI Specification
```bash
openapi-doc-generator --app ./app.py --format openapi --output openapi.json
```

### Generate Markdown Documentation
```bash
openapi-doc-generator --app ./app.py --format markdown --output API.md
```

### Generate Interactive Playground
```bash
openapi-doc-generator --app ./app.py --format html --output playground.html
```

## Command Line Options

### Required Arguments
- `--app`: Path to your application file

### Output Options
- `--format`: Output format (openapi, markdown, html, tests, guide)
- `--output`: Output file path
- `--title`: API title (default: inferred from application)
- `--api-version`: API version (default: 1.0.0)

### Behavior Options
- `--verbose`: Detailed progress output
- `--quiet`: Minimal output
- `--no-color`: Disable colored output
- `--performance-metrics`: Enable performance tracking
- `--log-format`: Logging format (text, json)

### Advanced Options
- `--old-spec`: Path to old OpenAPI spec for migration guide
- `--tests`: Path to test file for test generation
- `--config`: Path to configuration file

## Supported Frameworks

### Python Frameworks

#### FastAPI
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    id: int
    name: str

@app.get("/users/{user_id}")
async def get_user(user_id: int) -> User:
    """Retrieve user by ID."""
    return User(id=user_id, name="John Doe")
```

#### Flask
```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/api/users/<int:user_id>", methods=["GET"])
def get_user(user_id):
    """Retrieve user by ID."""
    return jsonify({"id": user_id, "name": "John Doe"})
```

#### Django
```python
# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('api/users/<int:user_id>/', views.get_user, name='get_user'),
]

# views.py
from django.http import JsonResponse

def get_user(request, user_id):
    """Retrieve user by ID."""
    return JsonResponse({"id": user_id, "name": "John Doe"})
```

### JavaScript Frameworks

#### Express.js
```javascript
const express = require('express');
const app = express();

/**
 * Retrieve user by ID
 * @param {number} id - User ID
 * @returns {Object} User object
 */
app.get('/api/users/:id', (req, res) => {
    res.json({ id: req.params.id, name: 'John Doe' });
});
```

### GraphQL
```bash
openapi-doc-generator --app ./schema.graphql --format graphql --output schema.json
```

## Output Formats

### OpenAPI Specification
Generates OpenAPI 3.0 compliant JSON/YAML specifications suitable for:
- API documentation tools (Swagger UI, Redoc)
- Code generation tools
- API testing tools
- Contract testing

### Markdown Documentation
Creates human-readable documentation with:
- Endpoint descriptions
- Parameter details
- Request/response examples
- Authentication information
- Error codes and responses

### Interactive Playground
Generates HTML pages with:
- Swagger UI integration
- Live API testing
- Code samples in multiple languages
- Authentication support

### Test Suite Generation
Creates automated test suites:
- Unit tests for each endpoint
- Integration test scenarios
- Performance test templates
- Security test cases

### Migration Guides
Compares API versions and generates:
- Breaking change documentation
- Migration steps
- Deprecated endpoint warnings
- New feature highlights

## Configuration

### Configuration File
Create `config.yaml`:
```yaml
# Framework-specific settings
frameworks:
  fastapi:
    include_internal: false
    example_generation: true
  express:
    typescript_support: true
    middleware_docs: true

# Output configuration
output:
  formats: ["openapi", "markdown"]
  include_examples: true
  authentication_docs: true

# Performance settings
performance:
  enable_caching: true
  cache_ttl: 3600
  max_memory_usage: "512MB"
```

### Environment Variables
```bash
export LOG_LEVEL=INFO                # Logging level
export OPENAPI_CACHE_DIR=/tmp/cache  # Cache directory
export OPENAPI_MAX_WORKERS=4         # Parallel processing
```

## Advanced Features

### Performance Monitoring
```bash
# Enable detailed performance metrics
openapi-doc-generator --app ./app.py --performance-metrics --log-format json

# Monitor memory usage
openapi-doc-generator --app ./app.py --performance-metrics | grep memory

# Cache analysis
openapi-doc-generator --app ./app.py --performance-metrics | grep cache
```

### Custom Plugins
Create custom framework support:
```python
# my_plugin.py
def discover_routes(app_path):
    # Your route discovery logic
    return routes

# pyproject.toml
[project.entry-points."openapi_doc_generator.plugins"]
my_framework = "my_plugin:discover_routes"
```

### Batch Processing
```bash
# Process multiple applications
for app in apps/*.py; do
    openapi-doc-generator --app "$app" --format openapi --output "docs/$(basename $app .py).json"
done
```

## Docker Usage

### Basic Usage
```bash
docker run --rm -v $(pwd):/workspace \
    ghcr.io/danieleschmidt/openapi-doc-generator:latest \
    /workspace/app.py --format openapi --output /workspace/openapi.json
```

### Development Environment
```bash
# Using docker-compose
docker-compose --profile dev run openapi-doc-generator \
    /workspace/app.py --format markdown --output /workspace/API.md

# With custom configuration
docker run --rm -v $(pwd):/workspace \
    -v $(pwd)/config.yaml:/config.yaml \
    ghcr.io/danieleschmidt/openapi-doc-generator:latest \
    /workspace/app.py --config /config.yaml
```

### CI/CD Integration
```yaml
# GitHub Actions example
- name: Generate API Documentation
  run: |
    docker run --rm -v ${{ github.workspace }}:/workspace \
      ghcr.io/danieleschmidt/openapi-doc-generator:latest \
      /workspace/app.py --format openapi --output /workspace/openapi.json
```

## Best Practices

### Code Documentation
- Use clear, descriptive docstrings
- Include parameter types and descriptions
- Provide example request/response data
- Document error conditions

### API Design
- Follow RESTful conventions
- Use meaningful HTTP status codes
- Implement consistent error responses
- Version your APIs appropriately

### Documentation Maintenance
- Integrate generation into CI/CD pipeline
- Keep documentation close to code
- Review generated docs for accuracy
- Update examples with real data

## Troubleshooting

### Common Issues

#### Framework Not Detected
- Verify framework is properly imported
- Check supported framework versions
- Ensure application file is valid Python/JavaScript

#### Missing Routes
- Check route definition syntax
- Verify handler functions are properly decorated
- Review framework-specific plugin documentation

#### Performance Issues
- Use `--performance-metrics` for analysis
- Check available system memory
- Consider breaking large applications into modules

#### Output Formatting Issues
- Verify output path is writable
- Check file permissions
- Ensure sufficient disk space

### Error Codes
| Code   | Description | Solution |
|--------|-------------|----------|
| CLI001 | App file not found | Check file path and permissions |
| CLI002 | Old spec file missing | Verify old-spec path for migration |
| CLI003 | Old spec file invalid | Check OpenAPI spec format |
| CLI004 | Invalid output path | Ensure directory exists and is writable |
| CLI005 | Invalid tests path | Check test file path and format |

### Getting Help
- Check the [troubleshooting documentation](../runbooks/TROUBLESHOOTING.md)
- Review [GitHub issues](https://github.com/danieleschmidt/openapi-doc-generator/issues)
- Join community discussions
- Submit detailed bug reports with reproduction steps