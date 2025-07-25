# OpenAPI-Doc-Generator

![CI](https://github.com/danieleschmidt/openapi-doc-generator/actions/workflows/ci.yml/badge.svg)

Advanced tool that automatically discovers routes from popular Python and JavaScript web frameworks and generates comprehensive API documentation including OpenAPI 3.0 specs, markdown docs, and interactive playgrounds.

## Features
- Automatic route discovery and analysis for FastAPI, Express, Flask, Django, and Tornado
- Intelligent schema inference from code annotations and examples for dataclasses and Pydantic models
- Generates comprehensive OpenAPI 3.0 specifications
- Creates human-readable markdown documentation with examples
- Interactive API playground generation with Swagger UI
- Validates existing OpenAPI specs and suggests improvements
- Supports GraphQL schema introspection
- Automated test suite generation for discovered routes
- Continuous documentation deployment via GitHub Pages
- Generates API deprecation and migration guides
- Customizable API title and version via CLI options
- Plugin interface for additional frameworks
- **Performance metrics and monitoring** with detailed timing and memory usage tracking

## Quick Start

### Using Python Package
```bash
pip install -e .
openapi-doc-generator --app ./app.py --format markdown --output API.md
openapi-doc-generator --app ./app.py --format openapi --output openapi.json
openapi-doc-generator --app ./app.py --format openapi --title "My API" --api-version 2.0 --output openapi.json
openapi-doc-generator --app ./app.py --format html --output playground.html
openapi-doc-generator --app ./schema.graphql --format graphql --output schema.json
openapi-doc-generator --app ./app.py --tests tests/test_app.py
openapi-doc-generator --app ./app.py --format guide --old-spec old.json --output MIGRATION.md

# CLI options for better user experience
openapi-doc-generator --app ./app.py --verbose --format openapi  # Detailed progress output
openapi-doc-generator --app ./app.py --quiet --format markdown   # Minimal output
openapi-doc-generator --app ./app.py --no-color --format openapi # Disable colored output
openapi-doc-generator --app ./app.py --performance-metrics --log-format json  # Enable performance tracking

openapi-doc-generator --version
```

### Using Docker
```bash
# Pull the latest image
docker pull ghcr.io/danieleschmidt/openapi-doc-generator:latest

# Generate documentation for your app
docker run --rm -v $(pwd):/workspace ghcr.io/danieleschmidt/openapi-doc-generator:latest \
  /workspace/app.py --format markdown --output /workspace/API.md

# Generate OpenAPI spec with JSON logging
docker run --rm -v $(pwd):/workspace ghcr.io/danieleschmidt/openapi-doc-generator:latest \
  /workspace/app.py --format openapi --log-format json --output /workspace/openapi.json

# Use docker-compose for development
docker-compose --profile dev run openapi-doc-generator /workspace/app.py --help
```

Documentation for the example app in `examples/app.py` is automatically built
and published to GitHub Pages whenever changes are pushed to `main`.

## Docker Usage

### Pre-built Images
Pre-built Docker images are available on GitHub Container Registry:

```bash
# Latest stable release
docker pull ghcr.io/danieleschmidt/openapi-doc-generator:latest

# Specific version
docker pull ghcr.io/danieleschmidt/openapi-doc-generator:v0.1.0

# Latest from main branch
docker pull ghcr.io/danieleschmidt/openapi-doc-generator:main
```

### Building Locally
```bash
# Build the image
docker build -t openapi-doc-generator .

# Or use docker-compose
docker-compose build
```

### Docker Compose Development
```bash
# Start development environment
docker-compose --profile dev up

# Run specific commands
docker-compose --profile dev run openapi-doc-generator /workspace/app.py --format openapi --output /workspace/openapi.json

# Production-like testing
docker-compose --profile prod up
```

### Image Security
- Runs as non-root user (UID 1000)
- Multi-stage build for minimal image size
- Security scanned with Trivy in CI/CD
- Includes health check endpoint

## Development
Install pre-commit hooks for local secret scanning:
```bash
pre-commit install
```

### Plugins
Third-party route discovery plugins can be installed via the
`openapi_doc_generator.plugins` entry point. See [EXTENDING.md](EXTENDING.md)
for details.

## Logging
The CLI emits informational logs to stderr. Set the `LOG_LEVEL` environment
variable to `DEBUG` for verbose output during troubleshooting.

## CLI Error Codes
`openapi-doc-generator` exits with standardized codes when input validation fails:

| Code   | Meaning                     |
|-------|-----------------------------|
| CLI001 | App file not found          |
| CLI002 | Old spec file missing       |
| CLI003 | Old spec file is invalid    |
| CLI004 | `--output` path is invalid  |
| CLI005 | `--tests` path is invalid   |

## Testing
Run the test suite with:
```bash
coverage run -m pytest -q
coverage html  # view HTML report in htmlcov/index.html
```

## Usage
```python
from openapi_doc_generator import APIDocumentator

generator = APIDocumentator()
docs = generator.analyze_app("./app.py")

# Generate OpenAPI spec
spec = docs.generate_openapi_spec()
with open("openapi.json", "w") as f:
    json.dump(spec, f, indent=2)

# Validate OpenAPI spec
from openapi_doc_generator import SpecValidator
issues = SpecValidator().validate(spec)
if issues:
    print("Suggestions:\n" + "\n".join(issues))

# Generate markdown docs
markdown = docs.generate_markdown()
with open("API.md", "w") as f:
    f.write(markdown)
```

## Framework Examples

### Flask Application
```python
from flask import Flask

app = Flask(__name__)

@app.route("/api/users", methods=["GET", "POST"])
def users():
    """User management endpoint."""
    return {"users": []}

@app.route("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}
```

### Tornado Application
```python
import tornado.web

class MainHandler(tornado.web.RequestHandler):
    """Main page handler."""
    def get(self):
        """Handle GET requests."""
        self.write("Hello, world")

class UserHandler(tornado.web.RequestHandler):
    """User management handler."""
    def get(self, user_id):
        """Get user information."""
        self.write(f"User {user_id}")

application = tornado.web.Application([
    (r"/", MainHandler),
    (r"/user/([^/]+)", UserHandler),
])
```

### Django Application
```python
from django.urls import path
from . import views

urlpatterns = [
    path('api/users/', views.users, name='users'),
    path('api/users/<int:user_id>/', views.user_detail, name='user_detail'),
    path('health/', views.health, name='health'),
]
```

Generate documentation for any of these:
```bash
openapi-doc-generator --app ./your_app.py --format markdown --output API.md
```

## Framework Support
- **FastAPI**: Full type annotation support, automatic model extraction
- **Express.js**: Route parsing, JSDoc integration, TypeScript support
- **Flask**: Decorator analysis, Flask-RESTful integration
- **Django REST Framework**: Serializer introspection, viewset analysis
- **Tornado**: RequestHandler analysis, Application routing patterns

## Generated Documentation
- Complete endpoint documentation with parameters, responses, examples
- Interactive API explorer (Swagger UI integration)
- Code samples in multiple languages (Python, JavaScript, cURL)
- Authentication and error handling documentation
- Rate limiting and versioning information

## Performance Monitoring
The tool includes comprehensive performance metrics to help optimize documentation generation:

```bash
# Enable performance metrics with JSON logging
openapi-doc-generator --app ./app.py --performance-metrics --log-format json

# Sample performance output
{"timestamp":"2025-07-22T00:56:55Z","level":"INFO","logger":"openapi_doc_generator.discovery","message":"Performance: framework_detection completed in 0.57ms","correlation_id":"d04bfed0","duration_ms":0.57}
{"timestamp":"2025-07-22T00:56:55Z","level":"INFO","logger":"openapi_doc_generator.discovery","message":"Performance: route_discovery completed in 1.52ms","correlation_id":"d04bfed0","duration_ms":1.52,"route_count":5}
```

**Tracked Metrics:**
- Route discovery timing and memory usage
- Framework detection performance
- AST parsing cache hit rates
- Memory allocation patterns
- Operation-level performance summaries

**Integration Benefits:**
- Identify performance bottlenecks in large codebases
- Monitor memory usage for optimization
- Track performance improvements over time
- Debug slow documentation generation

## Configuration
```yaml
# config.yaml
frameworks:
  fastapi:
    include_internal: false
    example_generation: true
  express:
    typescript_support: true
    middleware_docs: true

output:
  formats: ["openapi", "markdown", "postman"]
  include_examples: true
  authentication_docs: true
```

## Advanced Features
- **Reflection-based Analysis**: Uses LLM to understand complex business logic
- **Example Generation**: Creates realistic API examples based on schema
- **Version Comparison**: Tracks API changes across versions
- **Integration Testing**: Validates generated docs against actual API responses
- **Automated Test Generation**: Produces pytest suites for discovered routes

## Roadmap
All roadmap items have been completed:
1. ✅ Add GraphQL schema support
2. ✅ Implement automated testing suite generation
3. ✅ Build CI/CD integration for documentation updates
4. ✅ Add API deprecation and migration guides

## License
MIT

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and development guidelines.

## Extending
See [EXTENDING.md](EXTENDING.md) for writing custom discovery plugins.
