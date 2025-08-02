# Development Guide

## Quick Start

### Prerequisites
- Python 3.8+
- Docker (optional, for containerized development)
- Git

### Setup
```bash
# Clone the repository
git clone https://github.com/danieleschmidt/openapi-doc-generator.git
cd openapi-doc-generator

# Install in development mode
pip install -e .

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest
```

## Development Workflow

### Branch Strategy
- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: Feature development branches
- `hotfix/*`: Critical fixes for production

### Making Changes
1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes with comprehensive tests
3. Run quality checks: `make lint test`
4. Commit with conventional commit messages
5. Push and create pull request

### Conventional Commits
```
feat: add new framework support
fix: resolve route discovery issue
docs: update API documentation
test: add integration test coverage
refactor: improve plugin architecture
```

## Code Standards

### Python Code Style
- Follow PEP 8 with Black formatting
- Use type hints for all public functions
- Docstrings for all modules, classes, and public functions
- Maximum line length: 88 characters (Black default)

### Testing Requirements
- Unit tests for all new functionality
- Integration tests for framework compatibility
- Performance tests for critical paths
- Security tests for input validation

### Documentation Standards
- Update README.md for user-facing changes
- Add ADRs for architectural decisions
- Include docstring examples for complex functions
- Update CHANGELOG.md for all releases

## Plugin Development

### Creating a Framework Plugin
1. Create new file in `src/openapi_doc_generator/plugins/`
2. Implement `discover_routes()` method
3. Add entry point in `pyproject.toml`
4. Create comprehensive test suite
5. Update documentation

### Plugin Interface
```python
from typing import List, Dict, Any

def discover_routes(app_path: str) -> List[Dict[str, Any]]:
    """
    Discover routes in the given application.
    
    Args:
        app_path: Path to the application file
        
    Returns:
        List of route dictionaries with keys:
        - path: URL path pattern
        - methods: HTTP methods
        - handler: Handler function name
        - docstring: Handler documentation
        - parameters: Route parameters
    """
    pass
```

## Testing

### Running Tests
```bash
# All tests
pytest

# Specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# With coverage
coverage run -m pytest
coverage html
```

### Performance Testing
```bash
# Run performance benchmarks
pytest tests/performance/ -v

# Generate performance report
python scripts/performance_analysis.py
```

### Security Testing
```bash
# Run security scans
python scripts/security_scan.py

# Check for vulnerabilities
safety check
bandit -r src/
```

## Build and Deployment

### Local Development
```bash
# Development server with hot reload
python -m openapi_doc_generator --app examples/app.py --watch

# Build documentation
make docs

# Build Docker image
docker build -t openapi-doc-generator .
```

### Release Process
1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release tag: `git tag v1.0.0`
4. Push tags: `git push --tags`
5. GitHub Actions handles automated release

## Debugging

### Common Issues
- **ImportError**: Ensure virtual environment is activated
- **Route not found**: Check framework-specific plugin implementation
- **Performance issues**: Enable performance metrics with `--performance-metrics`

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
openapi-doc-generator --app your-app.py --verbose

# JSON structured logging
openapi-doc-generator --app your-app.py --log-format json
```

### Profiling
```bash
# Python profiling
python -m cProfile -o profile.stats -m openapi_doc_generator --app your-app.py
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# Memory profiling
python -m memory_profiler -m openapi_doc_generator --app your-app.py
```

## IDE Configuration

### VS Code
Recommended extensions:
- Python
- Pylance
- Black Formatter
- GitLens
- Docker

### Settings
```json
{
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.testing.pytestEnabled": true,
    "editor.formatOnSave": true
}
```

## Troubleshooting

### Common Development Issues

#### Framework Detection Failures
- Verify framework is properly imported in application
- Check plugin compatibility with framework version
- Review framework-specific documentation

#### Performance Degradation
- Use `--performance-metrics` to identify bottlenecks
- Check AST cache hit rates
- Monitor memory usage with performance tests

#### Test Failures
- Ensure all dependencies are installed
- Check Python version compatibility
- Verify test data and fixtures are current

### Getting Help
- Check existing issues on GitHub
- Review documentation in `/docs/`
- Join community discussions
- Create detailed bug reports with reproduction steps