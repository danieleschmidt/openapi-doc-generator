# Development Guide

## Overview

This guide covers everything you need to know to contribute to the OpenAPI Doc Generator project.

## Quick Start

### Prerequisites

- Python 3.8+
- Git
- Docker (optional)
- VS Code (recommended)

### Setup Development Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/danieleschmidt/openapi-doc-generator.git
   cd openapi-doc-generator
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e .[dev]
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

5. **Verify installation**
   ```bash
   pytest tests/test_foundational.py -v
   openapi-doc-generator --version
   ```

### Alternative: Use Dev Container

If you have VS Code and Docker installed:

1. Open the project in VS Code
2. Press `Ctrl+Shift+P` and select "Dev Containers: Reopen in Container"
3. The development environment will be automatically configured

## Project Structure

```
├── src/
│   └── openapi_doc_generator/
│       ├── __init__.py
│       ├── cli.py              # Command-line interface
│       ├── config.py           # Configuration management
│       ├── discovery.py        # Route discovery engine
│       ├── documentator.py     # Main documentation generator
│       ├── github_hygiene.py   # Repository hygiene tools
│       ├── graphql.py          # GraphQL support
│       ├── health_server.py    # Health check endpoints
│       ├── logging_config.py   # Structured logging
│       ├── markdown.py         # Markdown generation
│       ├── migration.py        # API migration guides
│       ├── monitoring.py       # Metrics and observability
│       ├── playground.py       # Interactive playground
│       ├── schema.py           # Schema inference
│       ├── spec.py             # OpenAPI specification
│       ├── testsuite.py        # Test suite generation
│       ├── utils.py            # Utility functions
│       ├── validator.py        # Validation tools
│       ├── plugins/            # Framework plugins
│       │   ├── __init__.py
│       │   ├── aiohttp.py
│       │   ├── starlette.py
│       │   └── tornado.py
│       └── templates/          # Output templates
│           ├── __init__.py
│           └── api.md.jinja
├── tests/                      # Test suite
│   ├── integration/
│   ├── performance/
│   ├── security/
│   └── unit/
├── docs/                       # Documentation
├── examples/                   # Example applications
├── .github/                    # GitHub workflows
└── .devcontainer/              # Development container
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Follow the coding standards and write tests for your changes:

```bash
# Run tests while developing
pytest tests/ -v

# Run specific test file
pytest tests/test_your_feature.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### 3. Quality Checks

```bash
# Format code
ruff format .

# Check linting
ruff check .

# Type checking
mypy src/

# Security scan
bandit -r src/

# Run all checks
make ci
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat: add your feature description"
```

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Coding Standards

### Code Style

- Follow PEP 8
- Use type hints for all functions
- Write docstrings for all public functions
- Keep functions small and focused
- Use descriptive variable names

### Example Code Structure

```python
"""Module docstring describing the purpose."""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from .monitoring import performance_timer
from .logging_config import get_logger

logger = get_logger(__name__)


class ExampleClass:
    """Class docstring describing the purpose and usage.
    
    Args:
        config: Configuration dictionary
        timeout: Optional timeout in seconds
    """
    
    def __init__(self, config: Dict[str, Any], timeout: Optional[int] = None):
        self.config = config
        self.timeout = timeout or 30
        
    @performance_timer("example_operation")
    def process_data(self, data: List[str]) -> Dict[str, Any]:
        """Process input data and return results.
        
        Args:
            data: List of strings to process
            
        Returns:
            Dictionary with processing results
            
        Raises:
            ValueError: If data is empty or invalid
        """
        if not data:
            raise ValueError("Data cannot be empty")
        
        logger.info(f"Processing {len(data)} items")
        
        # Implementation here
        results = {"processed": len(data)}
        
        logger.info("Processing completed successfully")
        return results
```

### Testing Standards

```python
"""Test module for example functionality."""

import pytest
from unittest.mock import Mock, patch

from openapi_doc_generator.example import ExampleClass


class TestExampleClass:
    """Test suite for ExampleClass."""
    
    @pytest.fixture
    def example_config(self):
        """Fixture providing example configuration."""
        return {"setting1": "value1", "setting2": True}
    
    @pytest.fixture
    def example_instance(self, example_config):
        """Fixture providing ExampleClass instance."""
        return ExampleClass(example_config)
    
    def test_process_data_success(self, example_instance):
        """Test successful data processing."""
        data = ["item1", "item2", "item3"]
        result = example_instance.process_data(data)
        
        assert result["processed"] == 3
        assert isinstance(result, dict)
    
    def test_process_data_empty_input(self, example_instance):
        """Test handling of empty input."""
        with pytest.raises(ValueError, match="Data cannot be empty"):
            example_instance.process_data([])
    
    @pytest.mark.integration
    def test_integration_workflow(self, example_instance):
        """Test complete integration workflow."""
        # Integration test implementation
        pass
```

## Architecture Guidelines

### Plugin System

When adding new framework support:

1. Create a new plugin in `src/openapi_doc_generator/plugins/`
2. Implement the standard plugin interface
3. Add entry point in `pyproject.toml`
4. Write comprehensive tests
5. Update documentation

```python
# Example plugin structure
from typing import List, Dict, Any
from ..schema import Route

class NewFrameworkPlugin:
    """Plugin for New Framework support."""
    
    def discover_routes(self, file_path: str) -> List[Route]:
        """Discover routes from New Framework application.
        
        Args:
            file_path: Path to application file
            
        Returns:
            List of discovered routes
        """
        # Implementation here
        pass
```

### Error Handling

Use consistent error handling patterns:

```python
from .exceptions import (
    DocumentationError,
    FrameworkNotSupportedError,
    InvalidConfigurationError
)

def risky_operation():
    try:
        # Operation that might fail
        pass
    except SpecificException as e:
        logger.error(f"Operation failed: {e}", exc_info=True)
        raise DocumentationError(f"Unable to complete operation: {e}") from e
```

### Logging

Use structured logging with correlation IDs:

```python
from .logging_config import get_logger, set_correlation_id

logger = get_logger(__name__)

def process_request(request_id: str):
    set_correlation_id(request_id)
    
    logger.info("Starting request processing", extra={
        "request_id": request_id,
        "operation": "process_request"
    })
    
    # Processing logic
    
    logger.info("Request processing completed")
```

## Performance Guidelines

### Optimization Tips

1. **Use caching** for expensive operations
2. **Profile code** to identify bottlenecks
3. **Use generators** for large datasets
4. **Implement timeouts** for external operations
5. **Monitor memory usage** in long-running processes

### Performance Testing

```python
@pytest.mark.performance
def test_performance_requirement(performance_timer):
    """Test that operation completes within time limit."""
    performance_timer.start()
    
    # Operation to test
    result = expensive_operation()
    
    performance_timer.stop()
    
    # Should complete in under 1 second
    assert performance_timer.elapsed < 1.0
    assert result is not None
```

## Documentation Guidelines

### Code Documentation

- Write clear docstrings for all public APIs
- Include examples in docstrings
- Document complex algorithms
- Keep documentation up to date

### User Documentation

- Write user-focused documentation
- Include practical examples
- Test all code examples
- Keep documentation concise but complete

## Release Process

### Version Management

We use semantic versioning (SemVer):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Create release PR
5. Tag release after merge
6. Monitor release deployment

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure package is installed in development mode
pip install -e .[dev]

# Check PYTHONPATH
export PYTHONPATH="/path/to/project/src:$PYTHONPATH"
```

**Test Failures**
```bash
# Run specific failing test with verbose output
pytest tests/test_failing.py::test_function -v -s

# Run with debugger
pytest tests/test_failing.py::test_function --pdb
```

**Performance Issues**
```bash
# Profile code execution
python -m cProfile -o profile_output script.py

# Memory profiling
python -m memory_profiler script.py
```

### Development Tools

**Useful Commands**
```bash
# Format and check code
make format
make lint

# Run security checks
make security

# Generate documentation
make docs

# Run performance tests
make benchmark

# Clean build artifacts
make clean
```

**VS Code Extensions**

Install these recommended extensions:
- Python
- Pylance  
- Black Formatter
- Ruff
- GitHub Copilot
- Docker
- GitHub Actions

## Getting Help

### Resources

- **Documentation**: [docs/](./README.md)
- **Issues**: [GitHub Issues](https://github.com/danieleschmidt/openapi-doc-generator/issues)
- **Discussions**: [GitHub Discussions](https://github.com/danieleschmidt/openapi-doc-generator/discussions)
- **Security**: [SECURITY.md](../.github/SECURITY.md)

### Contact

- **Maintainers**: See [CODEOWNERS](../.github/CODEOWNERS)
- **Community**: Join our discussions on GitHub
- **Security**: security@terragonlabs.com

---

Happy coding! 🚀