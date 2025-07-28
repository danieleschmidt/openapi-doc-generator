# Development Setup

Quick setup guide for OpenAPI-Doc-Generator development.

## Prerequisites

- Python 3.8+ ([Download](https://www.python.org/downloads/))
- Git ([Download](https://git-scm.com/downloads))
- pip (comes with Python)

## Quick Start

```bash
# Clone and setup
git clone https://github.com/danieleschmidt/openapi-doc-generator.git
cd openapi-doc-generator
pip install -e .[dev]

# Verify installation
pytest tests/test_foundational.py -v
openapi-doc-generator --version
```

## Essential Commands

- **Test**: `pytest`
- **Lint**: `ruff check .`
- **Format**: `ruff format .`
- **Coverage**: `pytest --cov=src`

## Resources

- [Contributing Guide](../CONTRIBUTING.md)
- [Architecture Overview](ARCHITECTURE.md)
- [Python Packaging Guide](https://packaging.python.org/)
- [pytest Documentation](https://docs.pytest.org/)

For detailed development workflow, see [Contributing Guide](../CONTRIBUTING.md).