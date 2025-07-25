# Contributing

Thank you for helping improve **OpenAPI-Doc-Generator**! This guide will help you get started with contributing to the project.

## Development Setup

### Prerequisites
- Python 3.8+
- Git

### Quick Setup
1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/openapi-doc-generator.git
   cd openapi-doc-generator
   ```

3. Install the project in editable mode with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks (optional but recommended):
   ```bash
   pre-commit install
   ```

## Development Workflow

### Running Tests
Execute the full test suite:
```bash
# Run all tests
pytest

# Run with coverage
coverage run -m pytest
coverage html  # Open htmlcov/index.html for coverage report

# Run specific test files
pytest tests/test_tornado_plugin.py

# Run performance benchmarks
pytest tests/test_performance_benchmarks.py
```

### Code Quality
Format and lint the code before submitting changes:
```bash
# Auto-fix linting issues
ruff check --fix .

# Check code formatting
ruff format --check .

# Security scanning
bandit -r src/

# Complexity analysis
radon cc src/ -n B
```

### Testing Your Changes
Test your changes against different frameworks:
```bash
# Test with example apps
openapi-doc-generator --app examples/app.py --format markdown
openapi-doc-generator --app examples/tornado_app.py --format openapi
```

## Contributing Guidelines

### Adding New Framework Support
To add support for a new web framework:

1. **Create a Plugin**: Add a new plugin file in `src/openapi_doc_generator/plugins/`
   ```python
   from ..discovery import RouteInfo, RoutePlugin, register_plugin
   
   class MyFrameworkPlugin(RoutePlugin):
       def detect(self, source: str) -> bool:
           return "my_framework" in source.lower()
       
       def discover(self, app_path: str) -> List[RouteInfo]:
           # Implementation here
           pass
   
   register_plugin(MyFrameworkPlugin())
   ```

2. **Update Plugin Registry**: Add your plugin to `src/openapi_doc_generator/plugins/__init__.py`

3. **Add Entry Point**: Update `pyproject.toml` to register the plugin entry point

4. **Write Tests**: Create comprehensive tests in `tests/test_my_framework_plugin.py`

5. **Update Documentation**: Add your framework to the README and examples

### Bug Reports
When reporting bugs, please include:
- Python version
- Framework and version being analyzed
- Minimal code example that reproduces the issue
- Expected vs. actual behavior
- Error messages or logs

### Feature Requests
For new features:
- Check existing issues first
- Describe the use case and problem it solves
- Provide examples of how the feature would be used
- Consider implementation complexity

### Pull Request Process
1. **Create a Branch**: Use a descriptive name
   ```bash
   git checkout -b feature/add-fastapi-websocket-support
   git checkout -b fix/tornado-regex-patterns
   ```

2. **Make Changes**: Follow the coding standards
   - Write tests for new functionality
   - Update documentation as needed
   - Keep commits atomic and well-described

3. **Test Thoroughly**:
   ```bash
   pytest  # All tests should pass
   ruff check .  # No linting errors
   bandit -r src/  # No security issues
   ```

4. **Update Documentation**: 
   - Add framework examples if applicable
   - Update README.md if adding new features
   - Add docstrings to new functions/classes

5. **Submit PR**: 
   - Use a clear title and description
   - Reference any related issues
   - Include examples of the change in action

### Code Style Guidelines
- Follow PEP 8 standards (enforced by ruff)
- Use type hints for all function parameters and return types
- Write descriptive docstrings for all public functions and classes
- Keep functions focused and under 50 lines when possible
- Use meaningful variable and function names

### Performance Considerations
- Run performance benchmarks for changes that might affect speed:
  ```bash
  python tests/test_performance_benchmarks.py
  ```
- AST parsing is cached - be mindful of cache invalidation
- Consider memory usage for large applications
- Profile code changes if they affect core discovery logic

## Getting Help
- Join discussions in GitHub Issues
- Check the documentation in `docs/` directory
- Look at existing plugin implementations for examples
- Feel free to ask questions in your pull request

## Recognition
Contributors will be recognized in the project's changelog and release notes. Significant contributions may be highlighted in the README.

Thank you for contributing to OpenAPI-Doc-Generator!
