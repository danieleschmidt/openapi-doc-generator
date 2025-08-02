# Testing Patterns and Guidelines

## Overview

This document outlines testing patterns, best practices, and guidelines for the OpenAPI Doc Generator project.

## Test Organization

### Directory Structure
```
tests/
├── conftest.py                 # Shared fixtures and configuration
├── fixtures/                   # Test data and mock objects
│   ├── __init__.py
│   ├── sample_apps.py          # Sample application code
│   ├── sample_openapi_specs.py # OpenAPI specification examples
│   └── test_data.py            # Mock data and utilities
├── unit/                       # Unit tests
├── integration/                # Integration tests
├── performance/                # Performance benchmarks
└── security/                   # Security validation tests
```

### Test Categories

#### Unit Tests
- Test individual functions and classes in isolation
- Use mocks for external dependencies
- Fast execution (<1s per test)
- High coverage of edge cases

```python
def test_route_discovery_flask_basic(mock_ast_parser):
    """Test basic Flask route discovery."""
    mock_ast_parser.return_value = mock_routes
    result = discover_routes("flask_app.py")
    assert len(result) == 3
    assert result[0]["path"] == "/api/users"
```

#### Integration Tests
- Test component interactions
- Use real file system operations
- Test CLI commands end-to-end
- Validate output formats

```python
def test_cli_generate_openapi_spec(temp_dir, sample_flask_app):
    """Test CLI generates valid OpenAPI spec."""
    app_file = create_temp_app_file(sample_flask_app, temp_dir)
    output_file = temp_dir / "output.json"
    
    result = run_cli(["--app", str(app_file), "--format", "openapi", "--output", str(output_file)])
    
    assert result.returncode == 0
    assert output_file.exists()
    spec = json.loads(output_file.read_text())
    assert spec["openapi"] == "3.0.3"
```

#### Performance Tests
- Benchmark critical operations
- Memory usage monitoring
- Regression detection
- Scalability testing

```python
@pytest.mark.performance
def test_large_codebase_performance(large_flask_app):
    """Test performance with large codebase."""
    start_time = time.time()
    memory_before = get_memory_usage()
    
    result = analyze_app(large_flask_app)
    
    duration = time.time() - start_time
    memory_after = get_memory_usage()
    
    assert duration < 10.0  # Should complete in under 10 seconds
    assert memory_after - memory_before < 100 * 1024 * 1024  # Less than 100MB
```

#### Security Tests
- Input validation testing
- Dependency vulnerability scanning
- Code security analysis
- Output sanitization

```python
@pytest.mark.security
def test_malicious_input_handling():
    """Test handling of malicious input."""
    malicious_code = "import os; os.system('rm -rf /')"
    
    with pytest.raises(SecurityError):
        analyze_app_content(malicious_code)
```

## Testing Patterns

### Fixture Usage

#### Shared Fixtures (conftest.py)
```python
@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def sample_flask_app():
    """Sample Flask application for testing."""
    return FLASK_APP_BASIC
```

#### Parameterized Tests
```python
@pytest.mark.parametrize("framework,app_code,expected_routes", [
    ("flask", FLASK_APP_BASIC, 3),
    ("fastapi", FASTAPI_APP_BASIC, 3),
    ("django", DJANGO_VIEWS, 4),
])
def test_framework_route_discovery(framework, app_code, expected_routes):
    """Test route discovery across frameworks."""
    result = discover_routes(app_code, framework)
    assert len(result) == expected_routes
```

### Mocking Strategies

#### External Dependencies
```python
@patch('openapi_doc_generator.discovery.ast.parse')
def test_ast_parsing_error_handling(mock_parse):
    """Test AST parsing error handling."""
    mock_parse.side_effect = SyntaxError("Invalid syntax")
    
    with pytest.raises(ParseError):
        discover_routes("invalid_app.py")
```

#### File System Operations
```python
@patch('pathlib.Path.exists')
@patch('pathlib.Path.read_text')
def test_file_not_found_handling(mock_read, mock_exists):
    """Test file not found error handling."""
    mock_exists.return_value = False
    
    result = run_cli(["--app", "nonexistent.py"])
    assert result.returncode == 1
    assert "CLI001" in result.stderr
```

### Error Testing Patterns

#### Expected Exceptions
```python
def test_invalid_openapi_spec_validation():
    """Test validation of invalid OpenAPI spec."""
    invalid_spec = {"openapi": "3.0.3"}  # Missing required fields
    
    with pytest.raises(ValidationError) as exc_info:
        validate_openapi_spec(invalid_spec)
    
    assert "info field is required" in str(exc_info.value)
```

#### CLI Error Codes
```python
def test_cli_error_codes():
    """Test CLI returns correct error codes."""
    test_cases = [
        (["--app", "nonexistent.py"], 1, "CLI001"),
        (["--app", "app.py", "--format", "invalid"], 1, "Invalid format"),
        (["--app", "app.py", "--output", "/root/readonly"], 1, "CLI004"),
    ]
    
    for args, expected_code, expected_message in test_cases:
        result = run_cli(args)
        assert result.returncode == expected_code
        assert expected_message in result.stderr
```

## Performance Testing

### Benchmarking
```python
def test_route_discovery_benchmark():
    """Benchmark route discovery performance."""
    app_sizes = [10, 50, 100, 500]  # Number of routes
    
    for size in app_sizes:
        app_code = generate_large_app(size)
        
        start_time = time.perf_counter()
        result = discover_routes(app_code)
        duration = time.perf_counter() - start_time
        
        # Performance should scale linearly
        expected_max_time = size * 0.01  # 10ms per route
        assert duration < expected_max_time
        assert len(result) == size
```

### Memory Testing
```python
def test_memory_usage_with_large_apps():
    """Test memory usage doesn't grow excessively."""
    import psutil
    process = psutil.Process()
    
    initial_memory = process.memory_info().rss
    
    # Process multiple large applications
    for _ in range(10):
        large_app = generate_large_app(1000)
        discover_routes(large_app)
        
        current_memory = process.memory_info().rss
        memory_growth = current_memory - initial_memory
        
        # Memory growth should be reasonable
        assert memory_growth < 50 * 1024 * 1024  # Less than 50MB growth
```

## Best Practices

### Test Naming
- Use descriptive test names that explain the scenario
- Include the expected outcome in the name
- Use the pattern: `test_<component>_<scenario>_<expected_result>`

### Test Structure
1. **Arrange**: Set up test data and conditions
2. **Act**: Execute the code under test
3. **Assert**: Verify the expected outcomes

### Test Data Management
- Use fixtures for reusable test data
- Keep test data small and focused
- Use factories for generating varied test data

### Coverage Goals
- Unit tests: 95%+ line coverage
- Integration tests: Cover all major workflows
- Performance tests: Cover critical performance paths
- Security tests: Cover all input validation paths

### Continuous Integration
- All tests must pass before merging
- Performance tests run on every PR
- Security tests run on every commit
- Coverage reports generated automatically

## Test Execution

### Local Development
```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest -m performance
pytest -m security

# Run with coverage
coverage run -m pytest
coverage html

# Run performance benchmarks
pytest tests/performance/ --benchmark-only
```

### CI/CD Pipeline
```yaml
- name: Run Unit Tests
  run: pytest tests/unit/ -v --junitxml=unit-results.xml

- name: Run Integration Tests
  run: pytest tests/integration/ -v --junitxml=integration-results.xml

- name: Run Security Tests
  run: pytest tests/security/ -v --junitxml=security-results.xml

- name: Generate Coverage Report
  run: |
    coverage run -m pytest
    coverage xml
    coverage html
```

This comprehensive testing framework ensures reliability, performance, and security of the OpenAPI Doc Generator while maintaining high code quality standards.