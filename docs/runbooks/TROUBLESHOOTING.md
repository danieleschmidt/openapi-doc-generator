# Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### Issue: Package Installation Fails

**Symptoms:**
```
ERROR: Could not build wheels for openapi-doc-generator
```

**Solutions:**

1. **Update pip and build tools:**
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -e .[dev]
   ```

2. **Check Python version:**
   ```bash
   python --version  # Should be 3.8+
   ```

3. **Install system dependencies (Linux/macOS):**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install python3-dev build-essential
   
   # macOS
   xcode-select --install
   ```

4. **Use virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e .[dev]
   ```

#### Issue: Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'openapi_doc_generator'
```

**Solutions:**

1. **Install in development mode:**
   ```bash
   pip install -e .
   ```

2. **Check PYTHONPATH:**
   ```bash
   export PYTHONPATH="/path/to/project/src:$PYTHONPATH"
   ```

3. **Verify installation:**
   ```bash
   pip list | grep openapi-doc-generator
   python -c "import openapi_doc_generator; print('OK')"
   ```

### Runtime Issues

#### Issue: File Not Found Errors

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'app.py'
```

**Solutions:**

1. **Use absolute paths:**
   ```bash
   openapi-doc-generator --app /full/path/to/app.py
   ```

2. **Check current directory:**
   ```bash
   ls -la app.py  # Verify file exists
   pwd           # Check current directory
   ```

3. **Use Docker with volume mounting:**
   ```bash
   docker run -v $(pwd):/workspace openapi-doc-generator /workspace/app.py
   ```

#### Issue: Permission Denied

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied: '/output/docs.md'
```

**Solutions:**

1. **Check file permissions:**
   ```bash
   ls -la /output/
   chmod 755 /output/
   ```

2. **Run with correct user (Docker):**
   ```bash
   docker run --user $(id -u):$(id -g) openapi-doc-generator
   ```

3. **Use writable output directory:**
   ```bash
   mkdir -p ./output
   openapi-doc-generator --app app.py --output ./output/docs.md
   ```

### Framework Detection Issues

#### Issue: Framework Not Detected

**Symptoms:**
```
WARNING: No supported framework detected in app.py
```

**Solutions:**

1. **Verify framework imports:**
   ```python
   # Ensure proper imports are present
   from flask import Flask  # For Flask
   from fastapi import FastAPI  # For FastAPI
   ```

2. **Check app structure:**
   ```python
   # Flask example
   from flask import Flask
   app = Flask(__name__)  # Variable must be named 'app'
   
   # FastAPI example  
   from fastapi import FastAPI
   app = FastAPI()  # Variable must be named 'app'
   ```

3. **Use explicit framework specification:**
   ```bash
   openapi-doc-generator --app app.py --framework flask
   ```

#### Issue: Incomplete Route Discovery

**Symptoms:**
```
INFO: Found 2 routes (expected more)
```

**Solutions:**

1. **Check route definitions:**
   ```python
   # Ensure routes are properly decorated
   @app.route("/api/users", methods=["GET", "POST"])
   def users():
       pass
   ```

2. **Verify app structure:**
   ```python
   # Make sure routes are defined before app.run()
   app = Flask(__name__)
   
   @app.route("/test")
   def test():
       return "test"
   
   if __name__ == "__main__":
       app.run()  # Routes must be defined before this
   ```

3. **Enable debug logging:**
   ```bash
   LOG_LEVEL=DEBUG openapi-doc-generator --app app.py --verbose
   ```

### Performance Issues

#### Issue: Slow Documentation Generation

**Symptoms:**
- Long processing times (>30 seconds for small apps)
- High memory usage

**Solutions:**

1. **Enable performance monitoring:**
   ```bash
   openapi-doc-generator --app app.py --performance-metrics
   ```

2. **Check file size:**
   ```bash
   ls -lh app.py  # Large files (>1MB) may be slow
   ```

3. **Optimize application structure:**
   ```python
   # Avoid complex nested imports
   # Keep route definitions simple
   # Remove unused imports
   ```

4. **Use caching:**
   ```bash
   # Subsequent runs should be faster due to AST caching
   openapi-doc-generator --app app.py  # First run
   openapi-doc-generator --app app.py  # Second run (faster)
   ```

#### Issue: Memory Errors

**Symptoms:**
```
MemoryError: Unable to allocate memory
```

**Solutions:**

1. **Increase memory limits (Docker):**
   ```bash
   docker run --memory=2g openapi-doc-generator
   ```

2. **Process smaller files:**
   ```bash
   # Split large applications into smaller modules
   # Process modules individually
   ```

3. **Monitor memory usage:**
   ```python
   import psutil
   process = psutil.Process()
   print(f"Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")
   ```

### Output Issues

#### Issue: Empty or Invalid Output

**Symptoms:**
- Empty OpenAPI specification
- Malformed JSON output
- Missing route information

**Solutions:**

1. **Verify input application:**
   ```bash
   python app.py  # Test that app runs correctly
   ```

2. **Check output format:**
   ```bash
   # Validate JSON output
   openapi-doc-generator --app app.py --format openapi --output spec.json
   python -m json.tool spec.json
   ```

3. **Use validation:**
   ```bash
   openapi-doc-generator --app app.py --validate
   ```

4. **Enable verbose logging:**
   ```bash
   openapi-doc-generator --app app.py --verbose
   ```

#### Issue: Missing Documentation

**Symptoms:**
- Routes found but no descriptions
- Empty endpoint documentation

**Solutions:**

1. **Add docstrings:**
   ```python
   @app.route("/api/users")
   def get_users():
       """Get all users from the system.
       
       Returns:
           List of user objects
       """
       return users
   ```

2. **Use type hints:**
   ```python
   from typing import List
   
   @app.route("/api/users")
   def get_users() -> List[dict]:
       """Get all users."""
       return users
   ```

3. **Check framework-specific features:**
   ```python
   # FastAPI automatic documentation
   from fastapi import FastAPI
   from pydantic import BaseModel
   
   class User(BaseModel):
       name: str
       email: str
   
   @app.get("/users", response_model=List[User])
   def get_users():
       return users
   ```

### Docker Issues

#### Issue: Container Startup Failures

**Symptoms:**
```
docker: Error response from daemon: container startup failed
```

**Solutions:**

1. **Check Docker installation:**
   ```bash
   docker --version
   docker run hello-world
   ```

2. **Verify image:**
   ```bash
   docker pull ghcr.io/danieleschmidt/openapi-doc-generator:latest
   docker images | grep openapi-doc-generator
   ```

3. **Check volume mounts:**
   ```bash
   # Ensure source directory exists
   ls -la $(pwd)
   docker run -v $(pwd):/workspace openapi-doc-generator
   ```

4. **Review container logs:**
   ```bash
   docker logs container_name
   ```

#### Issue: Permission Issues in Container

**Symptoms:**
```
Permission denied writing output files
```

**Solutions:**

1. **Use correct user mapping:**
   ```bash
   docker run --user $(id -u):$(id -g) \
     -v $(pwd):/workspace \
     openapi-doc-generator
   ```

2. **Fix output directory permissions:**
   ```bash
   mkdir -p output
   chmod 777 output
   ```

3. **Use rootless containers:**
   ```bash
   podman run -v $(pwd):/workspace openapi-doc-generator
   ```

### CI/CD Issues

#### Issue: Tests Failing in CI

**Symptoms:**
- Tests pass locally but fail in CI
- Timeout errors in CI

**Solutions:**

1. **Check CI environment:**
   ```yaml
   # .github/workflows/ci.yml
   - name: Debug environment
     run: |
       python --version
       pip list
       env | sort
   ```

2. **Increase timeouts:**
   ```yaml
   - name: Run tests
     run: pytest tests/ --timeout=300
     timeout-minutes: 10
   ```

3. **Use matrix testing:**
   ```yaml
   strategy:
     matrix:
       python-version: [3.8, 3.9, "3.10", "3.11", "3.12"]
   ```

4. **Cache dependencies:**
   ```yaml
   - uses: actions/cache@v3
     with:
       path: ~/.cache/pip
       key: ${{ runner.os }}-pip-${{ hashFiles('**/pyproject.toml') }}
   ```

### Development Issues

#### Issue: Pre-commit Hooks Failing

**Symptoms:**
```
Ruff....................................................................Failed
```

**Solutions:**

1. **Install pre-commit properly:**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

2. **Run hooks manually:**
   ```bash
   pre-commit run --all-files
   ```

3. **Fix formatting issues:**
   ```bash
   ruff format .
   ruff check --fix .
   ```

4. **Skip hooks temporarily:**
   ```bash
   git commit --no-verify -m "temp commit"
   ```

#### Issue: IDE/Editor Issues

**Symptoms:**
- Import errors in IDE
- Type checking not working
- Debugging not working

**Solutions:**

1. **Configure Python interpreter:**
   - VS Code: `Ctrl+Shift+P` → "Python: Select Interpreter"
   - PyCharm: Settings → Project → Python Interpreter

2. **Install IDE extensions:**
   ```bash
   # VS Code extensions
   code --install-extension ms-python.python
   code --install-extension charliermarsh.ruff
   ```

3. **Configure workspace settings:**
   ```json
   // .vscode/settings.json
   {
     "python.defaultInterpreterPath": "./venv/bin/python",
     "python.analysis.extraPaths": ["./src"]
   }
   ```

## Diagnostic Commands

### System Information

```bash
# Check Python environment
python --version
pip --version
pip list | grep openapi

# Check system resources
free -h        # Memory
df -h          # Disk space
ps aux | head  # Running processes
```

### Application Diagnostics

```bash
# Test basic functionality
openapi-doc-generator --version
openapi-doc-generator --help

# Run with debug logging
LOG_LEVEL=DEBUG openapi-doc-generator --app app.py --verbose

# Performance monitoring
openapi-doc-generator --app app.py --performance-metrics

# Health check (if server running)
curl http://localhost:8080/health
curl http://localhost:8080/metrics
```

### Container Diagnostics

```bash
# Inspect container
docker inspect openapi-doc-generator:latest

# Run interactive shell
docker run -it --entrypoint /bin/bash openapi-doc-generator:latest

# Check container health
docker ps --filter "name=openapi-doc-generator"
docker stats openapi-doc-generator
```

## Getting Help

### Log Collection

When reporting issues, include:

1. **System information:**
   ```bash
   uname -a
   python --version
   pip list > requirements.txt
   ```

2. **Error logs:**
   ```bash
   LOG_LEVEL=DEBUG openapi-doc-generator --app app.py 2>&1 | tee debug.log
   ```

3. **Minimal reproduction case:**
   ```python
   # Simplest possible app.py that reproduces the issue
   from flask import Flask
   app = Flask(__name__)
   
   @app.route("/test")
   def test():
       return "test"
   ```

### Support Channels

- **Issues**: [GitHub Issues](https://github.com/danieleschmidt/openapi-doc-generator/issues)
- **Discussions**: [GitHub Discussions](https://github.com/danieleschmidt/openapi-doc-generator/discussions)
- **Security**: security@terragonlabs.com
- **Documentation**: [Project Documentation](../README.md)

### Issue Template

When creating an issue, include:

```markdown
## Environment
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.9.0]
- Package version: [e.g., 1.2.0]
- Installation method: [pip/docker/source]

## Expected Behavior
[What you expected to happen]

## Actual Behavior
[What actually happened]

## Steps to Reproduce
1. [First step]
2. [Second step]
3. [And so on...]

## Error Messages
```
[Full error message/traceback]
```

## Additional Context
[Any other relevant information]
```

---

If you can't find a solution here, don't hesitate to ask for help!