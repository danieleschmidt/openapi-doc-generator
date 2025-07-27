# Security Guidelines for Contributors

## Overview

This document provides security guidelines for contributors to the OpenAPI Doc Generator project. Following these guidelines helps ensure the security and integrity of the codebase and user data.

## Secure Coding Practices

### Input Validation

Always validate and sanitize user inputs:

```python
# Good: Validate file paths
def validate_file_path(file_path: str) -> str:
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {file_path}")
    return str(path)

# Bad: Using user input directly
def analyze_file(file_path):
    with open(file_path) as f:  # Potential path traversal
        return f.read()
```

### Error Handling

Avoid exposing sensitive information in error messages:

```python
# Good: Generic error messages
try:
    process_sensitive_data()
except DatabaseError:
    logger.error("Database operation failed", exc_info=True)
    raise ProcessingError("Unable to process request")

# Bad: Exposing internal details
except DatabaseError as e:
    raise Exception(f"Database error: {e.connection_string}")
```

### Secrets Management

Never commit secrets or credentials:

```python
# Good: Use environment variables
api_key = os.getenv("API_KEY")
if not api_key:
    raise ConfigurationError("API_KEY environment variable required")

# Bad: Hardcoded secrets
api_key = "sk-1234567890abcdef"  # Never do this!
```

## Dependency Security

### Dependency Management

- Use pinned versions in production
- Regularly update dependencies
- Monitor security advisories

```toml
# pyproject.toml - Use minimum required versions
dependencies = [
    "jinja2>=3.1.0",  # Minimum version with security fixes
    "graphql-core>=3.2.0"
]

# For development, pin exact versions
dev-dependencies = [
    "pytest==7.4.3",
    "bandit==1.7.5"
]
```

### Security Scanning

Run security scans regularly:

```bash
# Check for known vulnerabilities
safety check

# Audit dependencies
pip-audit

# Static code analysis
bandit -r src/
```

## Container Security

### Dockerfile Best Practices

```dockerfile
# Use specific base image versions
FROM python:3.12.1-slim

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install security updates
RUN apt-get update && apt-get upgrade -y && \
    rm -rf /var/lib/apt/lists/*

# Use non-root user
USER appuser

# Set secure defaults
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
```

### Runtime Security

```bash
# Run with security options
docker run \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --user 1000:1000 \
  openapi-doc-generator
```

## Code Review Security Checklist

### For Reviewers

- [ ] No hardcoded secrets or credentials
- [ ] Input validation is present and correct
- [ ] Error handling doesn't expose sensitive information
- [ ] Dependencies are from trusted sources
- [ ] No obvious security vulnerabilities
- [ ] Logging doesn't include sensitive data
- [ ] File operations use safe paths
- [ ] Network requests use HTTPS where applicable

### For Contributors

Before submitting a PR:

- [ ] Run security scans locally
- [ ] Update dependencies if needed
- [ ] Add appropriate tests for security-sensitive code
- [ ] Document any security considerations
- [ ] Follow the principle of least privilege

## Testing Security

### Security Test Examples

```python
def test_path_traversal_protection():
    """Test protection against path traversal attacks."""
    with pytest.raises(ValueError):
        analyzer.analyze_app("../../../etc/passwd")

def test_no_code_execution():
    """Ensure malicious code is not executed during analysis."""
    malicious_code = '''
import os
os.system("rm -rf /")
    '''
    # Should not execute the malicious code
    result = analyzer.analyze_code(malicious_code)
    assert result is not None

def test_secrets_not_logged():
    """Ensure secrets don't appear in logs."""
    with capture_logs() as logs:
        process_config({"api_key": "secret123"})
    
    for log_entry in logs:
        assert "secret123" not in log_entry
```

## Incident Response

### If You Discover a Vulnerability

1. **Do not commit the discovery to public repos**
2. **Report privately** to security@terragonlabs.com
3. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if known)

### If You Accidentally Commit Secrets

1. **Immediately rotate the exposed credentials**
2. **Remove from git history**:
   ```bash
   git filter-branch --force --index-filter \
   'git rm --cached --ignore-unmatch path/to/file' \
   --prune-empty --tag-name-filter cat -- --all
   ```
3. **Report the incident**
4. **Update security practices**

## Security Tools Integration

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ['-c', 'pyproject.toml']
```

### CI/CD Security

```yaml
# .github/workflows/security.yml
- name: Security Scan
  run: |
    bandit -r src/ -f sarif -o bandit-report.sarif
    safety check --json --output safety-report.json
    pip-audit --format json --output audit-report.json
```

## Security Configuration

### Recommended Settings

```python
# security_config.py
SECURITY_SETTINGS = {
    "max_file_size_mb": 10,
    "timeout_seconds": 30,
    "allowed_extensions": [".py", ".js", ".ts"],
    "sanitize_output": True,
    "validate_paths": True,
    "enable_logging": True,
    "log_sensitive_data": False
}
```

### Environment Variables

```bash
# Production security settings
export SECURITY_STRICT_MODE=true
export LOG_SENSITIVE_DATA=false
export VALIDATE_ALL_INPUTS=true
export ENABLE_RATE_LIMITING=true
```

## Common Security Pitfalls

### Avoid These Patterns

1. **Dynamic Code Execution**
   ```python
   # Never do this
   eval(user_input)
   exec(code_from_file)
   ```

2. **Unsafe File Operations**
   ```python
   # Dangerous
   open(user_provided_path)
   subprocess.run(user_command, shell=True)
   ```

3. **Information Disclosure**
   ```python
   # Don't expose internals
   return {"error": str(exception), "traceback": traceback.format_exc()}
   ```

4. **Insufficient Validation**
   ```python
   # Always validate
   if not isinstance(input_data, expected_type):
       raise ValueError("Invalid input type")
   ```

## Resources

### Security Tools

- [Bandit](https://bandit.readthedocs.io/) - Python security linter
- [Safety](https://pyup.io/safety/) - Dependency vulnerability scanner
- [pip-audit](https://pip-audit.readthedocs.io/) - Python package auditor
- [detect-secrets](https://github.com/Yelp/detect-secrets) - Secret scanner

### Learning Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Guidelines](https://python.org/dev/security/)
- [Container Security Best Practices](https://kubernetes.io/docs/concepts/security/)

### Standards and Frameworks

- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [ISO 27001](https://www.iso.org/isoiec-27001-information-security.html)
- [CWE Top 25](https://cwe.mitre.org/top25/)

---

Remember: Security is everyone's responsibility. When in doubt, ask for a security review.