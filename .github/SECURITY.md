# Security Policy

## Supported Versions

We actively support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.2.x   | :white_check_mark: |
| 1.1.x   | :white_check_mark: |
| 1.0.x   | :x:                |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability, please follow these steps:

### 1. Do NOT open a public issue

Please do not report security vulnerabilities through public GitHub issues.

### 2. Send a private report

Send an email to: **security@terragonlabs.com**

Include the following information:
- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Suggested fix (if available)

### 3. Response Timeline

- **Initial Response**: Within 24 hours
- **Assessment**: Within 72 hours
- **Fix Timeline**: Critical issues within 7 days, others within 30 days

### 4. Disclosure Policy

- We will acknowledge receipt of your vulnerability report
- We will confirm the vulnerability and determine its impact
- We will work on a fix and prepare a security advisory
- We will notify you when the fix is released
- We will publicly disclose the vulnerability after users have had time to update

## Security Measures

### Code Security

- **Static Analysis**: All code is scanned with Bandit and CodeQL
- **Dependency Scanning**: Regular security audits with Safety and pip-audit
- **Container Scanning**: Docker images scanned with Trivy
- **Secret Scanning**: Automated detection of exposed secrets

### Input Validation

- All file paths are validated and sanitized
- No arbitrary code execution during analysis
- Input size limits to prevent DoS attacks
- Malformed syntax handling

### Runtime Security

- Non-root container execution
- Read-only filesystem where possible
- Minimal container attack surface
- Resource limits and timeouts

### Development Security

- Pre-commit hooks for security scanning
- Secure coding guidelines
- Regular security training
- Code review requirements

## Security Best Practices for Users

### Container Security

```bash
# Run with read-only filesystem
docker run --read-only -v /tmp --tmpfs /tmp:noexec,nosuid,size=100m openapi-doc-generator

# Drop all capabilities
docker run --cap-drop=ALL --security-opt=no-new-privileges openapi-doc-generator

# Use specific user
docker run --user 1000:1000 openapi-doc-generator
```

### File System Security

```bash
# Limit file access
docker run -v /path/to/source:/workspace:ro openapi-doc-generator

# Use temporary output directory
docker run -v /tmp/output:/output openapi-doc-generator
```

### Network Security

```bash
# Disable network access if not needed
docker run --network none openapi-doc-generator

# Use custom bridge network
docker network create --driver bridge secure-network
docker run --network secure-network openapi-doc-generator
```

## Known Security Considerations

### AST Parsing

While we use Python's AST module to parse source code safely, users should be aware:

- Large files may consume significant memory
- Complex nested structures could cause performance issues
- Malformed Python syntax will raise exceptions

### File System Access

The tool requires read access to source files:

- Only provide access to necessary directories
- Use read-only mounts when possible
- Avoid running on untrusted code without sandboxing

### Container Environment

When using Docker:

- Always use the latest version
- Regularly update base images
- Monitor for security advisories

## Security Updates

### Automatic Updates

- Dependabot automatically creates PRs for security updates
- CI/CD pipeline includes security scanning
- Critical security updates are fast-tracked

### Manual Updates

Check for updates regularly:

```bash
# Check for package updates
pip install --upgrade openapi-doc-generator

# Check container updates
docker pull ghcr.io/danieleschmidt/openapi-doc-generator:latest
```

### Security Notifications

- Subscribe to repository releases for security updates
- Follow security advisories on GitHub
- Monitor CVE databases for dependency vulnerabilities

## Compliance

### Standards

- OWASP Top 10 considerations
- NIST Cybersecurity Framework alignment
- ISO 27001 security principles

### Auditing

- All security events are logged
- Metrics collection for security monitoring
- Regular security assessments

### Data Protection

- No sensitive data is stored or transmitted
- Generated documentation is sanitized
- Temporary files are securely cleaned up

## Security Tools

### Integrated Security Scanning

```yaml
# Example GitHub Actions security workflow
- name: Security Scan
  run: |
    bandit -r src/
    safety check
    pip-audit
```

### Container Security

```bash
# Scan container for vulnerabilities
trivy image openapi-doc-generator:latest
```

### Local Development

```bash
# Install security tools
pip install bandit safety pip-audit

# Run security checks
make security
```

## Emergency Response

In case of a security incident:

1. **Immediate**: Stop using affected versions
2. **Assessment**: Evaluate impact on your systems  
3. **Mitigation**: Apply workarounds if available
4. **Update**: Install security patches as soon as available
5. **Verification**: Confirm fix resolves the issue

## Contact

For security-related questions or concerns:

- **Email**: security@terragonlabs.com
- **GitHub**: Open a private security advisory
- **Documentation**: https://docs.terragonlabs.com/security

---

**Last Updated**: January 2025

We appreciate your help in keeping OpenAPI Doc Generator secure!