# Container Security Guide

## Overview

This document outlines the security measures implemented in the OpenAPI Doc Generator container images and provides guidelines for secure deployment.

## Security Architecture

### Multi-Stage Build Security
- **Builder Stage**: Contains build tools and dependencies
- **Production Stage**: Minimal runtime environment only
- **Separation**: Build artifacts isolated from runtime
- **Size Reduction**: Smaller attack surface

### Base Image Security
- **Minimal Base**: `python:3.11-slim` reduces vulnerabilities
- **Regular Updates**: Base images updated with security patches
- **Scanning**: Automated vulnerability scanning with Trivy
- **Provenance**: Official Python Docker images with known provenance

## User Security

### Non-Root Execution
```dockerfile
# Create dedicated user
RUN groupadd -r openapi && useradd -r -g openapi -u 1000 openapi

# Switch to non-root user
USER openapi
```

Benefits:
- Prevents privilege escalation attacks
- Limits container breakout impact
- Complies with security best practices
- Compatible with restricted environments

### File System Permissions
```dockerfile
# Set proper ownership
COPY --chown=openapi:openapi . .

# Create directories with correct permissions
RUN mkdir -p /app/output && chown -R openapi:openapi /app
```

## Runtime Security

### Environment Variables
```dockerfile
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
```

Security implications:
- `PYTHONUNBUFFERED`: Ensures logs are visible for monitoring
- `PYTHONDONTWRITEBYTECODE`: Prevents bytecode cache files

### Read-Only Root Filesystem
The container is designed to work with read-only root filesystem:

```bash
# Run with read-only filesystem
docker run --read-only \
  --tmpfs /tmp \
  --tmpfs /app/output \
  openapi-doc-generator:latest
```

### Capability Dropping
```bash
# Run with minimal capabilities
docker run --cap-drop=ALL \
  --cap-add=SETUID \
  --cap-add=SETGID \
  openapi-doc-generator:latest
```

## Network Security

### Default Network Configuration
- No exposed ports by default
- No network services running
- Outbound connections only as needed
- DNS resolution for package repositories

### Network Isolation
```bash
# Run with no network access
docker run --network=none \
  -v $(pwd):/workspace:ro \
  openapi-doc-generator:latest /workspace/app.py
```

## Supply Chain Security

### Image Signing and Verification
```bash
# Enable Docker Content Trust
export DOCKER_CONTENT_TRUST=1

# Verify image signatures
docker trust inspect ghcr.io/danieleschmidt/openapi-doc-generator:latest
```

### Software Bill of Materials (SBOM)
```bash
# Generate SBOM
syft packages openapi-doc-generator:latest -o spdx-json

# Scan for vulnerabilities
grype openapi-doc-generator:latest
```

### Dependency Scanning
Automated scanning in CI/CD:
- **Trivy**: Container vulnerability scanning
- **Safety**: Python dependency scanning
- **Bandit**: Static code security analysis
- **pip-audit**: Package vulnerability checking

## Secrets Management

### No Secrets in Images
- No hardcoded credentials
- No API keys or tokens
- No private keys or certificates
- Runtime injection only

### Secret Injection Patterns
```bash
# Environment variables
docker run -e API_KEY="$API_KEY" openapi-doc-generator:latest

# Mounted secrets
docker run -v /path/to/secrets:/secrets:ro openapi-doc-generator:latest

# Docker secrets (Swarm mode)
echo "secret_value" | docker secret create api_key -
docker service create --secret api_key openapi-doc-generator:latest
```

## Kubernetes Security

### Security Context
```yaml
apiVersion: v1
kind: Pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000
  containers:
  - name: openapi-doc-generator
    image: openapi-doc-generator:latest
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
    resources:
      limits:
        memory: "512Mi"
        cpu: "500m"
      requests:
        memory: "256Mi"
        cpu: "250m"
```

### Pod Security Standards
- **Restricted**: Highest security, recommended for production
- **Baseline**: Minimal restrictions
- **Privileged**: No restrictions (not recommended)

### Network Policies
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: openapi-doc-generator-policy
spec:
  podSelector:
    matchLabels:
      app: openapi-doc-generator
  policyTypes:
  - Ingress
  - Egress
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS only
```

## Security Monitoring

### Health Checks
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import openapi_doc_generator; print('OK')" || exit 1
```

### Logging Configuration
```bash
# Structured logging for security monitoring
docker run \
  -e LOG_FORMAT=json \
  -e LOG_LEVEL=INFO \
  openapi-doc-generator:latest
```

### Security Events
Monitor for:
- Container startup/shutdown
- File system access attempts
- Network connection attempts
- Process execution
- Resource usage anomalies

## Vulnerability Management

### Scanning Schedule
- **Base Images**: Weekly automated scans
- **Dependencies**: Daily dependency checks
- **Code**: On every commit
- **Release Images**: Before publication

### Vulnerability Response
1. **Critical**: Immediate patch and rebuild
2. **High**: Patch within 24 hours
3. **Medium**: Patch within 7 days
4. **Low**: Patch in next regular release

### Security Updates
```bash
# Check for updates
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image openapi-doc-generator:latest

# Update dependencies
pip install --upgrade safety pip-audit
safety check
pip-audit
```

## Compliance and Standards

### CIS Docker Benchmark
- CIS-DI-0001: Create user for container ✅
- CIS-DI-0005: Enable Content trust ✅
- CIS-DI-0006: Add HEALTHCHECK ✅
- CIS-DI-0008: Remove setuid and setgid ✅
- CIS-DI-0009: Use COPY instead of ADD ✅

### NIST Framework Alignment
- **Identify**: Asset inventory and risk assessment
- **Protect**: Access controls and security measures
- **Detect**: Monitoring and alerting
- **Respond**: Incident response procedures
- **Recover**: Business continuity planning

### SLSA Compliance
- **Source**: Code provenance verification
- **Build**: Reproducible builds
- **Dependencies**: Dependency verification
- **Deployment**: Secure deployment practices

## Security Testing

### Automated Security Tests
```bash
# Container security testing
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(pwd):/workspace \
  aquasec/trivy config /workspace

# Runtime security testing
docker run --rm --cap-drop=ALL openapi-doc-generator:latest --version
```

### Penetration Testing
Regular security assessments:
- Container escape attempts
- Privilege escalation testing
- Network security validation
- Secret exposure testing

## Incident Response

### Security Incident Playbook
1. **Detection**: Automated alerting on security events
2. **Containment**: Immediate container isolation
3. **Investigation**: Forensic analysis of logs
4. **Remediation**: Patch deployment and verification
5. **Recovery**: Service restoration
6. **Lessons Learned**: Process improvement

### Emergency Procedures
```bash
# Emergency container stop
docker stop $(docker ps -q --filter ancestor=openapi-doc-generator)

# Remove potentially compromised images
docker rmi openapi-doc-generator:latest

# Rebuild from verified source
git verify-tag v1.0.0
docker build --no-cache -t openapi-doc-generator:latest .
```

This comprehensive security framework ensures the OpenAPI Doc Generator container images meet enterprise security requirements while maintaining operational efficiency.