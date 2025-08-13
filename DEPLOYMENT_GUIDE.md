# 🚀 Production Deployment Guide

## Quick Start Production Deployment

### Using Docker Compose (Recommended)
```bash
# Set environment variables
export VERSION=0.1.0
export BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
export VCS_REF=$(git rev-parse HEAD)

# Deploy production stack
docker-compose -f docker-compose.production.yml up -d

# Generate documentation
docker-compose -f docker-compose.production.yml exec openapi-doc-generator \
  openapi-doc-generator --app /workspace/app.py --format openapi --output /output/spec.json

# View health status
curl http://localhost:8080/health
```

### Using Pre-built Docker Images
```bash
# Latest stable release
docker pull ghcr.io/danieleschmidt/openapi-doc-generator:latest

# Generate documentation
docker run --rm -v $(pwd):/workspace -v $(pwd)/output:/output \
  ghcr.io/danieleschmidt/openapi-doc-generator:latest \
  --app /workspace/app.py --format openapi --output /output/spec.json

# With performance monitoring
docker run --rm -v $(pwd):/workspace -v $(pwd)/output:/output \
  ghcr.io/danieleschmidt/openapi-doc-generator:latest \
  --app /workspace/app.py --format openapi --performance-metrics \
  --log-format json --output /output/spec.json
```

### Using Python Package
```bash
# Install from PyPI (when published)
pip install openapi-doc-generator

# Generate documentation
openapi-doc-generator --app ./app.py --format openapi --output openapi.json

# With quantum planning
openapi-doc-generator --app ./app.py --format quantum-plan \
  --quantum-temperature 1.0 --quantum-resources 4
```

## 🛡️ Security Configuration

### Container Security
- **Non-root user**: Runs as UID 1000 with minimal privileges
- **Read-only filesystem**: Container filesystem is read-only
- **Dropped capabilities**: All capabilities dropped except essential ones
- **No new privileges**: Prevents privilege escalation
- **Security scanning**: Automated Trivy security scans

### Input Validation
- **Path sanitization**: All file paths validated and sanitized
- **Input limits**: File size limits (100MB default)
- **Resource monitoring**: Memory and CPU usage tracking
- **Timeout protection**: Operation timeouts prevent DoS

## 📊 Monitoring & Observability

### Health Endpoints
```bash
# Container health
curl http://localhost:8080/health

# Detailed metrics
curl http://localhost:8080/metrics

# Performance statistics
curl http://localhost:8080/performance
```

### Structured Logging
```bash
# Enable JSON logging for production
openapi-doc-generator --app ./app.py --log-format json --performance-metrics
```

### Performance Monitoring
- **Request tracing**: Correlation IDs for all operations
- **Memory tracking**: Before/after memory usage
- **Execution timing**: Detailed operation timing
- **Resource utilization**: CPU, memory, disk usage

## 🌍 Global Deployment

### Multi-Region Configuration
```bash
# US deployment
openapi-doc-generator --app ./app.py --region US --compliance ccpa

# European deployment  
openapi-doc-generator --app ./app.py --region EU --compliance gdpr --language de

# Asia-Pacific deployment
openapi-doc-generator --app ./app.py --region APAC --compliance pdpa-sg --language ja
```

### Internationalization
Supported languages: `en`, `es`, `fr`, `de`, `ja`, `zh`, `pt`, `it`, `ru`, `ko`

```bash
# German output
openapi-doc-generator --app ./app.py --language de --format markdown

# Auto-detect system language
openapi-doc-generator --app ./app.py --format openapi
```

## ⚡ Performance Optimization

### Quantum-Enhanced Planning
```bash
# High-performance quantum planning
openapi-doc-generator --app ./app.py --format quantum-plan \
  --quantum-temperature 0.5 --quantum-resources 8 \
  --quantum-cooling-rate 0.98 --performance-metrics
```

### Caching Configuration
```bash
# Enable all optimizations
export OPENAPI_ENABLE_CACHING=true
export OPENAPI_PARALLEL_PROCESSING=true
export OPENAPI_ADAPTIVE_SCALING=true

openapi-doc-generator --app ./app.py --performance-metrics
```

### Horizontal Scaling
```bash
# Scale with multiple instances
docker-compose -f docker-compose.production.yml up --scale openapi-doc-generator=3
```

## 🔧 Configuration Management

### Environment Variables
```bash
# Core settings
LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT=json                   # standard, json
PYTHONUNBUFFERED=1               # Disable output buffering

# Performance settings
OPENAPI_ENABLE_CACHING=true      # Enable AST caching
OPENAPI_MAX_FILE_SIZE=104857600  # 100MB file size limit
OPENAPI_TIMEOUT_SECONDS=300      # 5-minute operation timeout

# Monitoring settings
MONITORING_ENABLED=true          # Enable health monitoring
METRICS_RETENTION_HOURS=24       # Metrics retention period
```

### Configuration Files
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

security:
  validation_level: "strict"
  input_sanitization: true
  resource_limits: true

performance:
  enable_caching: true
  parallel_processing: true
  adaptive_scaling: true
```

## 🚨 Troubleshooting

### Common Issues

**Memory Issues**
```bash
# Check memory usage
docker stats openapi-doc-generator

# Increase memory limits
docker run --memory=2g --memory-swap=4g ...
```

**Performance Issues**
```bash
# Enable detailed profiling
openapi-doc-generator --app ./app.py --performance-metrics --verbose

# Check system resources
curl http://localhost:8080/health
```

**Security Issues**
```bash
# Run security scan
bandit -r src/
docker run --rm -v $(pwd):/workspace aquasec/trivy fs /workspace
```

### Debug Mode
```bash
# Enable debug logging
LOG_LEVEL=DEBUG openapi-doc-generator --app ./app.py --verbose

# Profile performance issues
python -m cProfile -o profile.stats openapi-doc-generator --app ./app.py
```

## 📈 Scaling Strategies

### Vertical Scaling
- Increase container CPU/memory limits
- Enable quantum optimization features
- Use SSD storage for better I/O performance

### Horizontal Scaling
- Deploy multiple container instances
- Use load balancer for request distribution
- Implement result caching layer

### Auto-Scaling
```yaml
# docker-compose.yml scaling configuration
services:
  openapi-doc-generator:
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
```

## 🔐 Compliance & Governance

### GDPR Compliance
```bash
openapi-doc-generator --app ./app.py --compliance gdpr --region EU
```

### SOC 2 Features
- Audit logging for all operations
- Access control and authentication
- Data encryption at rest and in transit
- Regular security scanning and updates

### Enterprise Features
- SAML/OIDC integration support
- Enterprise audit logging
- Custom plugin development
- Priority support channels

---

For detailed examples and advanced configuration, see the [examples/](examples/) directory and [EXTENDING.md](EXTENDING.md) guide.