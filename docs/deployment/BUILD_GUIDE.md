# Build Guide

## Overview

This guide covers building, packaging, and distributing the OpenAPI Doc Generator in various formats.

## Build Requirements

### Development Dependencies
- Python 3.8+
- pip 21.0+
- build tools (setuptools, wheel)
- Docker (for containerized builds)
- Git (for version metadata)

### System Dependencies
- make (for automation)
- bash (for scripts)
- curl or wget (for downloads)

## Local Development Build

### Quick Setup
```bash
# Clone repository
git clone https://github.com/danieleschmidt/openapi-doc-generator.git
cd openapi-doc-generator

# Development installation
make dev

# Verify installation
openapi-doc-generator --version
```

### Manual Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

## Package Builds

### Python Package (Wheel/Source)
```bash
# Clean previous builds
make clean

# Build package
make build

# Output: dist/openapi_doc_generator-X.Y.Z-py3-none-any.whl
#         dist/openapi_doc_generator-X.Y.Z.tar.gz
```

### Build Configuration
Package metadata is defined in `pyproject.toml`:
```toml
[project]
name = "openapi_doc_generator"
version = "0.1.0"
requires-python = ">=3.8"
dependencies = [
    "jinja2>=3.1",
    "graphql-core>=3.2",
]
```

## Docker Builds

### Standard Build
```bash
# Build with latest tag
make docker-build

# Build with custom tag
docker build -t openapi-doc-generator:v1.0.0 .

# Build with build arguments
docker build \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  --build-arg VERSION=1.0.0 \
  --build-arg VCS_REF=$(git rev-parse --short HEAD) \
  -t openapi-doc-generator:v1.0.0 .
```

### Multi-Platform Builds
```bash
# Enable buildx
docker buildx create --use

# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --tag openapi-doc-generator:v1.0.0 \
  --push .
```

### Development Build
```bash
# Use docker-compose for development
docker-compose --profile dev build

# Run development container
docker-compose --profile dev run openapi-doc-generator \
  /workspace/examples/app.py --format openapi
```

## Build Optimization

### Image Size Optimization
- Multi-stage build reduces final image size
- Only runtime dependencies in production stage
- Minimal base image (python:3.11-slim)
- .dockerignore excludes unnecessary files

### Build Speed Optimization
- Layer caching for dependencies
- Separate requirements installation
- Efficient file copying order
- Build argument caching

### Security Hardening
- Non-root user (UID 1000)
- Read-only root filesystem compatible
- Security scanning with Trivy
- Minimal attack surface

## Build Verification

### Package Verification
```bash
# Install built package
pip install dist/openapi_doc_generator-*.whl

# Test installation
openapi-doc-generator --version
openapi-doc-generator --help

# Run basic functionality test
openapi-doc-generator --app examples/app.py --format openapi
```

### Docker Verification
```bash
# Test container health
docker run --rm openapi-doc-generator:latest --version

# Test with mounted volume
docker run --rm -v $(pwd):/workspace \
  openapi-doc-generator:latest /workspace/examples/app.py --format openapi

# Check container security
docker run --rm -it openapi-doc-generator:latest whoami
# Should output: openapi (not root)
```

## Build Automation

### Makefile Targets
```bash
make build          # Build Python package
make docker-build   # Build Docker image
make clean          # Clean build artifacts
make ci             # Full CI pipeline
```

### Environment Variables
```bash
export BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
export VERSION=1.0.0
export VCS_REF=$(git rev-parse --short HEAD)
```

## Continuous Integration

### GitHub Actions
Build process is automated in `.github/workflows/`:
- `ci.yml`: Build and test on every push
- `release.yml`: Build and publish on tags
- `docker.yml`: Build and push Docker images

### Build Matrix
Tests across multiple configurations:
- Python versions: 3.8, 3.9, 3.10, 3.11
- Operating systems: Ubuntu, macOS, Windows
- Architectures: amd64, arm64

## Release Process

### Version Management
```bash
# Update version in pyproject.toml
# Create and push tag
git tag v1.0.0
git push origin v1.0.0

# GitHub Actions automatically:
# 1. Builds packages
# 2. Runs tests
# 3. Creates GitHub release
# 4. Publishes to PyPI
# 5. Builds and pushes Docker images
```

### Manual Release
```bash
# Build packages
make build

# Test packages
pip install dist/openapi_doc_generator-*.whl

# Upload to PyPI (requires authentication)
twine upload dist/*

# Build and push Docker image
docker build -t ghcr.io/danieleschmidt/openapi-doc-generator:v1.0.0 .
docker push ghcr.io/danieleschmidt/openapi-doc-generator:v1.0.0
```

## Build Troubleshooting

### Common Issues

#### Permission Errors
```bash
# Fix file permissions
chmod +x scripts/*.py

# Fix Docker permissions
sudo usermod -aG docker $USER
newgrp docker
```

#### Build Failures
```bash
# Clean everything
make clean
docker system prune -a

# Rebuild with verbose output
pip install -e . -v
docker build --no-cache --progress=plain .
```

#### Import Errors
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Verify package installation
pip show openapi-doc-generator
```

### Performance Issues
```bash
# Enable BuildKit for faster Docker builds
export DOCKER_BUILDKIT=1

# Use build cache
docker build --cache-from openapi-doc-generator:latest .

# Parallel package building
pip install --upgrade build setuptools wheel
python -m build --sdist --wheel --outdir dist/ .
```

## Distribution

### PyPI Publishing
```bash
# Install publishing tools
pip install twine

# Build and upload
python -m build
twine upload dist/*
```

### Container Registry
```bash
# Tag for registry
docker tag openapi-doc-generator:latest \
  ghcr.io/danieleschmidt/openapi-doc-generator:latest

# Push to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
docker push ghcr.io/danieleschmidt/openapi-doc-generator:latest
```

### Binary Distribution
```bash
# Create standalone executable with PyInstaller
pip install pyinstaller
pyinstaller --onefile src/openapi_doc_generator/cli.py

# Output: dist/cli (standalone executable)
```

This comprehensive build system ensures reliable, secure, and efficient distribution across multiple platforms and deployment scenarios.