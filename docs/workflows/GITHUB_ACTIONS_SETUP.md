# GitHub Actions Setup Guide

This document provides instructions for setting up the complete CI/CD pipeline for the OpenAPI-Doc-Generator repository.

## Required GitHub Actions Workflows

The repository currently has workflow templates in `docs/workflows/` but needs active workflows in `.github/workflows/`. 

### 1. Core CI/CD Workflow

**File**: `.github/workflows/ci.yml`

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  release:
    types: [ published ]

env:
  PYTHON_VERSION: "3.11"
  NODE_VERSION: "18"

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.8", "3.9", "3.10", "3.11"]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]
    
    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml --cov-report=term-missing
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Run security scans
      run: |
        make security
    
    - name: Upload security results
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: security_results.json

  docker:
    runs-on: ubuntu-latest
    needs: [test, security]
    steps:
    - uses: actions/checkout@v4
    - name: Build Docker image
      run: docker build -t openapi-doc-generator:latest .
    
    - name: Test Docker image
      run: |
        docker run --rm openapi-doc-generator:latest --version
```

### 2. Release Automation

**File**: `.github/workflows/release.yml`

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      packages: write
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"
    
    - name: Build package
      run: |
        pip install build
        python -m build
    
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: |
          ghcr.io/danieleschmidt/openapi-doc-generator:latest
          ghcr.io/danieleschmidt/openapi-doc-generator:${{ github.ref_name }}
```

## Setup Instructions

### 1. Create Workflow Directory
```bash
mkdir -p .github/workflows
```

### 2. Copy Templates
Copy the workflow templates from `docs/workflows/` to `.github/workflows/` and rename:
- `ci-advanced-template.yml` → `ci.yml`
- `release-automation-template.yml` → `release.yml`
- `security-advanced-template.yml` → `security.yml`

### 3. Configure Secrets
Add the following secrets in repository settings:
- `PYPI_API_TOKEN`: PyPI publishing token
- `CODECOV_TOKEN`: Codecov integration token
- `DOCKER_REGISTRY_TOKEN`: Container registry access

### 4. Enable Actions
1. Go to repository Settings → Actions → General
2. Set "Actions permissions" to "Allow all actions and reusable workflows"
3. Enable "Allow GitHub Actions to create and approve pull requests"

### 5. Branch Protection
Configure branch protection rules for `main` branch:
- Require status checks to pass
- Require branches to be up to date
- Require review from code owners

## Status Checks

The workflows will provide these status checks:
- ✅ Tests pass on Python 3.8-3.11
- ✅ Security scans pass
- ✅ Code coverage meets threshold
- ✅ Docker image builds successfully
- ✅ Documentation generation works

## Monitoring

Enable workflow notifications:
1. Repository Settings → Notifications
2. Enable "Actions" notifications
3. Configure Slack/Teams integration if needed

## Next Steps

After setting up workflows:
1. Monitor first workflow runs
2. Adjust coverage thresholds if needed
3. Configure additional integrations
4. Set up deployment environments