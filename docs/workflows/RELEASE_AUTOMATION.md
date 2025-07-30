# Release Automation Pipeline

Advanced automated release and deployment pipeline for OpenAPI-Doc-Generator.

## Release Strategy Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Development   │────│   Integration   │────│    Staging      │────│   Production    │
│                 │    │                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │                       │
        │                       │                       │                       │
   PR Validation          Release Candidate        Pre-Production         Production
                                                      Testing              Deployment
```

## Automated Versioning

### Semantic Release Configuration

**File**: `.releaserc.json`

```json
{
  "preset": "conventionalcommits",
  "branches": [
    "main",
    {"name": "develop", "prerelease": "beta"},
    {"name": "release/*", "prerelease": "${name.replace(/^release\\//g, \"rc.\")}"}
  ],
  "plugins": [
    "@semantic-release/commit-analyzer",
    "@semantic-release/release-notes-generator",
    "@semantic-release/changelog",
    ["@semantic-release/exec", {
      "prepareCmd": "python scripts/update_version.py ${nextRelease.version}",
      "publishCmd": "python -m build && twine upload dist/*"
    }],
    ["@semantic-release/github", {
      "assets": [
        {"path": "dist/*.whl", "label": "Python Wheel"},
        {"path": "dist/*.tar.gz", "label": "Source Distribution"},
        {"path": "docs/generated/openapi.json", "label": "OpenAPI Specification"},
        {"path": "sbom.json", "label": "Software Bill of Materials"}
      ]
    }],
    "@semantic-release/git"
  ]
}
```

### Version Update Script

**File**: `scripts/update_version.py`

```python
#!/usr/bin/env python3
"""Automated version updating for releases."""

import re
import sys
from pathlib import Path

def update_version(new_version: str):
    """Update version in all relevant files."""
    files_to_update = {
        "pyproject.toml": r'version = ".*"',
        "src/openapi_doc_generator/__init__.py": r'__version__ = ".*"',
        "docs/SETUP_REQUIRED.md": r'version: .*',
        "Dockerfile": r'ARG VERSION=.*'
    }
    
    for file_path, pattern in files_to_update.items():
        path = Path(file_path)
        if not path.exists():
            continue
            
        content = path.read_text()
        
        if "pyproject.toml" in file_path:
            replacement = f'version = "{new_version}"'
        elif "__init__.py" in file_path:
            replacement = f'__version__ = "{new_version}"'
        elif "SETUP_REQUIRED.md" in file_path:
            replacement = f'version: {new_version}'
        elif "Dockerfile" in file_path:
            replacement = f'ARG VERSION={new_version}'
        
        updated_content = re.sub(pattern, replacement, content)
        path.write_text(updated_content)
        
        print(f"Updated {file_path} with version {new_version}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_version.py <version>")
        sys.exit(1)
    
    version = sys.argv[1]
    update_version(version)
```

## Deployment Pipeline

### Production Deployment Workflow

**File**: `.github/workflows/deploy.yml`

```yaml
name: Production Deployment

on:
  release:
    types: [published]
  workflow_dispatch:
    inputs:
      environment:
        description: 'Deployment environment'
        required: true
        default: 'staging'
        type: choice
        options:
        - staging
        - production

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

  build-and-push:
    runs-on: ubuntu-latest
    needs: security-scan
    permissions:
      contents: read
      packages: write
    outputs:
      image-digest: ${{ steps.build.outputs.digest }}
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=semver,pattern={{major}}
    
    - name: Build and push Docker image
      id: build
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
        build-args: |
          BUILD_DATE=${{ fromJSON(steps.meta.outputs.json).labels['org.opencontainers.image.created'] }}
          VERSION=${{ fromJSON(steps.meta.outputs.json).labels['org.opencontainers.image.version'] }}
          VCS_REF=${{ github.sha }}

  deploy-staging:
    if: github.event_name == 'workflow_dispatch' && github.event.inputs.environment == 'staging'
    runs-on: ubuntu-latest
    needs: build-and-push
    environment: staging
    
    steps:
    - name: Deploy to staging
      run: |
        echo "Deploying to staging environment"
        # Add staging deployment logic here

  deploy-production:
    if: github.event_name == 'release' || (github.event_name == 'workflow_dispatch' && github.event.inputs.environment == 'production')
    runs-on: ubuntu-latest
    needs: build-and-push
    environment: production
    
    steps:
    - name: Deploy to production
      run: |
        echo "Deploying to production environment"
        # Add production deployment logic here
    
    - name: Generate SBOM
      uses: anchore/sbom-action@v0
      with:
        image: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}@${{ needs.build-and-push.outputs.image-digest }}
        format: spdx-json
        output-file: sbom.json
    
    - name: Upload SBOM to release
      uses: softprops/action-gh-release@v1
      if: github.event_name == 'release'
      with:
        files: sbom.json

  post-deployment:
    runs-on: ubuntu-latest
    needs: [deploy-staging, deploy-production]
    if: always() && (needs.deploy-staging.result == 'success' || needs.deploy-production.result == 'success')
    
    steps:
    - name: Run smoke tests
      run: |
        # Add smoke tests here
        echo "Running post-deployment smoke tests"
    
    - name: Update deployment status
      uses: actions/github-script@v7
      with:
        script: |
          const environment = '${{ github.event.inputs.environment || "production" }}';
          const status = 'success';
          console.log(`Deployment to ${environment}: ${status}`);
```

## Container Registry Management

### Multi-Architecture Builds

```yaml
# .github/workflows/build-multiarch.yml
name: Multi-Architecture Build

on:
  push:
    branches: [main]
  schedule:
    - cron: '0 2 * * 0'  # Weekly base image updates

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        platform:
          - linux/amd64
          - linux/arm64
          - linux/arm/v7
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up QEMU
      uses: docker/setup-qemu-action@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Build for ${{ matrix.platform }}
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: ${{ matrix.platform }}
        push: false
        tags: openapi-doc-generator:${{ matrix.platform }}
```

## Deployment Environments

### Staging Environment

```yaml
# k8s/staging/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: openapi-doc-generator-staging
  namespace: staging
spec:
  replicas: 2
  selector:
    matchLabels:
      app: openapi-doc-generator
      environment: staging
  template:
    metadata:
      labels:
        app: openapi-doc-generator
        environment: staging
    spec:
      containers:
      - name: app
        image: ghcr.io/danieleschmidt/openapi-doc-generator:latest
        ports:
        - containerPort: 8080
        env:
        - name: ENVIRONMENT
          value: "staging"
        - name: LOG_LEVEL
          value: "DEBUG"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Production Environment

```yaml
# k8s/production/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: openapi-doc-generator-production
  namespace: production
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: openapi-doc-generator
      environment: production
  template:
    metadata:
      labels:
        app: openapi-doc-generator
        environment: production
    spec:
      containers:
      - name: app
        image: ghcr.io/danieleschmidt/openapi-doc-generator:latest
        ports:
        - containerPort: 8080
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 10
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - openapi-doc-generator
              topologyKey: kubernetes.io/hostname
```

## Rollback Strategy

### Automated Rollback

```bash
#!/bin/bash
# scripts/rollback.sh

ENVIRONMENT=${1:-production}
PREVIOUS_VERSION=${2}

if [ -z "$PREVIOUS_VERSION" ]; then
    echo "Getting previous successful deployment..."
    PREVIOUS_VERSION=$(kubectl get deployment openapi-doc-generator-${ENVIRONMENT} -o jsonpath='{.metadata.annotations.deployment\.kubernetes\.io\/revision}')
    PREVIOUS_VERSION=$((PREVIOUS_VERSION - 1))
fi

echo "Rolling back to version: $PREVIOUS_VERSION"

kubectl rollout undo deployment/openapi-doc-generator-${ENVIRONMENT} --to-revision=$PREVIOUS_VERSION -n ${ENVIRONMENT}

echo "Waiting for rollback to complete..."
kubectl rollout status deployment/openapi-doc-generator-${ENVIRONMENT} -n ${ENVIRONMENT}

echo "Running post-rollback verification..."
./scripts/smoke_tests.sh ${ENVIRONMENT}
```

## Monitoring and Alerting

### Release Monitoring

```yaml
# monitoring/release-alerts.yml
groups:
- name: release-monitoring
  rules:
  - alert: DeploymentFailed
    expr: kube_deployment_status_condition{condition="Progressing",status="false"} == 1
    for: 10m
    labels:
      severity: critical
    annotations:
      summary: "Deployment failed for {{ $labels.deployment }}"
      
  - alert: HighErrorRatePostRelease
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected after release"
```

## Usage Instructions

### Manual Release
```bash
# Create release candidate
git checkout -b release/v1.2.0
git push origin release/v1.2.0

# Automatic release from main
git tag -a v1.2.0 -m "Release v1.2.0"
git push origin v1.2.0
```

### Emergency Rollback
```bash
# Quick rollback
./scripts/rollback.sh production

# Rollback to specific version
./scripts/rollback.sh production v1.1.0
```

## Next Steps

1. Set up semantic-release configuration
2. Configure deployment environments
3. Implement monitoring and alerting
4. Test rollback procedures
5. Document incident response procedures