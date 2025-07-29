# Manual GitHub Workflows Setup Guide

## Overview

Due to GitHub App permission restrictions, the workflow files need to be manually added to your repository. This guide provides step-by-step instructions for implementing the comprehensive CI/CD automation.

## Workflow Files to Create

### 1. `.github/workflows/ci.yml`
```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  PYTHON_VERSION: "3.12"

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.8", "3.9", "3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run pre-commit hooks
      run: pre-commit run --all-files
    
    - name: Run tests with coverage
      run: |
        coverage run -m pytest tests/ -v
        coverage xml
        coverage report
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: false

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Security scan with bandit
      run: bandit -r src/ -f sarif -o bandit-results.sarif
    
    - name: Upload SARIF results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: bandit-results.sarif
    
    - name: Dependency vulnerability scan
      run: |
        safety check --json --output safety-report.json || true
        pip-audit --format json --output pip-audit-report.json || true

  quality:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Type checking with mypy
      run: mypy src/
    
    - name: Code complexity analysis
      run: radon cc src/ --min C
```

### 2. `.github/workflows/security.yml`
```yaml
name: Security

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 6 * * 1'  # Weekly security scan

permissions:
  contents: read
  security-events: write

jobs:
  codeql:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
    
    - name: Initialize CodeQL
      uses: github/codeql-action/init@v2
      with:
        languages: python
    
    - name: Autobuild
      uses: github/codeql-action/autobuild@v2
    
    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2

  dependency-scan:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.12"
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Generate SBOM
      run: |
        pip install cyclonedx-bom
        cyclonedx-py --output-format json --output-file sbom.json
    
    - name: Upload SBOM
      uses: actions/upload-artifact@v4
      with:
        name: sbom
        path: sbom.json
    
    - name: Vulnerability scanning with safety
      run: |
        safety check --full-report --json --output safety-report.json
    
    - name: Audit Python packages
      run: |
        pip-audit --format json --output pip-audit.json
    
    - name: Upload security reports
      uses: actions/upload-artifact@v4
      with:
        name: security-reports
        path: |
          safety-report.json
          pip-audit.json

  container-scan:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Build Docker image
      run: docker build -t openapi-doc-generator:test .
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'openapi-doc-generator:test'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
```

### 3. `.github/workflows/release.yml`
```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

permissions:
  contents: write
  packages: write

jobs:
  release:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.12"
    
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Test package installation
      run: |
        pip install dist/*.whl
        openapi-doc-generator --version
    
    - name: Generate release notes
      id: changelog
      run: |
        pip install gitpython
        python scripts/generate_changelog.py > RELEASE_NOTES.md
    
    - name: Create GitHub Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: ${{ github.ref_name }}
        release_name: Release ${{ github.ref_name }}
        body_path: RELEASE_NOTES.md
        draft: false
        prerelease: false
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
      run: twine upload dist/*

  docker:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Log in to GitHub Container Registry
      uses: docker/login-action@v3
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ghcr.io/${{ github.repository }}
        tags: |
          type=ref,event=tag
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=raw,value=latest,enable={{is_default_branch}}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        platforms: linux/amd64,linux/arm64
```

### 4. `.github/workflows/compliance.yml`
```yaml
name: Compliance

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 9 * * 1'  # Weekly compliance check

permissions:
  contents: read
  issues: write
  pull-requests: write

jobs:
  compliance-check:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.12"
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
        pip install pip-licenses cyclonedx-bom
    
    - name: Generate SBOM
      run: python scripts/generate_sbom.py
    
    - name: Run compliance checks
      run: python scripts/compliance_check.py --strict
    
    - name: Upload compliance report
      uses: actions/upload-artifact@v4
      with:
        name: compliance-report
        path: |
          compliance_report.json
          sbom.json
    
    - name: Comment compliance status on PR
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v7
      with:
        script: |
          const fs = require('fs');
          
          try {
            const report = JSON.parse(fs.readFileSync('compliance_report.json', 'utf8'));
            const score = report.compliance_score;
            const status = report.overall_status;
            const failedChecks = report.failed_checks || [];
            
            const statusEmoji = {
              'compliant': '✅',
              'partially_compliant': '⚠️', 
              'non_compliant': '❌'
            };
            
            const body = `## Compliance Check Results ${statusEmoji[status]}
            
            **Overall Status:** ${status}  
            **Compliance Score:** ${score}%
            
            ${failedChecks.length > 0 ? `
            **Failed Checks:**
            ${failedChecks.map(check => `- ${check}`).join('\n')}
            ` : '**All compliance checks passed!** 🎉'}
            
            <details>
            <summary>View detailed compliance report</summary>
            
            \`\`\`json
            ${JSON.stringify(report, null, 2)}
            \`\`\`
            </details>
            `;
            
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: body
            });
          } catch (error) {
            console.log('Could not read compliance report:', error);
          }

  license-compliance:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.12"
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
        pip install pip-licenses
    
    - name: Check license compliance
      run: |
        pip-licenses --format=json --output-file=licenses.json
        python -c "
        import json
        with open('licenses.json') as f:
            licenses = json.load(f)
        
        blocked = ['GPL-3.0', 'AGPL-3.0', 'LGPL-3.0']
        violations = [p for p in licenses if p.get('License') in blocked]
        
        if violations:
            print('License violations found:')
            for v in violations:
                print(f\"  {v.get('Name')}: {v.get('License')}\")
            exit(1)
        else:
            print('All licenses are compliant')
        "
    
    - name: Upload license report
      uses: actions/upload-artifact@v4
      with:
        name: license-report
        path: licenses.json

  sbom-generation:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.12"
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
        pip install cyclonedx-bom
    
    - name: Generate SBOM with CycloneDX
      run: |
        cyclonedx-py --output-format json --output-file cyclonedx-sbom.json
        python scripts/generate_sbom.py
    
    - name: Validate SBOM
      run: |
        python -c "
        import json
        import sys
        
        try:
            with open('sbom.json') as f:
                sbom = json.load(f)
            
            required_fields = ['bomFormat', 'specVersion', 'metadata', 'components']
            missing = [f for f in required_fields if f not in sbom]
            
            if missing:
                print(f'SBOM validation failed. Missing fields: {missing}')
                sys.exit(1)
            
            component_count = len(sbom.get('components', []))
            print(f'SBOM validation passed. Components: {component_count}')
            
        except Exception as e:
            print(f'SBOM validation error: {e}')
            sys.exit(1)
        "
    
    - name: Upload SBOM artifacts
      uses: actions/upload-artifact@v4
      with:
        name: sbom-reports
        path: |
          sbom.json
          cyclonedx-sbom.json
```

## Implementation Steps

### 1. Create Workflow Directory
```bash
mkdir -p .github/workflows
```

### 2. Add Each Workflow File
Copy each workflow configuration above into the corresponding file in `.github/workflows/`

### 3. Configure Repository Secrets
In your GitHub repository settings, add these secrets:
- `PYPI_TOKEN` - For PyPI publishing (if using release workflow)
- `CODECOV_TOKEN` - For code coverage reporting (optional)

### 4. Configure Repository Settings
- Enable "Allow GitHub Actions to create and approve pull requests"
- Set up branch protection rules requiring workflow checks
- Enable dependency graph and Dependabot security updates

### 5. Test Workflows
1. Push to a test branch first
2. Verify all workflows execute successfully
3. Check that security scanning produces results
4. Validate compliance reporting works

## Additional Configuration

### Branch Protection Rules
Configure these rules for the `main` branch:
- Require status checks to pass before merging
- Require branches to be up to date before merging
- Include these status checks:
  - `test` (from CI workflow)
  - `security` (from CI workflow)
  - `quality` (from CI workflow)
  - `compliance-check` (from compliance workflow)

### Required Permissions
Ensure the repository has these permissions enabled:
- Actions: Read and write
- Contents: Read and write
- Issues: Write (for compliance reporting)
- Pull requests: Write (for compliance reporting)
- Security events: Write (for security scanning)

## Troubleshooting

### Common Issues
1. **Workflow not triggering**: Check event triggers and branch names
2. **Permission errors**: Verify repository permissions and secrets
3. **Dependency issues**: Ensure all required packages are in pyproject.toml
4. **Security scan failures**: Review and update security tool configurations

### Support
- Check workflow logs in the Actions tab
- Review the implementation report for detailed guidance
- Refer to individual tool documentation for specific issues

---

*These workflows provide comprehensive CI/CD automation, security scanning, and compliance validation for your Python project.*