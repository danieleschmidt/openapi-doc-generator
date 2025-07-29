#!/usr/bin/env python3
"""Generate Software Bill of Materials (SBOM) for the project."""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

def run_command(cmd: List[str]) -> str:
    """Run command and return output."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running {' '.join(cmd)}: {e}")
        return ""

def get_installed_packages() -> List[Dict[str, Any]]:
    """Get list of installed Python packages."""
    try:
        output = run_command([sys.executable, "-m", "pip", "list", "--format=json"])
        return json.loads(output) if output else []
    except json.JSONDecodeError:
        return []

def get_git_info() -> Dict[str, str]:
    """Get Git repository information."""
    return {
        "commit_hash": run_command(["git", "rev-parse", "HEAD"]),
        "branch": run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "remote_url": run_command(["git", "config", "--get", "remote.origin.url"]),
        "last_commit_date": run_command(["git", "log", "-1", "--format=%ci"])
    }

def generate_sbom() -> Dict[str, Any]:
    """Generate SBOM data structure."""
    packages = get_installed_packages()
    git_info = get_git_info()
    
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.4",
        "serialNumber": f"urn:uuid:{datetime.now().isoformat()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "tools": [
                {
                    "vendor": "terragon-labs",
                    "name": "sbom-generator",
                    "version": "1.0.0"
                }
            ],
            "component": {
                "type": "application",
                "name": "openapi-doc-generator",
                "version": "0.1.0",
                "description": "Advanced OpenAPI documentation generator",
                "repository": git_info.get("remote_url", ""),
                "commit": git_info.get("commit_hash", ""),
                "branch": git_info.get("branch", "")
            }
        },
        "components": []
    }
    
    # Add Python packages as components
    for pkg in packages:
        component = {
            "type": "library",
            "name": pkg["name"],
            "version": pkg["version"],
            "purl": f"pkg:pypi/{pkg['name']}@{pkg['version']}",
            "scope": "required"
        }
        sbom["components"].append(component)
    
    return sbom

def main():
    """Main function."""
    print("Generating SBOM...")
    sbom = generate_sbom()
    
    output_file = Path("sbom.json")
    with open(output_file, "w") as f:
        json.dump(sbom, f, indent=2, sort_keys=True)
    
    print(f"SBOM generated: {output_file}")
    print(f"Components found: {len(sbom['components'])}")

if __name__ == "__main__":
    main()