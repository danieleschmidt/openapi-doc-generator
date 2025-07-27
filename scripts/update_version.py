#!/usr/bin/env python3
"""Script to update version in pyproject.toml and __init__.py"""

import sys
import toml
from pathlib import Path


def update_version(new_version: str) -> None:
    """Update version in pyproject.toml and __init__.py files."""
    
    # Update pyproject.toml
    pyproject_path = Path("pyproject.toml")
    if pyproject_path.exists():
        with open(pyproject_path) as f:
            data = toml.load(f)
        
        data["project"]["version"] = new_version
        
        with open(pyproject_path, "w") as f:
            toml.dump(data, f)
        
        print(f"Updated pyproject.toml version to {new_version}")
    
    # Update __init__.py
    init_path = Path("src/openapi_doc_generator/__init__.py")
    if init_path.exists():
        content = init_path.read_text()
        
        # Replace version line
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('__version__'):
                lines[i] = f'__version__ = "{new_version}"'
                break
        else:
            # Add version if not found
            lines.insert(0, f'__version__ = "{new_version}"')
        
        init_path.write_text('\n'.join(lines))
        print(f"Updated __init__.py version to {new_version}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_version.py <version>")
        sys.exit(1)
    
    version = sys.argv[1]
    update_version(version)
    print(f"Version updated to {version}")