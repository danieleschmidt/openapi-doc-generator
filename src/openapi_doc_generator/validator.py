"""OpenAPI specification validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class SpecValidator:
    """Validate OpenAPI specifications and suggest improvements."""

    def validate(self, spec: Dict[str, Any]) -> List[str]:
        """Return a list of suggestions for improving the spec."""
        if not isinstance(spec, dict):
            raise TypeError("spec must be a dict")

        suggestions: List[str] = []

        version = spec.get("openapi", "")
        if not version.startswith("3"):
            suggestions.append("OpenAPI version should be 3.x")

        paths = spec.get("paths", {})
        for path, operations in paths.items():
            if not operations:
                suggestions.append(f"Path '{path}' has no operations")
                continue
            for method, op in operations.items():
                if "summary" not in op:
                    suggestions.append(
                        f"Operation '{method} {path}' is missing summary"
                    )

        return suggestions
