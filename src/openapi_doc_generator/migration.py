"""Generate migration guides between API versions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class MigrationGuideGenerator:
    """Compare OpenAPI specs and generate a markdown migration guide."""

    old_spec: Dict[str, Any]
    new_spec: Dict[str, Any]

    def generate_markdown(self) -> str:
        old_paths = self.old_spec.get("paths", {})
        new_paths = self.new_spec.get("paths", {})

        removed: List[str] = []
        added: List[str] = []

        for path, ops in old_paths.items():
            for method in ops.keys():
                if path not in new_paths or method not in new_paths[path]:
                    removed.append(f"{method.upper()} {path}")

        for path, ops in new_paths.items():
            for method in ops.keys():
                if path not in old_paths or method not in old_paths[path]:
                    added.append(f"{method.upper()} {path}")

        lines = ["# API Migration Guide", ""]
        if removed:
            lines.append("## Removed Endpoints")
            for item in removed:
                lines.append(f"- {item}")
            lines.append("")
        if added:
            lines.append("## New Endpoints")
            for item in added:
                lines.append(f"- {item}")
            lines.append("")
        if not added and not removed:
            lines.append("No changes detected.")
        return "\n".join(lines)
