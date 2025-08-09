"""Utilities for generating markdown documentation from OpenAPI specs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from jinja2 import Template

from .templates import load_template


@dataclass
class MarkdownGenerator:
    """Convert OpenAPI specifications into markdown documentation."""

    template_name: str = "api.md.jinja"

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)

    def generate(self, spec: Dict[str, Any] | None) -> str:
        """Return markdown for the provided OpenAPI specification."""
        if not isinstance(spec, dict):
            raise TypeError("spec must be a dict")

        template = Template(load_template(self.template_name))
        routes: List[Dict[str, Any]] = []
        paths = spec.get("paths", {})
        for path, methods in paths.items():
            for method, operation in methods.items():
                name = (
                    operation.get("summary") or operation.get("operationId") or method
                )
                routes.append(
                    {
                        "path": path,
                        "methods": [method.upper()],
                        "name": name,
                    }
                )

        context = {
            "title": spec.get("info", {}).get("title", "API"),
            "routes": routes,
        }
        self._logger.debug("Rendering markdown for %d routes", len(routes))
        return template.render(context)
