"""Utilities for generating markdown documentation from OpenAPI specs."""

from __future__ import annotations

from dataclasses import dataclass
import logging
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
        routes = self._extract_routes_from_spec(spec)
        context = self._build_template_context(spec, routes)
        
        self._logger.debug("Rendering markdown for %d routes", len(routes))
        return template.render(context)

    def _extract_routes_from_spec(self, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract route information from OpenAPI specification."""
        routes: List[Dict[str, Any]] = []
        paths = spec.get("paths", {})
        
        for path, methods in paths.items():
            for method, operation in methods.items():
                route_info = self._create_route_info(path, method, operation)
                routes.append(route_info)
        
        return routes

    def _create_route_info(self, path: str, method: str, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Create route information dictionary for a single operation."""
        name = operation.get("summary") or operation.get("operationId") or method
        return {
            "path": path,
            "methods": [method.upper()],
            "name": name,
        }

    def _build_template_context(self, spec: Dict[str, Any], routes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build context dictionary for template rendering."""
        return {
            "title": spec.get("info", {}).get("title", "API"),
            "routes": routes,
        }
