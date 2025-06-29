"""High level API documentation orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import List

from .discovery import RouteDiscoverer, RouteInfo
from .schema import SchemaInferer, SchemaInfo
from .spec import OpenAPISpecGenerator
from .markdown import MarkdownGenerator


@dataclass
class DocumentationResult:
    routes: List[RouteInfo]
    schemas: List[SchemaInfo]

    def generate_openapi_spec(self, title: str = "API", version: str = "1.0.0") -> dict:
        """Return OpenAPI specification for the analyzed app."""
        generator = OpenAPISpecGenerator(
            self.routes, self.schemas, title=title, version=version
        )
        return generator.generate()

    def generate_markdown(self, title: str = "API", version: str = "1.0.0") -> str:
        """Return markdown documentation for the analyzed app."""
        spec = self.generate_openapi_spec(title=title, version=version)
        return MarkdownGenerator().generate(spec)


class APIDocumentator:
    """Analyze an application to generate documentation artifacts."""

    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)

    def analyze_app(self, app_path: str) -> DocumentationResult:
        self._logger.info("Discovering routes from %s", app_path)
        routes = RouteDiscoverer(app_path).discover()
        try:
            schemas = SchemaInferer(app_path).infer()
        except FileNotFoundError:
            self._logger.info("No models found in %s", app_path)
            schemas = []
        return DocumentationResult(routes=routes, schemas=schemas)
