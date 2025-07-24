"""OpenAPI specification generation utilities."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, List, Any

from .discovery import RouteInfo
from .schema import SchemaInfo
from .config import config


def _type_to_openapi(py_type: str) -> str:
    mapping = {
        "int": "integer",
        "float": "number",
        "str": "string",
        "bool": "boolean",
    }
    lower = py_type.lower()
    if lower in mapping:
        return mapping[lower]
    if lower.startswith("list") or lower.startswith("sequence"):
        return "array"
    if lower.startswith("dict") or lower.startswith("mapping"):
        return "object"
    return "string"


@dataclass
class OpenAPISpecGenerator:
    """Generate a basic OpenAPI 3 specification."""

    routes: List[RouteInfo]
    schemas: List[SchemaInfo]

    title: str = config.DEFAULT_API_TITLE
    version: str = config.DEFAULT_API_VERSION

    def generate(self) -> Dict[str, Any]:
        logger = logging.getLogger(self.__class__.__name__)
        logger.debug("Generating OpenAPI spec")
        spec = self._create_base_spec()
        self._add_paths_to_spec(spec)
        self._add_schemas_to_spec(spec)
        return spec
    
    def _create_base_spec(self) -> Dict[str, Any]:
        """Create the base OpenAPI specification structure."""
        return {
            "openapi": config.OPENAPI_VERSION,
            "info": {"title": self.title, "version": self.version},
            "paths": {},
            "components": {"schemas": {}},
        }
    
    def _add_paths_to_spec(self, spec: Dict[str, Any]) -> None:
        """Add route information to the paths section of the spec."""
        for route in self.routes:
            path_item = spec["paths"].setdefault(route.path or "/", {})
            for method in route.methods:
                operation = {
                    "summary": route.docstring or route.name,
                    "responses": config.DEFAULT_SUCCESS_RESPONSE,
                }
                path_item[method.lower()] = operation
    
    def _add_schemas_to_spec(self, spec: Dict[str, Any]) -> None:
        """Add schema definitions to the components section of the spec."""
        for schema in self.schemas:
            schema_obj = self._build_schema_object(schema)
            spec["components"]["schemas"][schema.name] = schema_obj
    
    def _build_schema_object(self, schema: SchemaInfo) -> Dict[str, Any]:
        """Build an OpenAPI schema object from a SchemaInfo."""
        properties = {}
        required = []
        for field in schema.fields:
            properties[field.name] = {"type": _type_to_openapi(field.type)}
            if field.required:
                required.append(field.name)
        
        schema_obj: Dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema_obj["required"] = required
        return schema_obj
