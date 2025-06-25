"""OpenAPI specification generation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any

from .discovery import RouteInfo
from .schema import SchemaInfo


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

    title: str = "API"
    version: str = "1.0.0"

    def generate(self) -> Dict[str, Any]:
        spec: Dict[str, Any] = {
            "openapi": "3.0.0",
            "info": {"title": self.title, "version": self.version},
            "paths": {},
            "components": {"schemas": {}},
        }

        for route in self.routes:
            path_item = spec["paths"].setdefault(route.path or "/", {})
            for method in route.methods:
                operation = {
                    "summary": route.docstring or route.name,
                    "responses": {"200": {"description": "Success"}},
                }
                path_item[method.lower()] = operation

        for schema in self.schemas:
            properties = {}
            required = []
            for field in schema.fields:
                properties[field.name] = {"type": _type_to_openapi(field.type)}
                if field.required:
                    required.append(field.name)
            schema_obj: Dict[str, Any] = {"type": "object", "properties": properties}
            if required:
                schema_obj["required"] = required
            spec["components"]["schemas"][schema.name] = schema_obj

        return spec
