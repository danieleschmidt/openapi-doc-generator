import re
from typing import Dict, Any
import yaml


_TYPE_MAP = {
    "int": "integer",
    "float": "number",
    "str": "string",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
    "": "string",
}


def _python_type_to_openapi(annotation: str) -> str:
    base = annotation.split("[")[0].strip()
    return _TYPE_MAP.get(base, "string")


def generate_openapi(parsed: Dict[str, Any], version: str = "3.0.3") -> Dict:
    """Generate OpenAPI dict from parsed file data."""
    title = parsed.get("title", "API")
    routes = parsed.get("routes", [])

    paths = {}
    for route in routes:
        path = route["path"] or "/"
        method = route["method"]
        func = route["function"]
        docstring = route.get("docstring", "")
        params = route.get("params", [])

        # Build parameters list (path params)
        path_param_names = set()
        for m in re.finditer(r"\{(\w+)\}", path):
            path_param_names.add(m.group(1))

        parameters = []
        for p in params:
            if p["name"] in path_param_names:
                parameters.append({
                    "name": p["name"],
                    "in": "path",
                    "required": True,
                    "schema": {"type": _python_type_to_openapi(p["annotation"])},
                })
            else:
                parameters.append({
                    "name": p["name"],
                    "in": "query",
                    "required": False,
                    "schema": {"type": _python_type_to_openapi(p["annotation"])},
                })

        operation = {
            "operationId": func,
            "summary": docstring.split("\n")[0] if docstring else func,
            "responses": {
                "200": {"description": "Successful response"},
            },
        }
        if parameters:
            operation["parameters"] = parameters
        if docstring:
            operation["description"] = docstring

        if path not in paths:
            paths[path] = {}
        paths[path][method] = operation

    spec = {
        "openapi": version,
        "info": {
            "title": title,
            "version": "1.0.0",
        },
        "paths": paths,
    }
    return spec


def to_yaml(spec: Dict) -> str:
    return yaml.dump(spec, default_flow_style=False, sort_keys=False, allow_unicode=True)
