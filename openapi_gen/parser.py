import ast
import re
from typing import List, Dict, Any
from pathlib import Path


def _get_string(node) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ""


def _parse_fastapi_routes(tree: ast.AST) -> List[Dict]:
    """Extract FastAPI route decorators: @app.get, @router.post, etc."""
    routes = []
    http_methods = {"get", "post", "put", "delete", "patch", "options", "head"}

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        for dec in node.decorator_list:
            # @app.get("/path") or @router.post("/path")
            if not isinstance(dec, ast.Call):
                continue
            func = dec.func
            if not isinstance(func, ast.Attribute):
                continue
            method = func.attr.lower()
            if method not in http_methods:
                continue

            path = ""
            if dec.args:
                path = _get_string(dec.args[0])

            # Extract path params from signature
            params = []
            for arg in node.args.args:
                name = arg.arg
                if name in ("self", "request", "response", "db", "session"):
                    continue
                annotation = ""
                if arg.annotation:
                    annotation = ast.unparse(arg.annotation)
                params.append({"name": name, "annotation": annotation})

            docstring = ast.get_docstring(node) or ""

            routes.append({
                "path": path,
                "method": method,
                "function": node.name,
                "docstring": docstring,
                "params": params,
                "framework": "fastapi",
            })

    return routes


def _parse_flask_routes(tree: ast.AST, source: str) -> List[Dict]:
    """Extract Flask routes: @app.route('/path', methods=['GET'])"""
    routes = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            func = dec.func
            # @app.route or @blueprint.route
            if not (isinstance(func, ast.Attribute) and func.attr == "route"):
                continue

            path = ""
            if dec.args:
                path = _get_string(dec.args[0])

            methods = ["get"]
            for kw in dec.keywords:
                if kw.arg == "methods" and isinstance(kw.value, ast.List):
                    methods = [_get_string(el).lower() for el in kw.value.elts if _get_string(el)]

            docstring = ast.get_docstring(node) or ""

            params = []
            for arg in node.args.args:
                name = arg.arg
                if name in ("self",):
                    continue
                annotation = ""
                if arg.annotation:
                    annotation = ast.unparse(arg.annotation)
                params.append({"name": name, "annotation": annotation})

            for method in methods:
                routes.append({
                    "path": path,
                    "method": method,
                    "function": node.name,
                    "docstring": docstring,
                    "params": params,
                    "framework": "flask",
                })

    return routes


def parse_file(filepath: str) -> Dict[str, Any]:
    """Parse a Python file, return dict with routes list and detected framework."""
    source = Path(filepath).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=filepath)

    # Detect framework by imports
    framework = "unknown"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:
                names = [node.module or ""]
            for n in names:
                if "fastapi" in (n or "").lower():
                    framework = "fastapi"
                    break
                if "flask" in (n or "").lower():
                    framework = "flask"
                    break

    fastapi_routes = _parse_fastapi_routes(tree)
    flask_routes = _parse_flask_routes(tree, source)

    routes = fastapi_routes + flask_routes

    # Detect app title from variable assignment: app = FastAPI(title="...")
    title = Path(filepath).stem
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Call):
                func = node.value.func
                func_name = ""
                if isinstance(func, ast.Name):
                    func_name = func.id
                elif isinstance(func, ast.Attribute):
                    func_name = func.attr
                if func_name in ("FastAPI", "Flask"):
                    for kw in node.value.keywords:
                        if kw.arg == "title":
                            title = _get_string(kw.value) or title

    return {"title": title, "framework": framework, "routes": routes}
