"""Route discovery utilities for supported frameworks."""

from __future__ import annotations

from dataclasses import dataclass
import ast
from pathlib import Path
import logging
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .plugins import RoutePlugin


@dataclass
class RouteInfo:
    """Information about a discovered API route."""

    path: str
    methods: List[str]
    name: str
    docstring: str | None = None


class RoutePlugin:
    """Base class for route discovery plugins."""

    def detect(self, source: str) -> bool:
        """Return True if the plugin can handle the given source."""
        raise NotImplementedError

    def discover(self, app_path: str) -> List[RouteInfo]:
        """Return discovered routes for the file."""
        raise NotImplementedError


_PLUGINS: List[RoutePlugin] = []


def register_plugin(plugin: RoutePlugin) -> None:
    """Register a new route discovery plugin."""
    _PLUGINS.append(plugin)


def get_plugins() -> List[RoutePlugin]:
    if not _PLUGINS:
        import importlib

        importlib.import_module("openapi_doc_generator.plugins")
    return list(_PLUGINS)


class RouteDiscoverer:
    """Discover routes from application source files."""

    def __init__(self, app_path: str) -> None:
        self.app_path = Path(app_path)
        self._logger = logging.getLogger(self.__class__.__name__)
        if not self.app_path.exists():
            raise FileNotFoundError(f"{app_path} does not exist")

    def discover(self) -> List[RouteInfo]:
        """Discover routes based on detected framework."""
        self._logger.debug("Scanning %s for routes", self.app_path)
        source = self.app_path.read_text()
        for plugin in get_plugins():
            if plugin.detect(source):
                self._logger.debug("Using plugin %s", plugin.__class__.__name__)
                return plugin.discover(str(self.app_path))
        lowered = source.lower()
        if "fastapi" in lowered:
            return self._discover_fastapi()
        if "flask" in lowered:
            return self._discover_flask()
        if "django" in lowered:
            return self._discover_django()
        if "express" in lowered:
            return self._discover_express()
        raise ValueError("Unable to determine framework for route discovery")

    # --- Framework specific discovery methods ---------------------------------
    def _discover_fastapi(self) -> List[RouteInfo]:
        tree = ast.parse(self.app_path.read_text(), filename=str(self.app_path))
        routes: List[RouteInfo] = []

        class Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                for deco in node.decorator_list:
                    if isinstance(deco, ast.Call) and isinstance(
                        deco.func, ast.Attribute
                    ):
                        method = deco.func.attr
                        if method in {"get", "post", "put", "patch", "delete"}:
                            if (
                                isinstance(deco.func.value, ast.Name)
                                and deco.func.value.id == "app"
                            ):
                                path = ""
                                if deco.args and isinstance(deco.args[0], ast.Constant):
                                    path = str(deco.args[0].value)
                                doc = ast.get_docstring(node)
                                routes.append(
                                    RouteInfo(
                                        path=path,
                                        methods=[method.upper()],
                                        name=node.name,
                                        docstring=doc,
                                    )
                                )
                self.generic_visit(node)

        Visitor().visit(tree)
        return routes

    def _discover_flask(self) -> List[RouteInfo]:
        tree = ast.parse(self.app_path.read_text(), filename=str(self.app_path))
        routes: List[RouteInfo] = []

        class Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                for deco in node.decorator_list:
                    if isinstance(deco, ast.Call) and isinstance(
                        deco.func, ast.Attribute
                    ):
                        if (
                            isinstance(deco.func.value, ast.Name)
                            and deco.func.value.id == "app"
                            and deco.func.attr == "route"
                        ):
                            path = ""
                            if deco.args and isinstance(deco.args[0], ast.Constant):
                                path = str(deco.args[0].value)
                            methods: List[str] = ["GET"]
                            for kw in deco.keywords:
                                if kw.arg == "methods" and isinstance(
                                    kw.value, (ast.List, ast.Tuple)
                                ):
                                    methods = []
                                    for elt in kw.value.elts:
                                        if isinstance(elt, ast.Constant):
                                            methods.append(str(elt.value).upper())
                            doc = ast.get_docstring(node)
                            routes.append(
                                RouteInfo(
                                    path=path,
                                    methods=methods,
                                    name=node.name,
                                    docstring=doc,
                                )
                            )
                self.generic_visit(node)

        Visitor().visit(tree)
        return routes

    def _discover_django(self) -> List[RouteInfo]:
        tree = ast.parse(self.app_path.read_text(), filename=str(self.app_path))
        routes: List[RouteInfo] = []

        class Visitor(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call) -> None:
                if isinstance(node.func, ast.Name) and node.func.id in {
                    "path",
                    "re_path",
                }:
                    if node.args and isinstance(node.args[0], ast.Constant):
                        path_value = str(node.args[0].value)
                        name = ""
                        if len(node.args) > 1:
                            target = node.args[1]
                            if isinstance(target, ast.Attribute):
                                name = target.attr
                            elif isinstance(target, ast.Name):
                                name = target.id
                        routes.append(
                            RouteInfo(path=path_value, methods=["GET"], name=name)
                        )
                self.generic_visit(node)

        Visitor().visit(tree)
        return routes

    def _discover_express(self) -> List[RouteInfo]:
        import re

        text = self.app_path.read_text()
        pattern = re.compile(r"app\.(get|post|put|patch|delete)\(['\"]([^'\"]+)['\"]")
        routes: List[RouteInfo] = []
        for match in pattern.finditer(text):
            method, path = match.group(1).upper(), match.group(2)
            name = path.strip("/").replace("/", "_") or "root"
            routes.append(RouteInfo(path=path, methods=[method], name=name))
        return routes
