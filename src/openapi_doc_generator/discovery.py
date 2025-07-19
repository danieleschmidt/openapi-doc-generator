"""Route discovery utilities for supported frameworks."""

from __future__ import annotations

from dataclasses import dataclass
import ast
from abc import ABC, abstractmethod
from importlib import metadata
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


class RoutePlugin(ABC):
    """Base class for route discovery plugins."""

    @abstractmethod
    def detect(self, source: str) -> bool:
        """Return True if the plugin can handle the given source."""
        pass

    @abstractmethod
    def discover(self, app_path: str) -> List[RouteInfo]:
        """Return discovered routes for the file."""
        pass


_PLUGINS: List[RoutePlugin] = []


def register_plugin(plugin: RoutePlugin) -> None:
    """Register a new route discovery plugin."""
    _PLUGINS.append(plugin)


def get_plugins() -> List[RoutePlugin]:
    if not _PLUGINS:
        import importlib

        importlib.import_module("openapi_doc_generator.plugins")
        for ep in metadata.entry_points(group="openapi_doc_generator.plugins"):
            try:
                plugin_cls = ep.load()
                register_plugin(plugin_cls())
            except (ImportError, ModuleNotFoundError, AttributeError) as e:  # pragma: no cover
                logging.getLogger(__name__).warning(
                    "Failed to import plugin %s: %s", ep.name, e
                )
            except (TypeError, ValueError) as e:  # pragma: no cover
                logging.getLogger(__name__).warning(
                    "Failed to instantiate plugin %s: %s", ep.name, e
                )
            except Exception as e:  # pragma: no cover - catch unexpected errors
                logging.getLogger(__name__).exception(
                    "Unexpected error loading plugin %s: %s", ep.name, e
                )
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
        
        framework = self._detect_framework(source)
        if framework == "fastapi":
            return self._discover_fastapi(source)
        elif framework == "flask":
            return self._discover_flask(source)
        elif framework == "django":
            return self._discover_django(source)
        elif framework == "express":
            return self._discover_express(source)
        else:
            raise ValueError("Unable to determine framework for route discovery")

    def _detect_framework(self, source: str) -> str | None:
        """Detect framework based on imports and patterns."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Fall back to string matching for non-Python files
            return self._detect_framework_fallback(source)
        
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
        
        # Check for framework-specific imports
        if any(imp.startswith("fastapi") for imp in imports):
            return "fastapi"
        elif any(imp.startswith("flask") for imp in imports):
            return "flask"
        elif any(imp.startswith("django") for imp in imports):
            return "django"
        
        # Fall back to content analysis
        return self._detect_framework_fallback(source)

    def _detect_framework_fallback(self, source: str) -> str | None:
        """Fallback framework detection using string matching."""
        lowered = source.lower()
        if "fastapi" in lowered or "from fastapi" in lowered:
            return "fastapi"
        elif "flask" in lowered or "from flask" in lowered:
            return "flask"
        elif "django" in lowered or "from django" in lowered:
            return "django"
        elif "express" in lowered or "require('express')" in lowered:
            return "express"
        return None

    # --- Framework specific discovery methods ---------------------------------
    def _discover_fastapi(self, source: str) -> List[RouteInfo]:
        tree = ast.parse(source, filename=str(self.app_path))
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

    def _discover_flask(self, source: str) -> List[RouteInfo]:
        tree = ast.parse(source, filename=str(self.app_path))
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

    def _discover_django(self, source: str) -> List[RouteInfo]:
        tree = ast.parse(source, filename=str(self.app_path))
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

    def _discover_express(self, source: str) -> List[RouteInfo]:
        import re

        text = source
        pattern = re.compile(r"app\.(get|post|put|patch|delete)\(['\"]([^'\"]+)['\"]")
        routes: List[RouteInfo] = []
        for match in pattern.finditer(text):
            method, path = match.group(1).upper(), match.group(2)
            name = path.strip("/").replace("/", "_") or "root"
            routes.append(RouteInfo(path=path, methods=[method], name=name))
        return routes
