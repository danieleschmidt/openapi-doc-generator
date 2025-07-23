"""Route discovery utilities for supported frameworks."""

from __future__ import annotations

from dataclasses import dataclass
import ast
from abc import ABC, abstractmethod
from importlib import metadata
from pathlib import Path
import logging
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .plugins import RoutePlugin


@dataclass
class RouteInfo:
    """Information about a discovered API route."""

    path: str
    methods: List[str]
    name: str
    docstring: Optional[str] = None


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
    """Get all registered route discovery plugins."""
    if not _PLUGINS:
        import importlib
        importlib.import_module("openapi_doc_generator.plugins")
        
        for ep in metadata.entry_points(group="openapi_doc_generator.plugins"):
            _load_single_plugin(ep)
    
    return list(_PLUGINS)


def _load_single_plugin(ep) -> None:
    """Load a single plugin entry point with consolidated error handling."""
    logger = logging.getLogger(__name__)
    
    try:
        plugin_cls = ep.load()
        register_plugin(plugin_cls())
    except (ImportError, ModuleNotFoundError, AttributeError) as e:  # pragma: no cover
        logger.warning("Failed to import plugin %s: %s", ep.name, e)
    except (TypeError, ValueError) as e:  # pragma: no cover
        logger.warning("Failed to instantiate plugin %s: %s", ep.name, e)
    except Exception as e:  # pragma: no cover
        # Catch-all for any other plugin loading errors
        logger.exception("Unexpected plugin loading error %s: %s", ep.name, e)


class RouteDiscoverer:
    """Discover routes from application source files."""

    def __init__(self, app_path: str) -> None:
        self.app_path = Path(app_path)
        self._logger = logging.getLogger(self.__class__.__name__)
        if not self.app_path.exists():
            raise FileNotFoundError(f"{app_path} does not exist")

    def discover(self) -> List[RouteInfo]:
        """Discover routes based on detected framework."""
        from .utils import measure_performance

        @measure_performance("route_discovery")
        def _discover_routes():
            self._logger.debug("Scanning %s for routes", self.app_path)
            source = self.app_path.read_text()
            for plugin in get_plugins():
                if plugin.detect(source):
                    self._logger.debug("Using plugin %s", plugin.__class__.__name__)
                    return plugin.discover(str(self.app_path))

            framework = self._detect_framework(source)
            if framework == "fastapi":
                routes = self._discover_fastapi(source)
            elif framework == "flask":
                routes = self._discover_flask(source)
            elif framework == "django":
                routes = self._discover_django(source)
            elif framework == "express":
                routes = self._discover_express(source)
            else:
                raise ValueError("Unable to determine framework for route discovery")

            # Log route count for performance tracking
            self._logger.info(
                f"Discovered {len(routes)} routes",
                extra={
                    "operation": "route_discovery",
                    "route_count": len(routes),
                    "framework": framework,
                },
            )
            return routes

        return _discover_routes()

    def _extract_imports_from_ast(self, source: str) -> Optional[set[str]]:
        """Extract import names from AST, return None if parsing fails."""
        try:
            from .utils import get_cached_ast
            tree = get_cached_ast(source, str(self.app_path))
        except SyntaxError:
            return None

        imports = set()
        for node in ast.walk(tree):
            self._process_import_node(node, imports)
        return imports
    
    def _process_import_node(self, node: ast.AST, imports: set[str]) -> None:
        """Process an AST node and extract import information."""
        if isinstance(node, ast.Import):
            self._handle_direct_import(node, imports)
        elif isinstance(node, ast.ImportFrom):
            self._handle_from_import(node, imports)
    
    def _handle_direct_import(self, node: ast.Import, imports: set[str]) -> None:
        """Handle direct import statements (import module)."""
        for alias in node.names:
            imports.add(alias.name)
    
    def _handle_from_import(self, node: ast.ImportFrom, imports: set[str]) -> None:
        """Handle from-import statements (from module import ...)."""
        if node.module:
            imports.add(node.module)

    def _detect_framework_from_imports(self, imports: set[str]) -> Optional[str]:
        """Detect framework from import names using priority order."""
        framework_patterns = [
            ("fastapi", "fastapi"),
            ("flask", "flask"),
            ("django", "django"),
        ]

        for framework, pattern in framework_patterns:
            if any(imp.startswith(pattern) for imp in imports):
                return framework
        return None

    def _detect_framework(self, source: str) -> Optional[str]:
        """Detect framework based on imports and patterns."""
        from .utils import measure_performance

        @measure_performance("framework_detection")
        def _detect():
            imports = self._extract_imports_from_ast(source)

            if imports is not None:
                framework = self._detect_framework_from_imports(imports)
                if framework:
                    return framework

            # Fall back to content analysis
            return self._detect_framework_fallback(source)

        return _detect()

    def _detect_framework_fallback(self, source: str) -> Optional[str]:
        """Fallback framework detection using string matching."""
        lowered = source.lower()
        
        # Framework detection patterns - more maintainable and testable
        framework_patterns = [
            ("fastapi", self._detect_fastapi_patterns),
            ("flask", self._detect_flask_patterns),
            ("django", self._detect_django_patterns),
            ("express", self._detect_express_patterns),
        ]
        
        for framework, detector in framework_patterns:
            if detector(lowered):
                return framework
        
        return None
    
    def _detect_fastapi_patterns(self, lowered_source: str) -> bool:
        """Detect FastAPI framework patterns."""
        return "fastapi" in lowered_source or "from fastapi" in lowered_source
    
    def _detect_flask_patterns(self, lowered_source: str) -> bool:
        """Detect Flask framework patterns."""
        return "flask" in lowered_source or "from flask" in lowered_source
    
    def _detect_django_patterns(self, lowered_source: str) -> bool:
        """Detect Django framework patterns."""
        return "django" in lowered_source or "from django" in lowered_source
    
    def _detect_express_patterns(self, lowered_source: str) -> bool:
        """Detect Express.js framework patterns."""
        return "express" in lowered_source or "require('express')" in lowered_source

    # --- Framework specific discovery methods ---------------------------------
    def _discover_fastapi(self, source: str) -> List[RouteInfo]:
        from .utils import get_cached_ast

        tree = get_cached_ast(source, str(self.app_path))
        routes: List[RouteInfo] = []
        discoverer = self

        class Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                for deco in node.decorator_list:
                    route_info = discoverer._extract_fastapi_route(deco, node)
                    if route_info:
                        routes.append(route_info)
                self.generic_visit(node)

        Visitor().visit(tree)
        return routes

    def _extract_fastapi_route(
        self, deco: ast.expr, node: ast.FunctionDef
    ) -> Optional[RouteInfo]:
        """Extract FastAPI route information from decorator and function node."""
        if not (isinstance(deco, ast.Call) and isinstance(deco.func, ast.Attribute)):
            return None
            
        method = deco.func.attr
        if method not in {"get", "post", "put", "patch", "delete"}:
            return None
            
        if not self._is_app_decorator(deco.func):
            return None
            
        path = self._extract_path_from_args(deco.args)
        doc = ast.get_docstring(node)
        
        return RouteInfo(
            path=path,
            methods=[method.upper()],
            name=node.name,
            docstring=doc,
        )

    def _discover_flask(self, source: str) -> List[RouteInfo]:
        from .utils import get_cached_ast

        tree = get_cached_ast(source, str(self.app_path))
        routes: List[RouteInfo] = []
        discoverer = self

        class Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                for deco in node.decorator_list:
                    route_info = discoverer._extract_flask_route(deco, node)
                    if route_info:
                        routes.append(route_info)
                self.generic_visit(node)

        Visitor().visit(tree)
        return routes

    def _extract_flask_route(
        self, deco: ast.expr, node: ast.FunctionDef
    ) -> Optional[RouteInfo]:
        """Extract Flask route information from decorator and function node."""
        if not self._is_flask_route_decorator(deco):
            return None
            
        path = self._extract_path_from_args(deco.args)
        methods = self._extract_flask_methods(deco.keywords)
        doc = ast.get_docstring(node)
        
        return RouteInfo(
            path=path,
            methods=methods,
            name=node.name,
            docstring=doc,
        )
        
    def _is_flask_route_decorator(self, deco: ast.expr) -> bool:
        """Check if decorator is a Flask route decorator."""
        if not (isinstance(deco, ast.Call) and isinstance(deco.func, ast.Attribute)):
            return False
            
        return (
            isinstance(deco.func.value, ast.Name)
            and deco.func.value.id == "app"
            and deco.func.attr == "route"
        )
        
    def _extract_flask_methods(self, keywords: List[ast.keyword]) -> List[str]:
        """Extract HTTP methods from Flask route keyword arguments."""
        methods: List[str] = ["GET"]  # Default Flask method
        
        for kw in keywords:
            if kw.arg == "methods" and isinstance(kw.value, (ast.List, ast.Tuple)):
                methods = []
                for elt in kw.value.elts:
                    if isinstance(elt, ast.Constant):
                        methods.append(str(elt.value).upper())
                        
        return methods

    def _discover_django(self, source: str) -> List[RouteInfo]:
        from .utils import get_cached_ast

        tree = get_cached_ast(source, str(self.app_path))
        routes: List[RouteInfo] = []
        discoverer = self

        class Visitor(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call) -> None:
                route_info = discoverer._extract_django_route(node)
                if route_info:
                    routes.append(route_info)
                self.generic_visit(node)

        Visitor().visit(tree)
        return routes
        
    def _extract_django_route(self, node: ast.Call) -> Optional[RouteInfo]:
        """Extract Django route information from Call node."""
        if not self._is_django_path_call(node):
            return None
            
        if not (node.args and isinstance(node.args[0], ast.Constant)):
            return None
            
        path_value = str(node.args[0].value)
        name = self._extract_django_view_name(node.args)
        
        return RouteInfo(path=path_value, methods=["GET"], name=name)
        
    def _is_django_path_call(self, node: ast.Call) -> bool:
        """Check if Call node is a Django path or re_path call."""
        return (
            isinstance(node.func, ast.Name) 
            and node.func.id in {"path", "re_path"}
        )
        
    def _extract_django_view_name(self, args: List[ast.expr]) -> str:
        """Extract view name from Django path arguments."""
        if len(args) > 1:
            target = args[1]
            if isinstance(target, ast.Attribute):
                return target.attr
            elif isinstance(target, ast.Name):
                return target.id
        return ""
        
    def _is_app_decorator(self, func: ast.Attribute) -> bool:
        """Check if decorator function is an app decorator."""
        return (
            isinstance(func.value, ast.Name)
            and func.value.id == "app"
        )
        
    def _extract_path_from_args(self, args: List[ast.expr]) -> str:
        """Extract path string from function arguments."""
        if args and isinstance(args[0], ast.Constant):
            return str(args[0].value)
        return ""

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
