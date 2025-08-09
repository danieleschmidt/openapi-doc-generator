"""Tornado route discovery plugin."""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import List, Optional

from ..discovery import RouteInfo, RoutePlugin, register_plugin

logger = logging.getLogger(__name__)


class TornadoPlugin(RoutePlugin):
    """Discover routes defined with Tornado framework."""

    def detect(self, source: str) -> bool:
        """Return True if the source contains Tornado imports."""
        return "tornado" in source.lower()

    def discover(self, app_path: str) -> List[RouteInfo]:
        """Discover routes from a Tornado application file."""
        try:
            source_path = Path(app_path)
            if not source_path.exists():
                logger.warning(f"File not found: {app_path}")
                return []

            content = source_path.read_text(encoding='utf-8')
            tree = ast.parse(content)

            # Find handler classes and their methods
            handlers = self._find_tornado_handlers(tree)

            # Find Application instantiation with route mappings
            routes = self._find_tornado_routes(tree, handlers)

            return routes

        except Exception as e:
            logger.warning(f"Error parsing Tornado app {app_path}: {e}")
            return []

    def _find_tornado_handlers(self, tree: ast.AST) -> dict[str, dict]:
        """Find Tornado RequestHandler classes and their HTTP methods."""
        handlers = {}

        for node in ast.walk(tree):
            if (isinstance(node, ast.ClassDef) and
                self._is_tornado_handler(node)):

                handler_info = {
                    'methods': [],
                    'docstring': ast.get_docstring(node)
                }

                # Find HTTP method handlers (get, post, put, delete, etc.)
                for item in node.body:
                    if (isinstance(item, ast.FunctionDef) and
                        self._is_http_method(item.name)):
                        handler_info['methods'].append(item.name.upper())

                handlers[node.name] = handler_info

        return handlers

    def _is_tornado_handler(self, node: ast.ClassDef) -> bool:
        """Check if a class inherits from tornado.web.RequestHandler."""
        for base in node.bases:
            if isinstance(base, ast.Attribute):
                # Handle tornado.web.RequestHandler
                if (isinstance(base.value, ast.Attribute) and
                    isinstance(base.value.value, ast.Name) and
                    base.value.value.id == 'tornado' and
                    base.value.attr == 'web' and
                    base.attr == 'RequestHandler'):
                    return True
            elif isinstance(base, ast.Name):
                # Handle direct RequestHandler import
                if base.id == 'RequestHandler':
                    return True
        return False

    def _is_http_method(self, method_name: str) -> bool:
        """Check if method name corresponds to an HTTP method."""
        http_methods = {
            'get', 'post', 'put', 'delete', 'patch',
            'head', 'options', 'trace'
        }
        return method_name.lower() in http_methods

    def _find_tornado_routes(self, tree: ast.AST, handlers: dict) -> List[RouteInfo]:
        """Find Tornado Application route definitions."""
        routes = []

        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and
                self._is_tornado_application(node)):

                # Extract route patterns from Application arguments
                routes.extend(self._extract_route_patterns(node, handlers))

        return routes

    def _is_tornado_application(self, node: ast.Call) -> bool:
        """Check if node is a tornado.web.Application call."""
        if isinstance(node.func, ast.Attribute):
            # Handle tornado.web.Application
            if (isinstance(node.func.value, ast.Attribute) and
                isinstance(node.func.value.value, ast.Name) and
                node.func.value.value.id == 'tornado' and
                node.func.value.attr == 'web' and
                node.func.attr == 'Application'):
                return True
        elif isinstance(node.func, ast.Name):
            # Handle direct Application import
            if node.func.id == 'Application':
                return True
        return False

    def _extract_route_patterns(self, app_node: ast.Call, handlers: dict) -> List[RouteInfo]:
        """Extract route patterns from Application constructor."""
        routes = []

        # Application first argument should be list of route tuples
        if not app_node.args:
            return routes

        routes_arg = app_node.args[0]
        if not isinstance(routes_arg, ast.List):
            return routes

        for route_item in routes_arg.elts:
            if isinstance(route_item, ast.Tuple) and len(route_item.elts) >= 2:
                # Extract pattern and handler
                pattern_node = route_item.elts[0]
                handler_node = route_item.elts[1]

                pattern = self._extract_string_value(pattern_node)
                handler_name = self._extract_handler_name(handler_node)

                if pattern and handler_name and handler_name in handlers:
                    handler_info = handlers[handler_name]

                    route = RouteInfo(
                        path=pattern,
                        methods=handler_info['methods'] or ['GET'],  # Default to GET if no methods found
                        name=handler_name,
                        docstring=handler_info['docstring']
                    )
                    routes.append(route)

        return routes

    def _extract_string_value(self, node: ast.expr) -> Optional[str]:
        """Extract string value from AST node."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        elif isinstance(node, ast.Str):  # Python < 3.8 compatibility
            return node.s
        return None

    def _extract_handler_name(self, node: ast.expr) -> Optional[str]:
        """Extract handler class name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return None


# Register the plugin
register_plugin(TornadoPlugin())
