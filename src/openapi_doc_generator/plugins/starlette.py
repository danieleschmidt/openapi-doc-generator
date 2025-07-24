"""Starlette route discovery plugin."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

from ..discovery import RouteInfo, RoutePlugin, register_plugin

logger = logging.getLogger(__name__)


class StarlettePlugin(RoutePlugin):
    """Discover routes defined with Starlette framework."""

    def detect(self, source: str) -> bool:
        """Return True if the source contains Starlette imports."""
        return "starlette" in source.lower()

    def discover(self, app_path: str) -> List[RouteInfo]:
        """Discover routes from a Starlette application file."""
        try:
            source_path = Path(app_path)
            if not source_path.exists():
                logger.warning(f"File not found: {app_path}")
                return []
            
            content = source_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            routes = []
            function_docs = self._extract_function_docstrings(tree)
            
            # Find route definitions
            for node in ast.walk(tree):
                if self._is_route_assignment(node):
                    extracted_routes = self._extract_routes_from_assignment(node, function_docs)
                    routes.extend(extracted_routes)
            
            return routes
            
        except (OSError, SyntaxError, UnicodeDecodeError) as e:
            logger.error(f"Error parsing {app_path}: {e}")
            return []

    def _extract_function_docstrings(self, tree: ast.AST) -> Dict[str, str]:
        """Extract docstrings from function definitions."""
        docs = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name:
                docstring = ast.get_docstring(node)
                if docstring:
                    docs[node.name] = docstring
        return docs

    def _is_route_assignment(self, node: ast.AST) -> bool:
        """Check if an assignment node defines routes."""
        if not isinstance(node, ast.Assign):
            return False
        
        # Look for assignments to 'routes' variable
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == 'routes':
                return True
        return False

    def _extract_routes_from_assignment(self, node: ast.Assign, function_docs: Dict[str, str]) -> List[RouteInfo]:
        """Extract route information from a routes assignment."""
        routes = []
        
        if isinstance(node.value, ast.List):
            for item in node.value.elts:
                route_info = self._parse_route_item(item, function_docs)
                if route_info:
                    routes.append(route_info)
        
        return routes

    def _parse_route_item(self, item: ast.AST, function_docs: Dict[str, str], mount_prefix: str = "") -> Optional[RouteInfo]:
        """Parse a single route item (Route, Mount, or WebSocketRoute)."""
        if not isinstance(item, ast.Call):
            return None
        
        # Get the route class name
        route_class = self._get_call_name(item)
        if not route_class:
            return None
        
        if route_class == "Route":
            return self._parse_route_call(item, function_docs, mount_prefix)
        elif route_class == "WebSocketRoute":
            return self._parse_websocket_route_call(item, function_docs, mount_prefix)
        elif route_class == "Mount":
            return self._parse_mount_call(item, function_docs, mount_prefix)
        
        return None

    def _get_call_name(self, call: ast.Call) -> Optional[str]:
        """Get the name of a function call."""
        if isinstance(call.func, ast.Name):
            return call.func.id
        elif isinstance(call.func, ast.Attribute):
            return call.func.attr
        return None

    def _parse_route_call(self, call: ast.Call, function_docs: Dict[str, str], mount_prefix: str = "") -> Optional[RouteInfo]:
        """Parse a Route() call."""
        if len(call.args) < 2:
            return None
        
        # Extract path
        path_arg = call.args[0]
        if not isinstance(path_arg, ast.Constant) or not isinstance(path_arg.value, str):
            return None
        path = mount_prefix + path_arg.value
        
        # Extract function name
        func_arg = call.args[1]
        func_name = self._get_function_name(func_arg)
        
        # Extract methods from keyword arguments
        methods = ['GET']  # Default method
        for keyword in call.keywords:
            if keyword.arg == 'methods' and isinstance(keyword.value, ast.List):
                methods = []
                for method_node in keyword.value.elts:
                    if isinstance(method_node, ast.Constant) and isinstance(method_node.value, str):
                        methods.append(method_node.value)
        
        # Get docstring if available
        docstring = function_docs.get(func_name) if func_name else None
        
        return RouteInfo(
            path=path,
            methods=methods,
            name=func_name or f"route_{abs(hash(path))}",
            docstring=docstring
        )

    def _parse_websocket_route_call(self, call: ast.Call, function_docs: Dict[str, str], mount_prefix: str = "") -> Optional[RouteInfo]:
        """Parse a WebSocketRoute() call."""
        if len(call.args) < 2:
            return None
        
        # Extract path
        path_arg = call.args[0]
        if not isinstance(path_arg, ast.Constant) or not isinstance(path_arg.value, str):
            return None
        path = mount_prefix + path_arg.value
        
        # Extract function name
        func_arg = call.args[1]
        func_name = self._get_function_name(func_arg)
        
        # Get docstring if available
        docstring = function_docs.get(func_name) if func_name else None
        
        return RouteInfo(
            path=path,
            methods=['WEBSOCKET'],
            name=func_name or f"websocket_{abs(hash(path))}",
            docstring=docstring
        )

    def _parse_mount_call(self, call: ast.Call, function_docs: Dict[str, str], mount_prefix: str = "") -> List[RouteInfo]:
        """Parse a Mount() call and extract nested routes."""
        if len(call.args) < 1:
            return []
        
        # Extract mount path
        path_arg = call.args[0]
        if not isinstance(path_arg, ast.Constant) or not isinstance(path_arg.value, str):
            return []
        mount_path = mount_prefix + path_arg.value
        
        # Find routes keyword argument
        routes = []
        for keyword in call.keywords:
            if keyword.arg == 'routes' and isinstance(keyword.value, ast.List):
                for item in keyword.value.elts:
                    route_info = self._parse_route_item(item, function_docs, mount_path)
                    if route_info:
                        if isinstance(route_info, list):
                            routes.extend(route_info)
                        else:
                            routes.append(route_info)
        
        return routes

    def _get_function_name(self, func_node: ast.AST) -> Optional[str]:
        """Extract function name from various AST node types."""
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Lambda):
            return f"lambda_{abs(hash(ast.dump(func_node)))}"
        elif isinstance(func_node, ast.Attribute):
            return func_node.attr
        return None


# Register the plugin
register_plugin(StarlettePlugin())