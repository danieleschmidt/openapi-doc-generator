"""Starlette route discovery plugin."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Optional, Dict
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
            
            # Extract all route assignments and function docstrings
            route_assignments = {}
            function_docs = self._extract_function_docstrings(tree)
            
            # First pass: collect all route assignments
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.endswith('routes'):
                            route_assignments[target.id] = node.value
            
            # Second pass: process main routes assignment
            routes = []
            main_routes = route_assignments.get('routes')
            if main_routes:
                routes.extend(self._process_route_list(main_routes, route_assignments, function_docs))
            
            return routes
            
        except (OSError, SyntaxError, UnicodeDecodeError) as e:
            logger.error(f"Error parsing {app_path}: {e}")
            return []

    def _extract_function_docstrings(self, tree: ast.AST) -> Dict[str, str]:
        """Extract docstrings from function definitions."""
        docs = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name:
                docstring = ast.get_docstring(node)
                if docstring:
                    docs[node.name] = docstring
        return docs

    def _process_route_list(self, route_list_node: ast.AST, route_assignments: Dict[str, ast.AST], 
                           function_docs: Dict[str, str], mount_prefix: str = "") -> List[RouteInfo]:
        """Process a list of route definitions."""
        routes = []
        
        if isinstance(route_list_node, ast.List):
            for item in route_list_node.elts:
                route_info = self._parse_route_item(item, route_assignments, function_docs, mount_prefix)
                if isinstance(route_info, list):
                    routes.extend(route_info)
                elif route_info:
                    routes.append(route_info)
        elif isinstance(route_list_node, ast.Name):
            # Handle variable reference
            referenced_list = route_assignments.get(route_list_node.id)
            if referenced_list:
                routes.extend(self._process_route_list(referenced_list, route_assignments, function_docs, mount_prefix))
        
        return routes

    def _parse_route_item(self, item: ast.AST, route_assignments: Dict[str, ast.AST], 
                         function_docs: Dict[str, str], mount_prefix: str = "") -> Optional[RouteInfo]:
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
            return self._parse_mount_call(item, route_assignments, function_docs, mount_prefix)
        
        return None

    def _get_call_name(self, call: ast.Call) -> Optional[str]:
        """Get the name of a function call."""
        if isinstance(call.func, ast.Name):
            return call.func.id
        elif isinstance(call.func, ast.Attribute):
            return call.func.attr
        return None

    def _extract_path_from_route_call(self, call: ast.Call, mount_prefix: str) -> Optional[str]:
        """Extract path from route call arguments."""
        if len(call.args) < 2:
            return None
        
        path_arg = call.args[0]
        if not isinstance(path_arg, ast.Constant) or not isinstance(path_arg.value, str):
            return None
        
        return mount_prefix + path_arg.value

    def _extract_methods_from_keywords(self, call: ast.Call) -> List[str]:
        """Extract HTTP methods from route call keywords."""
        methods = ['GET']  # Default method
        for keyword in call.keywords:
            if keyword.arg == 'methods' and isinstance(keyword.value, ast.List):
                methods = []
                for method_node in keyword.value.elts:
                    if isinstance(method_node, ast.Constant) and isinstance(method_node.value, str):
                        methods.append(method_node.value)
        return methods

    def _parse_route_call(self, call: ast.Call, function_docs: Dict[str, str], mount_prefix: str = "") -> Optional[RouteInfo]:
        """Parse a Route() call."""
        # Extract path
        path = self._extract_path_from_route_call(call, mount_prefix)
        if path is None:
            return None
        
        # Extract function name
        func_arg = call.args[1]
        func_name = self._get_function_name(func_arg)
        
        # Extract methods from keyword arguments
        methods = self._extract_methods_from_keywords(call)
        
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

    def _parse_mount_call(self, call: ast.Call, route_assignments: Dict[str, ast.AST], 
                         function_docs: Dict[str, str], mount_prefix: str = "") -> List[RouteInfo]:
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
            if keyword.arg == 'routes':
                routes.extend(self._process_route_list(keyword.value, route_assignments, function_docs, mount_path))
        
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