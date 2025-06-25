"""Schema inference utilities."""

from __future__ import annotations

from dataclasses import dataclass
import ast
from pathlib import Path
from typing import List


@dataclass
class FieldInfo:
    """Information about a single model field."""

    name: str
    type: str
    required: bool


@dataclass
class SchemaInfo:
    """Representation of an inferred schema/model."""

    name: str
    fields: List[FieldInfo]


class SchemaInferer:
    """Infer data models from Python source code."""

    def __init__(self, file_path: str) -> None:
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(file_path)

    # Public API
    def infer(self) -> List[SchemaInfo]:
        """Return all discovered models in the file."""
        tree = ast.parse(self.file_path.read_text(), filename=str(self.file_path))
        models: List[SchemaInfo] = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and self._is_model(node):
                models.append(self._process_class(node))
        return models

    # Internal helpers -------------------------------------------------
    def _is_model(self, node: ast.ClassDef) -> bool:
        for deco in node.decorator_list:
            if isinstance(deco, ast.Name) and deco.id == "dataclass":
                return True
            if isinstance(deco, ast.Attribute) and deco.attr == "dataclass":
                return True
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "BaseModel":
                return True
            if isinstance(base, ast.Attribute) and base.attr == "BaseModel":
                return True
        return False

    def _process_class(self, node: ast.ClassDef) -> SchemaInfo:
        fields: List[FieldInfo] = []
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                name = stmt.target.id
                type_str = ast.unparse(stmt.annotation)
                required = stmt.value is None
                fields.append(FieldInfo(name=name, type=type_str, required=required))
        return SchemaInfo(name=node.name, fields=fields)
