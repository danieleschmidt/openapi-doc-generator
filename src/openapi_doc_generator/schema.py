"""Schema inference utilities."""

from __future__ import annotations

from dataclasses import dataclass
import ast
from pathlib import Path
import logging
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
        self._logger = logging.getLogger(self.__class__.__name__)
        if not self.file_path.exists():
            raise FileNotFoundError(file_path)

    # Public API
    def infer(self) -> List[SchemaInfo]:
        """Return all discovered models in the file."""
        self._logger.debug("Inferring models from %s", self.file_path)
        try:
            from .utils import get_cached_ast

            source_code = self.file_path.read_text()
            tree = get_cached_ast(source_code, str(self.file_path))
        except (OSError, UnicodeDecodeError) as e:
            self._logger.warning("Failed to read file %s: %s", self.file_path, e)
            return []
        except SyntaxError as e:
            self._logger.warning("Syntax error in %s: %s", self.file_path, e)
            return []

        models: List[SchemaInfo] = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and self._is_model(node):
                try:
                    models.append(self._process_class(node))
                except Exception as e:
                    self._logger.warning("Failed to process class %s: %s", node.name, e)
                    continue
        return models

    # Internal helpers -------------------------------------------------
    def _is_model(self, node: ast.ClassDef) -> bool:
        def has_dataclass_decorator() -> bool:
            return any(
                isinstance(d, (ast.Name, ast.Attribute))
                and getattr(d, "id", getattr(d, "attr", None)) == "dataclass"
                for d in node.decorator_list
            )

        def inherits_base_model() -> bool:
            return any(
                isinstance(b, (ast.Name, ast.Attribute))
                and getattr(b, "id", getattr(b, "attr", None)) == "BaseModel"
                for b in node.bases
            )

        return has_dataclass_decorator() or inherits_base_model()

    def _process_class(self, node: ast.ClassDef) -> SchemaInfo:
        fields: List[FieldInfo] = []
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                name = stmt.target.id
                type_str = ast.unparse(stmt.annotation)
                required = stmt.value is None
                fields.append(FieldInfo(name=name, type=type_str, required=required))
        return SchemaInfo(name=node.name, fields=fields)
