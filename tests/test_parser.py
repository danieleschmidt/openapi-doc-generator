import ast
import textwrap
import tempfile
import os
import pytest
from openapi_gen.parser import parse_file, _parse_fastapi_routes, _parse_flask_routes


FASTAPI_SAMPLE = textwrap.dedent('''
    from fastapi import FastAPI
    app = FastAPI(title="My API")

    @app.get("/items")
    def list_items(skip: int = 0, limit: int = 10):
        """List all items."""
        return []

    @app.post("/items")
    async def create_item(name: str, price: float):
        """Create a new item."""
        return {}

    @app.get("/items/{item_id}")
    def get_item(item_id: int):
        """Get item by id."""
        return {}
''')

FLASK_SAMPLE = textwrap.dedent('''
    from flask import Flask
    app = Flask(__name__)

    @app.route("/users", methods=["GET", "POST"])
    def users():
        """User endpoint."""
        return []

    @app.route("/users/<int:user_id>", methods=["GET"])
    def get_user(user_id):
        """Get user."""
        return {}
''')


def write_temp(content, suffix=".py"):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    f.write(content)
    f.close()
    return f.name


def test_parse_fastapi_routes():
    path = write_temp(FASTAPI_SAMPLE)
    try:
        result = parse_file(path)
        routes = result["routes"]
        assert any(r["path"] == "/items" and r["method"] == "get" for r in routes)
        assert any(r["path"] == "/items" and r["method"] == "post" for r in routes)
    finally:
        os.unlink(path)


def test_parse_fastapi_title():
    path = write_temp(FASTAPI_SAMPLE)
    try:
        result = parse_file(path)
        assert result["title"] == "My API"
        assert result["framework"] == "fastapi"
    finally:
        os.unlink(path)


def test_parse_fastapi_docstrings():
    path = write_temp(FASTAPI_SAMPLE)
    try:
        result = parse_file(path)
        routes = result["routes"]
        get_routes = [r for r in routes if r["method"] == "get" and r["path"] == "/items"]
        assert get_routes[0]["docstring"] == "List all items."
    finally:
        os.unlink(path)


def test_parse_flask_routes():
    path = write_temp(FLASK_SAMPLE)
    try:
        result = parse_file(path)
        routes = result["routes"]
        methods = {r["method"] for r in routes}
        assert "get" in methods
        assert "post" in methods
        assert result["framework"] == "flask"
    finally:
        os.unlink(path)


def test_parse_empty_file():
    path = write_temp("# empty\n")
    try:
        result = parse_file(path)
        assert result["routes"] == []
    finally:
        os.unlink(path)


def test_parse_path_params():
    path = write_temp(FASTAPI_SAMPLE)
    try:
        result = parse_file(path)
        routes = result["routes"]
        item_route = next(r for r in routes if "{item_id}" in r["path"])
        assert any(p["name"] == "item_id" for p in item_route["params"])
    finally:
        os.unlink(path)
