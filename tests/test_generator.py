import pytest
from openapi_gen.generator import generate_openapi, to_yaml, _python_type_to_openapi


def test_type_map():
    assert _python_type_to_openapi("int") == "integer"
    assert _python_type_to_openapi("str") == "string"
    assert _python_type_to_openapi("float") == "number"
    assert _python_type_to_openapi("") == "string"


def test_generate_basic():
    parsed = {
        "title": "Test API",
        "framework": "fastapi",
        "routes": [
            {"path": "/hello", "method": "get", "function": "hello", "docstring": "Say hello", "params": []},
        ]
    }
    spec = generate_openapi(parsed)
    assert spec["openapi"] == "3.0.3"
    assert spec["info"]["title"] == "Test API"
    assert "/hello" in spec["paths"]
    assert "get" in spec["paths"]["/hello"]


def test_generate_yaml_output():
    parsed = {"title": "API", "framework": "fastapi", "routes": [
        {"path": "/x", "method": "post", "function": "create_x", "docstring": "", "params": []}
    ]}
    spec = generate_openapi(parsed)
    y = to_yaml(spec)
    assert "openapi:" in y
    assert "/x:" in y
