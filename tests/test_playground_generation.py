from openapi_doc_generator.playground import PlaygroundGenerator
import pytest


def test_generate_playground_html():
    spec = {"openapi": "3.0.0", "info": {"title": "Demo"}, "paths": {}}
    html = PlaygroundGenerator().generate(spec)
    assert "swagger-ui" in html
    assert "SwaggerUIBundle" in html
    assert "Demo" in html


def test_invalid_input():
    with pytest.raises(TypeError):
        PlaygroundGenerator().generate(None)
