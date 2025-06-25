import pytest

from openapi_doc_generator.markdown import MarkdownGenerator


def test_success():
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Sample API", "version": "1.0"},
        "paths": {
            "/hello": {
                "get": {"summary": "Say hello"}
            }
        },
        "components": {"schemas": {}}
    }
    markdown = MarkdownGenerator().generate(spec)
    assert "# Sample API" in markdown
    assert "## Say hello" in markdown
    assert "/hello" in markdown


def test_edge_case_invalid_input():
    with pytest.raises(TypeError):
        MarkdownGenerator().generate(None)
