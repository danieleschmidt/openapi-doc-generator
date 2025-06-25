import pytest
from openapi_doc_generator.documentator import DocumentationResult
from openapi_doc_generator.discovery import RouteInfo


def test_success():
    routes = [RouteInfo(path="/hello", methods=["GET"], name="hello")]
    result = DocumentationResult(routes=routes, schemas=[])
    markdown = result.generate_markdown(title="Sample API")
    assert "# Sample API" in markdown
    assert "/hello" in markdown


def test_edge_case_invalid_input():
    result = DocumentationResult(routes=None, schemas=None)
    with pytest.raises(TypeError):
        result.generate_markdown()
