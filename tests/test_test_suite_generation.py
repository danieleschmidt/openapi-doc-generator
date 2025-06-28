from openapi_doc_generator.testsuite import TestSuiteGenerator
from openapi_doc_generator.discovery import RouteInfo
from openapi_doc_generator.documentator import DocumentationResult


def test_generate_pytest():
    result = DocumentationResult(
        routes=[RouteInfo(path="/hi", methods=["GET"], name="hi")], schemas=[]
    )
    code = TestSuiteGenerator(result).generate_pytest()
    assert "def test_hi_get()" in code
    assert "requests.get('http://localhost/hi')" in code
