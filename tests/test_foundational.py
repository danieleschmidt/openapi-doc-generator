from openapi_doc_generator.utils import echo


def test_success():
    assert echo("hello") == "hello"


def test_edge_case_null_input():
    assert echo(None) is None
