from openapi_doc_generator.utils import echo


def test_success_case():
    assert echo("hi") == "hi"


def test_none_case():
    assert echo(None) is None
