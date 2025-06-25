import pytest

from openapi_doc_generator.templates import load_template


def test_success():
    template = load_template("api.md.jinja")
    assert "{{" in template and "}}" in template


def test_edge_case_invalid_input():
    with pytest.raises(FileNotFoundError):
        load_template("missing-template.jinja")
