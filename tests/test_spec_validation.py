from openapi_doc_generator.validator import SpecValidator
import pytest


def test_valid_spec():
    spec = {
        "openapi": "3.0.0",
        "paths": {"/hi": {"get": {"summary": "Say hi"}}},
    }
    suggestions = SpecValidator().validate(spec)
    assert suggestions == []


def test_spec_suggestions():
    spec = {"openapi": "2.0", "paths": {"/hi": {"get": {}}}}
    suggestions = SpecValidator().validate(spec)
    assert "OpenAPI version should be 3.x" in suggestions
    assert "Operation 'get /hi' is missing summary" in suggestions


def test_invalid_input():
    with pytest.raises(TypeError):
        SpecValidator().validate(None)
