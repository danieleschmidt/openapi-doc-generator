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


def test_xss_prevention_in_title():
    """Verify that potentially malicious HTML in title is properly escaped."""
    malicious_spec = {
        "openapi": "3.0.0",
        "info": {"title": "<script>alert('XSS')</script>Malicious API"},
        "paths": {},
    }
    html = PlaygroundGenerator().generate(malicious_spec)

    # The malicious script should be HTML escaped
    assert "&lt;script&gt;alert(&#x27;XSS&#x27;)&lt;/script&gt;" in html
    assert "<script>alert('XSS')</script>" not in html


def test_json_serialization_safety():
    """Verify that JSON serialization is safe from injection."""
    spec_with_quotes = {
        "openapi": "3.0.0",
        "info": {"title": "API with \"quotes\" and 'apostrophes'"},
        "paths": {},
    }
    html = PlaygroundGenerator().generate(spec_with_quotes)

    # JSON should be properly escaped
    assert '"API with \\"quotes\\" and \'apostrophes\'"' in html
