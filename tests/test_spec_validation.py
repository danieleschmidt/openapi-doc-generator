from openapi_doc_generator.validator import SpecValidator
import pytest


def test_valid_spec():
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Test API", "version": "1.0.0"},
        "paths": {"/hi": {"get": {"summary": "Say hi", "responses": {"200": {"description": "Success"}}}}},
    }
    suggestions = SpecValidator().validate(spec)
    assert suggestions == []


def test_spec_suggestions():
    spec = {"openapi": "2.0", "paths": {"/hi": {"get": {}}}}
    suggestions = SpecValidator().validate(spec)
    assert any("3.x" in s for s in suggestions)  # Updated to match new error message
    assert "Operation 'get /hi' is missing summary" in suggestions


def test_invalid_input():
    with pytest.raises(TypeError):
        SpecValidator().validate(None)


def test_comprehensive_openapi_validation():
    """Test comprehensive OpenAPI 3.0 specification validation."""
    validator = SpecValidator()
    
    # Test missing required fields
    minimal_spec = {"openapi": "3.0.0"}
    suggestions = validator.validate(minimal_spec)
    assert any("info" in s for s in suggestions)
    assert any("paths" in s for s in suggestions)
    
    # Test info section validation
    spec_missing_info_fields = {
        "openapi": "3.0.0",
        "info": {},  # Missing title and version
        "paths": {}
    }
    suggestions = validator.validate(spec_missing_info_fields)
    assert any("title" in s for s in suggestions)
    assert any("version" in s for s in suggestions)
    
    # Test invalid OpenAPI version formats
    invalid_version_spec = {
        "openapi": "invalid",
        "info": {"title": "Test", "version": "1.0.0"},
        "paths": {}
    }
    suggestions = validator.validate(invalid_version_spec)
    assert any("version" in s.lower() for s in suggestions)


def test_operation_validation():
    """Test operation-level validation rules."""
    validator = SpecValidator()
    
    # Test operation with no responses
    spec = {
        "openapi": "3.0.0", 
        "info": {"title": "Test", "version": "1.0.0"},
        "paths": {
            "/test": {
                "get": {
                    "summary": "Test endpoint"
                    # Missing responses
                }
            }
        }
    }
    suggestions = validator.validate(spec)
    assert any("responses" in s for s in suggestions)
    
    # Test invalid HTTP methods
    invalid_method_spec = {
        "openapi": "3.0.0",
        "info": {"title": "Test", "version": "1.0.0"}, 
        "paths": {
            "/test": {
                "invalid_method": {"summary": "Test"}
            }
        }
    }
    suggestions = validator.validate(invalid_method_spec)
    assert any("method" in s.lower() for s in suggestions)


def test_components_validation():
    """Test components section validation."""
    validator = SpecValidator()
    
    # Test components with empty schemas
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Test", "version": "1.0.0"},
        "paths": {},
        "components": {
            "schemas": {
                "EmptySchema": {}  # Schema with no properties or type
            }
        }
    }
    suggestions = validator.validate(spec)
    assert any("schema" in s.lower() for s in suggestions)


def test_security_validation():
    """Test security-related validation rules."""
    validator = SpecValidator()
    
    # Test missing security schemes when security is referenced
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Test", "version": "1.0.0"},
        "paths": {
            "/secure": {
                "get": {
                    "summary": "Secure endpoint",
                    "security": [{"api_key": []}],
                    "responses": {"200": {"description": "Success"}}
                }
            }
        }
        # Missing components.securitySchemes
    }
    suggestions = validator.validate(spec)
    assert any("security" in s.lower() for s in suggestions)


def test_edge_cases():
    """Test edge cases and error conditions for complete coverage."""
    validator = SpecValidator()
    
    # Test non-string OpenAPI version
    spec = {"openapi": 3.0, "info": {"title": "Test", "version": "1.0.0"}, "paths": {}}
    suggestions = validator.validate(spec)
    assert any("string" in s for s in suggestions)
    
    # Test non-dict info
    spec = {"openapi": "3.0.0", "info": "invalid", "paths": {}}
    suggestions = validator.validate(spec)
    assert any("object" in s for s in suggestions)
    
    # Test non-string info fields
    spec = {"openapi": "3.0.0", "info": {"title": 123, "version": 456}, "paths": {}}
    suggestions = validator.validate(spec)
    assert any("string" in s for s in suggestions)
    
    # Test non-dict paths
    spec = {"openapi": "3.0.0", "info": {"title": "Test", "version": "1.0.0"}, "paths": "invalid"}
    suggestions = validator.validate(spec)
    assert any("object" in s for s in suggestions)
    
    # Test non-dict path operations
    spec = {
        "openapi": "3.0.0", 
        "info": {"title": "Test", "version": "1.0.0"},
        "paths": {"/test": "invalid"}
    }
    suggestions = validator.validate(spec)
    assert any("object" in s for s in suggestions)
    
    # Test non-dict operation
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Test", "version": "1.0.0"},
        "paths": {"/test": {"get": "invalid"}}
    }
    suggestions = validator.validate(spec)
    assert any("object" in s for s in suggestions)
    
    # Test non-dict responses  
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Test", "version": "1.0.0"},
        "paths": {"/test": {"get": {"summary": "Test", "responses": "invalid"}}}
    }
    suggestions = validator.validate(spec)
    assert any("object" in s for s in suggestions)
    
    # Test empty responses
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Test", "version": "1.0.0"},
        "paths": {"/test": {"get": {"summary": "Test", "responses": {}}}}
    }
    suggestions = validator.validate(spec)
    assert any("no response" in s for s in suggestions)
    
    # Test non-dict components
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Test", "version": "1.0.0"},
        "paths": {},
        "components": "invalid"
    }
    suggestions = validator.validate(spec)
    assert any("object" in s for s in suggestions)
    
    # Test non-dict schema
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Test", "version": "1.0.0"},
        "paths": {},
        "components": {"schemas": {"TestSchema": "invalid"}}
    }
    suggestions = validator.validate(spec)
    assert any("object" in s for s in suggestions)
