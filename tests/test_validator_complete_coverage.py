"""Comprehensive tests to achieve 100% coverage for validator.py missing lines."""

import pytest
from openapi_doc_generator.validator import SpecValidator


class TestValidatorCompleteCoverage:
    """Test specific missing lines in validator.py for 100% coverage."""
    
    def test_missing_openapi_field(self):
        """Test validation when 'openapi' field is missing entirely (line 58)."""
        validator = SpecValidator()
        
        # Spec without 'openapi' field at all
        spec = {
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {}
        }
        
        suggestions = validator.validate(spec)
        
        # This should trigger line 58: suggestions.append("Missing required 'openapi' field")
        assert any("Missing required 'openapi' field" in s for s in suggestions)
        assert any("openapi" in s for s in suggestions)
    
    def test_path_with_no_operations(self):
        """Test validation when a path has no operations (lines 120-121)."""
        validator = SpecValidator()
        
        # Path that exists but has no HTTP method operations
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {
                "/empty-path": {}  # No operations like get, post, etc.
            }
        }
        
        suggestions = validator.validate(spec)
        
        # This should trigger lines 120-121: path has no operations
        assert any("has no operations" in s for s in suggestions)
        assert any("/empty-path" in s for s in suggestions)
    
    def test_components_with_none_schemas(self):
        """Test _validate_schemas when schemas is None (line 200)."""
        validator = SpecValidator()
        
        # Create a spec with components but schemas is None
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {},
            "components": {
                "schemas": None  # This should trigger the None check
            }
        }
        
        suggestions = validator.validate(spec)
        
        # The method should return early (line 200) and not crash
        # No specific assertion needed - just ensure no exception is raised
        assert isinstance(suggestions, list)
    
    def test_components_with_non_dict_schemas(self):
        """Test _validate_schemas when schemas is not a dict (line 200)."""
        validator = SpecValidator()
        
        # Create a spec with components but schemas is not a dict
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {},
            "components": {
                "schemas": "not-a-dict"  # This should trigger the not isinstance(schemas, dict) check
            }
        }
        
        suggestions = validator.validate(spec)
        
        # The method should return early (line 200) and not crash
        # No specific assertion needed - just ensure no exception is raised
        assert isinstance(suggestions, list)
    
    def test_schema_with_no_valid_properties(self):
        """Test schema validation for schemas without valid properties (lines 220-221)."""
        validator = SpecValidator()
        
        # Schema that has neither 'type', 'properties', nor '$ref'
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {},
            "components": {
                "schemas": {
                    "InvalidSchema": {
                        # Has some fields but not the required ones for proper schema definition
                        "description": "A schema without type, properties, or $ref",
                        "example": "some example"
                    }
                }
            }
        }
        
        suggestions = validator.validate(spec)
        
        # This should trigger lines 220-221: schema should define type, properties, or $ref
        schema_suggestions = [s for s in suggestions if "InvalidSchema" in s and "should define" in s]
        assert len(schema_suggestions) > 0
        assert any("type" in s and "properties" in s and "$ref" in s for s in schema_suggestions)
    
    def test_has_valid_schema_properties_method(self):
        """Test _has_valid_schema_properties method directly (line 230)."""
        validator = SpecValidator()
        
        # Test schema with 'type' - should return True
        schema_with_type = {"type": "string"}
        assert validator._has_valid_schema_properties(schema_with_type) is True
        
        # Test schema with 'properties' - should return True
        schema_with_properties = {"properties": {"name": {"type": "string"}}}
        assert validator._has_valid_schema_properties(schema_with_properties) is True
        
        # Test schema with '$ref' - should return True
        schema_with_ref = {"$ref": "#/components/schemas/OtherSchema"}
        assert validator._has_valid_schema_properties(schema_with_ref) is True
        
        # Test schema with none of the required properties - should return False
        schema_invalid = {"description": "No type, properties, or $ref"}
        assert validator._has_valid_schema_properties(schema_invalid) is False
        
        # Test schema with all properties - should return True (tests the 'or' logic)
        schema_with_all = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "$ref": "#/components/schemas/Base"
        }
        assert validator._has_valid_schema_properties(schema_with_all) is True
    
    def test_path_operations_edge_case(self):
        """Test edge case where path operations dict is empty."""
        validator = SpecValidator()
        
        # Path with empty operations dict (different from no operations key)
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {
                "/test": {}  # Empty dict - no HTTP methods
            }
        }
        
        suggestions = validator.validate(spec)
        
        # Should detect that the path has no operations
        path_suggestions = [s for s in suggestions if "/test" in s and "no operations" in s]
        assert len(path_suggestions) > 0
    
    def test_comprehensive_schema_validation_coverage(self):
        """Test comprehensive schema validation to ensure all edge cases are covered."""
        validator = SpecValidator()
        
        # Test multiple schemas with different validation issues
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {},
            "components": {
                "schemas": {
                    "ValidSchemaWithType": {
                        "type": "object"
                    },
                    "ValidSchemaWithProperties": {
                        "properties": {
                            "name": {"type": "string"}
                        }
                    },
                    "ValidSchemaWithRef": {
                        "$ref": "#/components/schemas/ValidSchemaWithType"
                    },
                    "InvalidSchemaNoDefiningProps": {
                        "description": "This schema has no type, properties, or $ref"
                    }
                }
            }
        }
        
        suggestions = validator.validate(spec)
        
        # Only the invalid schema should generate suggestions
        invalid_schema_suggestions = [s for s in suggestions if "InvalidSchemaNoDefiningProps" in s]
        assert len(invalid_schema_suggestions) > 0
        
        # Valid schemas should not generate suggestions
        valid_suggestions = [s for s in suggestions 
                           if any(name in s for name in ["ValidSchemaWithType", "ValidSchemaWithProperties", "ValidSchemaWithRef"])]
        assert len(valid_suggestions) == 0