from openapi_doc_generator.spec import OpenAPISpecGenerator, _type_to_openapi
from openapi_doc_generator.discovery import RouteInfo
from openapi_doc_generator.schema import SchemaInfo, FieldInfo


def test_generate_basic_spec():
    routes = [RouteInfo(path="/hello", methods=["GET"], name="hello")]
    fields = [FieldInfo(name="id", type="int", required=True)]
    schemas = [SchemaInfo(name="User", fields=fields)]
    spec = OpenAPISpecGenerator(
        routes=routes, schemas=schemas, title="Test API", version="1.0"
    ).generate()
    assert spec["info"]["title"] == "Test API"
    assert "/hello" in spec["paths"]
    assert spec["paths"]["/hello"]["get"]["summary"] == "hello"
    assert (
        spec["components"]["schemas"]["User"]["properties"]["id"]["type"] == "integer"
    )


def test_type_to_openapi_conversions():
    """Test all type conversion edge cases for complete coverage."""
    # Basic type mappings
    assert _type_to_openapi("int") == "integer"
    assert _type_to_openapi("float") == "number"
    assert _type_to_openapi("str") == "string"
    assert _type_to_openapi("bool") == "boolean"
    
    # Case insensitive
    assert _type_to_openapi("INT") == "integer"
    assert _type_to_openapi("Float") == "number"
    
    # List/Sequence types (lines 23-24)
    assert _type_to_openapi("list") == "array"
    assert _type_to_openapi("List[str]") == "array"
    assert _type_to_openapi("sequence") == "array"
    assert _type_to_openapi("Sequence[int]") == "array"
    
    # Dict/Mapping types (lines 25-26)
    assert _type_to_openapi("dict") == "object"
    assert _type_to_openapi("Dict[str, int]") == "object"
    assert _type_to_openapi("mapping") == "object"
    assert _type_to_openapi("Mapping[str, Any]") == "object"
    
    # Default fallback (line 27)
    assert _type_to_openapi("unknown_type") == "string"
    assert _type_to_openapi("CustomClass") == "string"
    assert _type_to_openapi("") == "string"
