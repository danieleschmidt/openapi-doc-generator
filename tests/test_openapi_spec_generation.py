from openapi_doc_generator.spec import OpenAPISpecGenerator
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
