import json
from openapi_doc_generator.graphql import GraphQLSchema
from openapi_doc_generator.cli import main


def test_graphql_introspection(tmp_path, capsys):
    schema = tmp_path / "schema.graphql"
    schema.write_text("""type Query { hello: String }""")
    result = GraphQLSchema(str(schema)).introspect()
    assert "__schema" in result

    main(["--app", str(schema), "--format", "graphql"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "__schema" in data
