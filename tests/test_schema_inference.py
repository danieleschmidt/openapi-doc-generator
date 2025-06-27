import textwrap
from openapi_doc_generator.schema import SchemaInferer


def test_dataclass_inference(tmp_path):
    file = tmp_path / "models.py"
    file.write_text(
        textwrap.dedent(
            """
            from dataclasses import dataclass

            @dataclass
            class User:
                id: int
                name: str = 'anon'
            """
        )
    )
    schemas = SchemaInferer(str(file)).infer()
    assert len(schemas) == 1
    schema = schemas[0]
    assert schema.name == "User"
    assert len(schema.fields) == 2
    id_field = next(f for f in schema.fields if f.name == "id")
    assert id_field.type == "int"
    assert id_field.required
    name_field = next(f for f in schema.fields if f.name == "name")
    assert not name_field.required


def test_pydantic_basemodel_inference(tmp_path):
    file = tmp_path / "models.py"
    file.write_text(
        textwrap.dedent(
            """
            from pydantic import BaseModel

            class Item(BaseModel):
                id: int
                price: float = 0.0
            """
        )
    )
    schemas = SchemaInferer(str(file)).infer()
    assert len(schemas) == 1
    schema = schemas[0]
    assert schema.name == "Item"
    assert len(schema.fields) == 2
