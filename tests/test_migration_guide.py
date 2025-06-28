from openapi_doc_generator.migration import MigrationGuideGenerator


def test_migration_guide():
    old = {"paths": {"/a": {"get": {}}, "/b": {"post": {}}}}
    new = {"paths": {"/a": {"get": {}}, "/c": {"get": {}}}}
    guide = MigrationGuideGenerator(old, new).generate_markdown()
    assert "POST /b" in guide
    assert "GET /c" in guide

def test_migration_guide_no_changes():
    spec = {"paths": {"/a": {"get": {}}}}
    guide = MigrationGuideGenerator(spec, spec).generate_markdown()
    assert "No changes detected." in guide
