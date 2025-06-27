import tomllib


def test_cli_script_entry_present():
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    scripts = data.get("project", {}).get("scripts", {})
    assert scripts.get("openapi-doc-generator") == "openapi_doc_generator.cli:main"
