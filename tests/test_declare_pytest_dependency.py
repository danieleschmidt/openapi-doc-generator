import tomllib


def test_pyproject_contains_pytest_dependency():
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    dev = data.get("project", {}).get("optional-dependencies", {}).get("dev")
    assert dev is not None
    assert "pytest" in dev
