import pathlib

def test_readme_mentions_pytest():
    text = pathlib.Path("README.md").read_text().lower()
    assert "pytest" in text
