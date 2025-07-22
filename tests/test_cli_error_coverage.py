"""Tests to improve CLI error path coverage."""

import pytest
from openapi_doc_generator.cli import main


def test_missing_old_spec_file_error(tmp_path, capsys):
    """Test CLI error when old spec file doesn't exist."""
    app = tmp_path / "app.py"
    app.write_text(
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n"
        "@app.get('/')\n"
        "def hi():\n"
        "    return 'hi'\n"
    )

    non_existent_spec = tmp_path / "non_existent.json"
    # Make sure the file doesn't exist
    assert not non_existent_spec.exists()

    # Test with non-existent old spec file
    with pytest.raises(SystemExit):
        main(
            [
                "--app",
                str(app),
                "--format",
                "guide",
                "--old-spec",
                str(non_existent_spec),
            ]
        )

    err = capsys.readouterr().err
    assert "CLI002" in err
    assert "not found" in err
