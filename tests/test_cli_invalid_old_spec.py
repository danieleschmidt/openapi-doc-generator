import logging
import pytest
from openapi_doc_generator.cli import main


def test_cli_invalid_old_spec(tmp_path, caplog, capsys):
    app = tmp_path / "app.py"
    app.write_text(
        "from fastapi import FastAPI\napp=FastAPI()\n@app.get('/')\ndef hi():\n    return 'hi'\n"
    )
    bad = tmp_path / "bad.json"
    bad.write_text("{bad")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit):
            main(
                [
                    "--app",
                    str(app),
                    "--format",
                    "guide",
                    "--old-spec",
                    str(bad),
                ]
            )
    assert "invalid json" in caplog.text.lower()
    err = capsys.readouterr().err.lower()
    assert "--old-spec is not valid json" in err
