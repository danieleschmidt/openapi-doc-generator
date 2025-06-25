import pytest
from openapi_doc_generator.cli import main


def test_success(tmp_path, capsys):
    app = tmp_path / "app.py"
    app.write_text(
        "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/hello')\ndef hello():\n    return 'hi'\n"
    )
    main(["--app", str(app)])
    out = capsys.readouterr().out
    assert "# API" in out
    assert "/hello" in out


def test_edge_case_invalid_input():
    with pytest.raises(SystemExit):
        main(["--app", "nonexistent.py"])
