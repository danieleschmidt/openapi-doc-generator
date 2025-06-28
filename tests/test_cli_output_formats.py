import json
from openapi_doc_generator.cli import main


def test_openapi_output(tmp_path, capsys):
    app = tmp_path / "app.py"
    app.write_text(
        "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/hi')\ndef hi():\n    return 'hi'\n"
    )
    main(["--app", str(app), "--format", "openapi"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "/hi" in data["paths"]


def test_html_output(tmp_path, capsys):
    app = tmp_path / "app.py"
    app.write_text(
        "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/hi')\ndef hi():\n    return 'hi'\n"
    )
    main(["--app", str(app), "--format", "html"])
    out = capsys.readouterr().out
    assert "swagger-ui" in out
