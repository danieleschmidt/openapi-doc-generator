import json
from openapi_doc_generator.cli import main


def test_cli_custom_title_version(tmp_path, capsys):
    app = tmp_path / "app.py"
    app.write_text(
        "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/')\ndef root():\n    return 'hi'\n"
    )
    main(
        [
            "--app",
            str(app),
            "--format",
            "openapi",
            "--title",
            "My API",
            "--api-version",
            "2.0",
        ]
    )
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["info"]["title"] == "My API"
    assert data["info"]["version"] == "2.0"
