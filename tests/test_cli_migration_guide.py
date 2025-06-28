import json
from openapi_doc_generator.cli import main


def test_cli_migration(tmp_path, capsys):
    app = tmp_path / "app.py"
    app.write_text(
        "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/new')\ndef new():\n    return 'new'\n"
    )
    old_spec = tmp_path / "old.json"
    old_spec.write_text(json.dumps({"paths": {"/old": {"get": {}}}}))

    main([
        "--app",
        str(app),
        "--format",
        "guide",
        "--old-spec",
        str(old_spec),
    ])
    out = capsys.readouterr().out
    assert "GET /old" in out
    assert "GET /new" in out
