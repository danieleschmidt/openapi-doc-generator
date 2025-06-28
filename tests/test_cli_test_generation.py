from openapi_doc_generator.cli import main


def test_cli_generates_tests(tmp_path):
    app = tmp_path / "app.py"
    app.write_text(
        "from fastapi import FastAPI\napp=FastAPI()\n@app.get('/hi')\ndef hi():\n    return 'hi'\n"
    )
    out_file = tmp_path / "test_app.py"
    main(["--app", str(app), "--tests", str(out_file)])
    code = out_file.read_text()
    assert "test_hi_get" in code
