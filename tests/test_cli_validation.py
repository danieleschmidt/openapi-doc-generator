import logging
import pytest

from openapi_doc_generator.cli import main


def test_output_path_directory(tmp_path, caplog, capsys):
    app = tmp_path / "app.py"
    app.write_text(
        "from fastapi import FastAPI\napp=FastAPI()\n@app.get('/')\ndef hi():\n    return 'hi'\n"
    )
    out_dir = tmp_path / "outdir"
    out_dir.mkdir()
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit):
            main(["--app", str(app), "--output", str(out_dir)])
    err = capsys.readouterr().err
    assert "CLI004" in err


def test_tests_parent_missing(tmp_path, caplog, capsys):
    app = tmp_path / "app.py"
    app.write_text(
        "from fastapi import FastAPI\napp=FastAPI()\n@app.get('/')\ndef hi():\n    return 'hi'\n"
    )
    invalid = tmp_path / "missing" / "test_app.py"
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit):
            main(["--app", str(app), "--tests", str(invalid)])
    err = capsys.readouterr().err
    assert "CLI005" in err
