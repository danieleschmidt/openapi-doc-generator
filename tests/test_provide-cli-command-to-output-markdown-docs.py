import logging
import pytest
from openapi_doc_generator.cli import main
from openapi_doc_generator.discovery import _PLUGINS


@pytest.fixture(autouse=True)
def clean_plugins():
    """Ensure plugins are cleared before and after each test."""
    _PLUGINS.clear()
    yield
    _PLUGINS.clear()


def test_success(tmp_path, capsys):
    app = tmp_path / "app.py"
    app.write_text(
        "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/hello')\ndef hello():\n    return 'hi'\n"
    )
    main(["--app", str(app)])
    out = capsys.readouterr().out
    assert "# API" in out
    assert "/hello" in out


def test_edge_case_invalid_input(caplog, capsys):
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit):
            main(["--app", "nonexistent.py"])
    err = capsys.readouterr().err
    assert "CLI001" in err
