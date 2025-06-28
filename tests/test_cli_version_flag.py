import re
import pytest
from openapi_doc_generator.cli import main
from openapi_doc_generator import __version__


def test_version_option(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert re.search(__version__, output)
