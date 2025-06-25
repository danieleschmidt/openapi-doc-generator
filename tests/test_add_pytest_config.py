import pathlib
import configparser

def test_file_exists():
    assert pathlib.Path("pytest.ini").is_file()

def test_contains_header():
    config = configparser.ConfigParser()
    config.read("pytest.ini")
    assert "pytest" in config
    assert config["pytest"].get("testpaths")
