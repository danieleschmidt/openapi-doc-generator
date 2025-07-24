import types
from importlib import metadata
import pytest

from openapi_doc_generator.discovery import RoutePlugin, get_plugins, _PLUGINS


@pytest.fixture(autouse=True)
def clean_plugins():
    """Ensure plugins are cleared before and after each test."""
    _PLUGINS.clear()
    yield
    _PLUGINS.clear()


class Dummy(RoutePlugin):
    def detect(self, source: str) -> bool:
        return False

    def discover(self, app_path: str):
        return []


def test_entry_point_loading(monkeypatch):

    def fake_entry_points(*, group: str):
        assert group == "openapi_doc_generator.plugins"
        return [types.SimpleNamespace(name="dummy", load=lambda: Dummy)]

    monkeypatch.setattr(metadata, "entry_points", fake_entry_points)
    plugins = get_plugins()
    assert any(isinstance(p, Dummy) for p in plugins)
