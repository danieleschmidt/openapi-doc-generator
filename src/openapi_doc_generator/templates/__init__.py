"""Markdown templates for API documentation."""

from importlib import resources


def load_template(name: str) -> str:
    """Return the contents of a template."""
    try:
        return resources.files(__package__).joinpath(name).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Template '{name}' not found") from exc
