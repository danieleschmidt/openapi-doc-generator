"""Built-in and third-party plugins for route discovery."""

from .aiohttp import AioHTTPPlugin  # noqa: F401 - register plugin on import

__all__ = ["AioHTTPPlugin"]
