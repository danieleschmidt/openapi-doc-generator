"""Built-in and third-party plugins for route discovery."""

from .aiohttp import AioHTTPPlugin  # noqa: F401 - register plugin on import
from .starlette import StarlettePlugin  # noqa: F401 - register plugin on import
from .tornado import TornadoPlugin  # noqa: F401 - register plugin on import

__all__ = ["AioHTTPPlugin", "StarlettePlugin", "TornadoPlugin"]
