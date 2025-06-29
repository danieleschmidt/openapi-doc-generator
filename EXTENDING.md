# Extending OpenAPI-Doc-Generator

Plugins can add support for additional web frameworks. Each plugin implements
the `RoutePlugin` interface and either registers itself via `register_plugin`
or exposes an entry point named `openapi_doc_generator.plugins`.

```python
# my_plugin.py
from openapi_doc_generator import RoutePlugin, register_plugin, RouteInfo

class CustomPlugin(RoutePlugin):
    def detect(self, source: str) -> bool:
        return "myframework" in source.lower()

    def discover(self, app_path: str) -> list[RouteInfo]:
        # inspect app_path and return routes
        return []

register_plugin(CustomPlugin())  # optional when using entry points
```

Import your plugin before calling the CLI or API:

```python
import my_plugin  # registers plugin
from openapi_doc_generator import RouteDiscoverer

routes = RouteDiscoverer("app.py").discover()
```

Alternatively, expose the plugin via an entry point so it is discovered
automatically:

```toml
[project.entry-points."openapi_doc_generator.plugins"]
myplugin = "my_package.my_plugin:CustomPlugin"
```
