# ADR-001: Plugin Architecture for Framework Support

## Status
Accepted

## Context
The OpenAPI Doc Generator needs to support multiple web frameworks (FastAPI, Flask, Django, Express, Tornado) with different route discovery mechanisms. Each framework has unique ways of defining routes, handlers, and metadata.

We need an extensible architecture that:
- Allows adding new framework support without modifying core code
- Maintains consistent interfaces across different plugins
- Enables community contributions for additional frameworks
- Provides clear separation of concerns

## Decision
Implement a plugin architecture using Python entry points with the following design:

1. **Common Interface**: All plugins implement a standard `discover_routes()` method
2. **Entry Points**: Plugins registered via `setuptools` entry points in `pyproject.toml`
3. **Dynamic Loading**: Plugins loaded at runtime based on framework detection
4. **Isolation**: Each plugin operates independently with minimal shared state

## Consequences

### Positive
- **Extensibility**: Easy to add new framework support
- **Maintainability**: Framework-specific code isolated in dedicated plugins
- **Community**: External developers can contribute plugins
- **Testing**: Each plugin can be tested independently
- **Performance**: Only needed plugins are loaded

### Negative
- **Complexity**: Additional abstraction layer increases system complexity
- **Debugging**: Plugin errors can be harder to trace
- **Dependencies**: Plugin-specific dependencies need careful management

## Alternatives Considered

### Monolithic Approach
All framework support built into core discovery engine.
- **Rejected**: Would create tight coupling and difficult maintenance

### External Package Architecture
Each framework support as separate installable packages.
- **Rejected**: Would fragment the user experience and complicate installation

### Configuration-Based Discovery
Framework patterns defined in configuration files.
- **Rejected**: Not flexible enough for complex framework patterns

## Implementation Details

```python
# Plugin interface
class FrameworkPlugin:
    def discover_routes(self, file_path: str) -> List[Route]:
        raise NotImplementedError

# Entry point registration
[project.entry-points."openapi_doc_generator.plugins"]
fastapi = "openapi_doc_generator.plugins.fastapi:FastAPIPlugin"
flask = "openapi_doc_generator.plugins.flask:FlaskPlugin"
```

## Date
2024-07-15