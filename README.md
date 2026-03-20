# openapi-doc-generator

Auto-generates OpenAPI 3.0 specs from Python FastAPI and Flask source code. Uses Python AST parsing — no runtime execution of your code needed.

## Usage

```bash
pip install PyYAML
python -m openapi_gen.cli app.py > openapi.yaml
# or after pip install -e .
openapi-gen app.py
openapi-gen app.py --title "My API" --version 3.1.0
```

## Features

- Parses FastAPI `@app.get/post/put/delete` decorators
- Parses Flask `@app.route` decorators with methods
- Extracts function signatures as query/path parameters
- Extracts docstrings as descriptions
- Detects path parameters from `{param}` in path strings
- Outputs valid OpenAPI 3.0.3 YAML

## Example

```python
# app.py
from fastapi import FastAPI
app = FastAPI(title="My API")

@app.get("/items")
def list_items(skip: int = 0, limit: int = 10):
    """List all items."""
    return []

@app.get("/items/{item_id}")
def get_item(item_id: int):
    """Get item by id."""
    return {}
```

```bash
$ openapi-gen app.py
openapi: 3.0.3
info:
  title: My API
  version: 1.0.0
paths:
  /items:
    get:
      operationId: list_items
      summary: List all items.
      ...
```

## Testing

```bash
pip install pytest PyYAML
pytest tests/ -v
```
