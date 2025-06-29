# Contributing

Thank you for helping improve **OpenAPI-Doc-Generator**!

## Setup
1. Install the project in editable mode:
   ```bash
   pip install -e .
   pip install ruff bandit pytest coverage radon
   ```

## Running Tests
Execute the full test suite with:
```bash
coverage run -m pytest -q
coverage html
```

## Linting
Format and lint the code before submitting a pull request:
```bash
ruff format --check .
ruff check .
bandit -r src
radon cc src -n B
```
