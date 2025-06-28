# OpenAPI-Doc-Generator

Agent that parses FastAPI / Express routes and emits OpenAPI 3 spec plus markdown docs, using prompt-templated reflection.

## Features
- Automatic route discovery and analysis for FastAPI, Express, Flask, and Django
- Intelligent schema inference from code annotations and examples for dataclasses and Pydantic models
- Generates comprehensive OpenAPI 3.0 specifications
- Creates human-readable markdown documentation with examples
- Interactive API playground generation with Swagger UI
- Validates existing OpenAPI specs and suggests improvements
- Supports GraphQL schema introspection
- Automated test suite generation for discovered routes
- Continuous documentation deployment via GitHub Pages
- Generates API deprecation and migration guides
- Customizable API title and version via CLI options

## Quick Start
```bash
pip install -e .
openapi-doc-generator --app ./app.py --format markdown --output API.md
openapi-doc-generator --app ./app.py --format openapi --output openapi.json
openapi-doc-generator --app ./app.py --format openapi --title "My API" --api-version 2.0 --output openapi.json
openapi-doc-generator --app ./app.py --format html --output playground.html
openapi-doc-generator --app ./schema.graphql --format graphql --output schema.json
openapi-doc-generator --app ./app.py --tests tests/test_app.py
openapi-doc-generator --app ./app.py --format guide --old-spec old.json --output MIGRATION.md
openapi-doc-generator --version
```

Documentation for the example app in `examples/app.py` is automatically built
and published to GitHub Pages whenever changes are pushed to `main`.

## Testing
Run the test suite with:
```bash
pytest -q
```

## Usage
```python
from openapi_doc_generator import APIDocumentator

generator = APIDocumentator()
docs = generator.analyze_app("./app.py")

# Generate OpenAPI spec
spec = docs.generate_openapi_spec()
with open("openapi.json", "w") as f:
    json.dump(spec, f, indent=2)

# Validate OpenAPI spec
from openapi_doc_generator import SpecValidator
issues = SpecValidator().validate(spec)
if issues:
    print("Suggestions:\n" + "\n".join(issues))

# Generate markdown docs
markdown = docs.generate_markdown()
with open("API.md", "w") as f:
    f.write(markdown)
```

## Framework Support
- **FastAPI**: Full type annotation support, automatic model extraction
- **Express.js**: Route parsing, JSDoc integration, TypeScript support
- **Flask**: Decorator analysis, Flask-RESTful integration
- **Django REST Framework**: Serializer introspection, viewset analysis

## Generated Documentation
- Complete endpoint documentation with parameters, responses, examples
- Interactive API explorer (Swagger UI integration)
- Code samples in multiple languages (Python, JavaScript, cURL)
- Authentication and error handling documentation
- Rate limiting and versioning information

## Configuration
```yaml
# config.yaml
frameworks:
  fastapi:
    include_internal: false
    example_generation: true
  express:
    typescript_support: true
    middleware_docs: true

output:
  formats: ["openapi", "markdown", "postman"]
  include_examples: true
  authentication_docs: true
```

## Advanced Features
- **Reflection-based Analysis**: Uses LLM to understand complex business logic
- **Example Generation**: Creates realistic API examples based on schema
- **Version Comparison**: Tracks API changes across versions
- **Integration Testing**: Validates generated docs against actual API responses
- **Automated Test Generation**: Produces pytest suites for discovered routes

## Roadmap
All roadmap items have been completed:
1. ✅ Add GraphQL schema support
2. ✅ Implement automated testing suite generation
3. ✅ Build CI/CD integration for documentation updates
4. ✅ Add API deprecation and migration guides

## License
MIT
