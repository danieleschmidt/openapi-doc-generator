import sys
import argparse
from .parser import parse_file
from .generator import generate_openapi, to_yaml


def main():
    parser = argparse.ArgumentParser(
        description="Generate OpenAPI spec from Python FastAPI/Flask source files."
    )
    parser.add_argument("file", nargs="+", help="Python source file(s) to parse")
    parser.add_argument("--version", default="3.0.3", help="OpenAPI version (default: 3.0.3)")
    parser.add_argument("--title", help="API title override")
    args = parser.parse_args()

    all_routes = []
    title = args.title or "API"
    framework = "unknown"

    for f in args.file:
        parsed = parse_file(f)
        all_routes.extend(parsed["routes"])
        if args.title is None:
            title = parsed.get("title", title)
        framework = parsed.get("framework", framework)

    combined = {"title": title, "framework": framework, "routes": all_routes}
    spec = generate_openapi(combined, version=args.version)
    print(to_yaml(spec))


if __name__ == "__main__":
    main()
