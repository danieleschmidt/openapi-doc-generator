import textwrap
from openapi_doc_generator.discovery import RouteDiscoverer


def test_flask_route_discovery(tmp_path):
    app_file = tmp_path / "app.py"
    app_file.write_text(
        textwrap.dedent(
            '''
            from flask import Flask

            app = Flask(__name__)

            @app.route("/hi", methods=["GET", "POST"])
            def hi():
                """Say hi"""
                return "hi"
            '''
        )
    )
    routes = RouteDiscoverer(str(app_file)).discover()
    assert len(routes) == 1
    route = routes[0]
    assert route.path == "/hi"
    assert set(route.methods) == {"GET", "POST"}
    assert route.name == "hi"
    assert route.docstring == "Say hi"
