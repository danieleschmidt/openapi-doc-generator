from openapi_doc_generator.discovery import RouteDiscoverer


def test_express_route_discovery(tmp_path):
    app = tmp_path / "app.js"
    app.write_text(
        "const express = require('express');\n"
        "const app = express();\n"
        "app.get('/hi', (req, res) => res.send('hi'));\n"
        "app.post('/submit', (req, res) => res.send('ok'));\n"
    )
    routes = RouteDiscoverer(str(app)).discover()
    assert len(routes) == 2
    paths = {r.path for r in routes}
    assert "/hi" in paths
    assert "/submit" in paths
