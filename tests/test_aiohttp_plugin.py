from openapi_doc_generator.discovery import RouteDiscoverer


def test_aiohttp_plugin(tmp_path):
    app = tmp_path / "app.py"
    app.write_text(
        "from aiohttp import web\n"
        "app = web.Application()\n"
        "app.router.add_get('/hello', lambda r: r)\n"
    )
    routes = RouteDiscoverer(str(app)).discover()
    assert len(routes) == 1
    r = routes[0]
    assert r.path == "/hello"
    assert r.methods == ["GET"]
