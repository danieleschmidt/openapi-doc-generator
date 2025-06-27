import textwrap
from openapi_doc_generator.discovery import RouteDiscoverer


def test_django_route_discovery(tmp_path):
    urls = tmp_path / "urls.py"
    urls.write_text(
        textwrap.dedent(
            """
            from django.urls import path
            from . import views

            urlpatterns = [
                path('home/', views.home, name='home'),
                path('about/', views.about),
            ]
            """
        )
    )
    routes = RouteDiscoverer(str(urls)).discover()
    assert len(routes) == 2
    paths = {r.path for r in routes}
    assert "home/" in paths
    assert "about/" in paths
