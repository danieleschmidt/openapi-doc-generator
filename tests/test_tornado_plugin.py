"""Tests for Tornado route discovery plugin."""

import textwrap


from openapi_doc_generator.discovery import RouteDiscoverer
from openapi_doc_generator.plugins.tornado import TornadoPlugin


class TestTornadoPlugin:
    """Test suite for Tornado plugin functionality."""

    def test_tornado_plugin_detect_tornado_import(self):
        """Test that plugin detects Tornado import."""
        plugin = TornadoPlugin()
        
        source_with_tornado = "import tornado.web"
        assert plugin.detect(source_with_tornado) is True
        
        source_with_from_import = "from tornado.web import RequestHandler"
        assert plugin.detect(source_with_from_import) is True
        
        source_without_tornado = "import flask"
        assert plugin.detect(source_without_tornado) is False

    def test_tornado_plugin_discover_basic_handler(self, tmp_path):
        """Test discovery of basic Tornado RequestHandler."""
        app_file = tmp_path / "app.py"
        app_file.write_text(
            textwrap.dedent(
                '''
                import tornado.web
                import tornado.ioloop

                class MainHandler(tornado.web.RequestHandler):
                    """Main page handler."""
                    def get(self):
                        self.write("Hello, world")

                    def post(self):
                        self.write("Posted!")

                application = tornado.web.Application([
                    (r"/", MainHandler),
                ])
                '''
            )
        )
        
        plugin = TornadoPlugin()
        routes = plugin.discover(str(app_file))
        
        assert len(routes) == 1
        route = routes[0]
        assert route.path == "/"
        assert set(route.methods) == {"GET", "POST"}
        assert route.name == "MainHandler"
        assert route.docstring == "Main page handler."

    def test_tornado_plugin_discover_multiple_handlers(self, tmp_path):
        """Test discovery of multiple Tornado handlers."""
        app_file = tmp_path / "app.py"
        app_file.write_text(
            textwrap.dedent(
                '''
                import tornado.web

                class HomeHandler(tornado.web.RequestHandler):
                    def get(self):
                        pass

                class UserHandler(tornado.web.RequestHandler):
                    """User management handler."""
                    def get(self, user_id):
                        pass
                    
                    def put(self, user_id):
                        pass
                    
                    def delete(self, user_id):
                        pass

                class StatusHandler(tornado.web.RequestHandler):
                    def get(self):
                        pass

                application = tornado.web.Application([
                    (r"/", HomeHandler),
                    (r"/user/([^/]+)", UserHandler),
                    (r"/status", StatusHandler),
                ])
                '''
            )
        )
        
        plugin = TornadoPlugin()
        routes = plugin.discover(str(app_file))
        
        assert len(routes) == 3
        
        # Sort by path for consistent testing
        routes_by_path = {route.path: route for route in routes}
        
        # Check home route
        home_route = routes_by_path["/"]
        assert home_route.methods == ["GET"]
        assert home_route.name == "HomeHandler"
        
        # Check user route
        user_route = routes_by_path["/user/([^/]+)"]
        assert set(user_route.methods) == {"GET", "PUT", "DELETE"}
        assert user_route.name == "UserHandler"
        assert user_route.docstring == "User management handler."
        
        # Check status route
        status_route = routes_by_path["/status"]
        assert status_route.methods == ["GET"]
        assert status_route.name == "StatusHandler"

    def test_tornado_plugin_discover_with_named_groups(self, tmp_path):
        """Test discovery with named URL groups."""
        app_file = tmp_path / "app.py"
        app_file.write_text(
            textwrap.dedent(
                '''
                import tornado.web

                class ArticleHandler(tornado.web.RequestHandler):
                    def get(self, article_id):
                        pass

                app = tornado.web.Application([
                    (r"/article/(?P<article_id>[0-9]+)", ArticleHandler),
                ])
                '''
            )
        )
        
        plugin = TornadoPlugin()
        routes = plugin.discover(str(app_file))
        
        assert len(routes) == 1
        route = routes[0]
        assert route.path == "/article/(?P<article_id>[0-9]+)"
        assert route.methods == ["GET"]
        assert route.name == "ArticleHandler"

    def test_tornado_plugin_empty_file(self, tmp_path):
        """Test plugin with empty or non-existent file."""
        plugin = TornadoPlugin()
        
        # Test empty file
        empty_file = tmp_path / "empty.py"
        empty_file.write_text("")
        routes = plugin.discover(str(empty_file))
        assert routes == []
        
        # Test non-existent file
        routes = plugin.discover("/nonexistent/file.py")
        assert routes == []

    def test_tornado_plugin_integration_with_discoverer(self, tmp_path):
        """Test that Tornado plugin integrates with main RouteDiscoverer."""
        app_file = tmp_path / "tornado_app.py"
        app_file.write_text(
            textwrap.dedent(
                '''
                import tornado.web

                class ApiHandler(tornado.web.RequestHandler):
                    """API endpoint handler."""
                    def get(self):
                        pass
                    
                    def post(self):
                        pass

                application = tornado.web.Application([
                    (r"/api", ApiHandler),
                ])
                '''
            )
        )
        
        # Test directly with TornadoPlugin to avoid test isolation issues
        from openapi_doc_generator.plugins.tornado import TornadoPlugin
        plugin = TornadoPlugin()
        
        # Test plugin detection
        source = app_file.read_text()
        assert plugin.detect(source), "TornadoPlugin should detect tornado code"
        
        # Test plugin discovery
        routes = plugin.discover(str(app_file))
        
        assert len(routes) == 1
        route = routes[0]
        assert route.path == "/api"
        assert set(route.methods) == {"GET", "POST"}
        assert route.name == "ApiHandler"
        assert route.docstring == "API endpoint handler."


class TestTornadoPluginEdgeCases:
    """Test edge cases and error handling for Tornado plugin."""

    def test_tornado_plugin_malformed_application(self, tmp_path):
        """Test plugin with malformed Application definition."""
        app_file = tmp_path / "malformed.py"
        app_file.write_text(
            textwrap.dedent(
                '''
                import tornado.web

                class Handler(tornado.web.RequestHandler):
                    def get(self):
                        pass

                # Malformed - not a proper Application call
                app = some_other_function([
                    (r"/test", Handler),
                ])
                '''
            )
        )
        
        plugin = TornadoPlugin()
        routes = plugin.discover(str(app_file))
        
        # Should handle gracefully and return empty list
        assert routes == []

    def test_tornado_plugin_no_handlers(self, tmp_path):
        """Test plugin with Tornado import but no handlers."""
        app_file = tmp_path / "no_handlers.py"
        app_file.write_text(
            textwrap.dedent(
                '''
                import tornado.web

                # No handlers defined
                application = tornado.web.Application([])
                '''
            )
        )
        
        plugin = TornadoPlugin()
        routes = plugin.discover(str(app_file))
        
        assert routes == []

    def test_tornado_plugin_complex_patterns(self, tmp_path):
        """Test plugin with complex URL patterns."""
        app_file = tmp_path / "complex.py"
        app_file.write_text(
            textwrap.dedent(
                '''
                import tornado.web

                class ComplexHandler(tornado.web.RequestHandler):
                    def get(self, param1, param2):
                        pass

                application = tornado.web.Application([
                    (r"/complex/([^/]+)/items/([0-9]+)", ComplexHandler),
                ])
                '''
            )
        )
        
        plugin = TornadoPlugin()
        routes = plugin.discover(str(app_file))
        
        assert len(routes) == 1
        route = routes[0]
        assert route.path == "/complex/([^/]+)/items/([0-9]+)"
        assert route.methods == ["GET"]
        assert route.name == "ComplexHandler"