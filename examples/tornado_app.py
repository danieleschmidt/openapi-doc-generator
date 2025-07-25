"""Example Tornado application for testing route discovery."""

import tornado.web
import tornado.ioloop


class MainHandler(tornado.web.RequestHandler):
    """Main page handler."""
    
    def get(self):
        """Handle GET requests to the main page."""
        self.write("Hello, Tornado!")

    def post(self):
        """Handle POST requests to the main page."""
        self.write("Posted to main page!")


class UserHandler(tornado.web.RequestHandler):
    """User management handler."""
    
    def get(self, user_id):
        """Get user information."""
        self.write(f"User {user_id}")
    
    def put(self, user_id):
        """Update user information."""
        self.write(f"Updated user {user_id}")
    
    def delete(self, user_id):
        """Delete user."""
        self.write(f"Deleted user {user_id}")


class StatusHandler(tornado.web.RequestHandler):
    """API status handler."""
    
    def get(self):
        """Get API status."""
        self.write({"status": "ok", "version": "1.0"})


def make_app():
    """Create and return the Tornado application."""
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/user/([^/]+)", UserHandler),
        (r"/status", StatusHandler),
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()