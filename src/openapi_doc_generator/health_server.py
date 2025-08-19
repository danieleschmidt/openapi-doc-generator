"""Simple HTTP server for health checks and metrics endpoints."""

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional
from urllib.parse import urlparse

from .monitoring import get_health_status, get_metrics, get_readiness_status


class HealthHandler(BaseHTTPRequestHandler):
    """HTTP handler for health and metrics endpoints."""

    def log_message(self, format, *args):
        """Override to use proper logging."""
        logging.getLogger("health_server").info(format % args)

    def do_GET(self):
        """Handle GET requests."""
        path = urlparse(self.path).path

        try:
            if path == "/health":
                self._handle_health()
            elif path == "/ready":
                self._handle_readiness()
            elif path == "/metrics":
                self._handle_metrics()
            elif path == "/version":
                self._handle_version()
            else:
                self._handle_not_found()
        except Exception as e:
            self._handle_error(str(e))

    def _handle_health(self):
        """Handle health check endpoint."""
        health_status = get_health_status()

        # Set status code based on health
        if health_status["status"] == "healthy":
            status_code = 200
        elif health_status["status"] == "degraded":
            status_code = 200  # Still responding
        else:
            status_code = 503  # Service unavailable

        self._send_json_response(health_status, status_code)

    def _handle_readiness(self):
        """Handle readiness check endpoint."""
        readiness_status = get_readiness_status()

        status_code = 200 if readiness_status["ready"] else 503

        self._send_json_response(readiness_status, status_code)

    def _handle_metrics(self):
        """Handle metrics endpoint (Prometheus format)."""
        metrics = get_metrics()

        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.send_header('Content-Length', str(len(metrics.encode())))
        self.end_headers()
        self.wfile.write(metrics.encode())

    def _handle_version(self):
        """Handle version endpoint."""
        version_info = {
            "version": "1.2.0",
            "build_date": "2025-01-15",
            "git_commit": "unknown",
            "python_version": "3.12"
        }

        self._send_json_response(version_info, 200)

    def _handle_not_found(self):
        """Handle 404 errors."""
        error_response = {
            "error": "Not Found",
            "message": "Available endpoints: /health, /ready, /metrics, /version"
        }

        self._send_json_response(error_response, 404)

    def _handle_error(self, error_message: str):
        """Handle internal errors."""
        error_response = {
            "error": "Internal Server Error",
            "message": error_message
        }

        self._send_json_response(error_response, 500)

    def _send_json_response(self, data: dict, status_code: int):
        """Send JSON response."""
        response_body = json.dumps(data, indent=2)

        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response_body.encode())))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(response_body.encode())


class HealthServer:
    """Simple HTTP server for health checks and metrics."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        self.host = host
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(__name__)

    def start(self) -> None:
        """Start the health server."""
        try:
            self.server = HTTPServer((self.host, self.port), HealthHandler)
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()

            self.logger.info(f"Health server started on {self.host}:{self.port}")
            self.logger.info("Available endpoints:")
            self.logger.info(f"  - http://{self.host}:{self.port}/health")
            self.logger.info(f"  - http://{self.host}:{self.port}/ready")
            self.logger.info(f"  - http://{self.host}:{self.port}/metrics")
            self.logger.info(f"  - http://{self.host}:{self.port}/version")

        except OSError as e:
            self.logger.error(f"Failed to start health server: {e}")
            raise

    def stop(self) -> None:
        """Stop the health server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()

        if self.thread:
            self.thread.join(timeout=5.0)

        self.logger.info("Health server stopped")

    def is_running(self) -> bool:
        """Check if server is running."""
        return self.thread is not None and self.thread.is_alive()


# Global health server instance
health_server: Optional[HealthServer] = None


def start_health_server(host: str = "127.0.0.1", port: int = 8080) -> HealthServer:
    """Start the global health server."""
    global health_server

    if health_server and health_server.is_running():
        return health_server

    health_server = HealthServer(host, port)
    health_server.start()
    return health_server


def stop_health_server() -> None:
    """Stop the global health server."""
    global health_server

    if health_server:
        health_server.stop()
        health_server = None


def get_health_server() -> Optional[HealthServer]:
    """Get the global health server instance."""
    return health_server
