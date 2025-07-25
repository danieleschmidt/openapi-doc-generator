"""Performance benchmarking tests for the OpenAPI Doc Generator."""

import time
import textwrap
from pathlib import Path
from typing import Dict
import statistics

import pytest

from openapi_doc_generator.discovery import RouteDiscoverer
from openapi_doc_generator.documentator import APIDocumentator


class PerformanceBenchmarks:
    """Collection of performance benchmarks."""

    @staticmethod
    def benchmark_route_discovery_simple(tmp_path) -> Dict[str, float]:
        """Benchmark route discovery for a simple Flask app."""
        app_file = tmp_path / "simple_app.py"
        app_file.write_text(
            textwrap.dedent(
                '''
                from flask import Flask
                app = Flask(__name__)

                @app.route("/")
                def index():
                    return "Hello"

                @app.route("/users")
                def users():
                    return "Users"
                '''
            )
        )
        
        iterations = 10
        times = []
        
        for _ in range(iterations):
            start = time.perf_counter()
            discoverer = RouteDiscoverer(str(app_file))
            routes = discoverer.discover()
            end = time.perf_counter()
            times.append(end - start)
        
        return {
            "mean_time": statistics.mean(times),
            "median_time": statistics.median(times),
            "min_time": min(times),
            "max_time": max(times),
            "route_count": len(routes),
            "iterations": iterations
        }

    @staticmethod
    def benchmark_route_discovery_complex(tmp_path) -> Dict[str, float]:
        """Benchmark route discovery for a complex Flask app with many routes."""
        app_file = tmp_path / "complex_app.py"
        
        # Generate a large Flask app with many routes
        routes_code = []
        for i in range(50):
            routes_code.append(f'''
@app.route("/api/v1/users/{i}")
def user_{i}():
    """Get user {i}."""
    return {{"user_id": {i}}}

@app.route("/api/v1/users/{i}", methods=["POST"])
def create_user_{i}():
    """Create user {i}."""
    return {{"created": {i}}}

@app.route("/api/v1/users/{i}", methods=["PUT"])
def update_user_{i}():
    """Update user {i}."""
    return {{"updated": {i}}}
''')
        
        app_content = f'''
from flask import Flask
app = Flask(__name__)
{''.join(routes_code)}
'''
        
        app_file.write_text(app_content)
        
        iterations = 5  # Fewer iterations for complex test
        times = []
        
        for _ in range(iterations):
            start = time.perf_counter()
            discoverer = RouteDiscoverer(str(app_file))
            routes = discoverer.discover()
            end = time.perf_counter()
            times.append(end - start)
        
        return {
            "mean_time": statistics.mean(times),
            "median_time": statistics.median(times),
            "min_time": min(times),
            "max_time": max(times),
            "route_count": len(routes),
            "iterations": iterations
        }

    @staticmethod
    def benchmark_full_documentation_generation(tmp_path) -> Dict[str, float]:
        """Benchmark full documentation generation pipeline."""
        app_file = tmp_path / "doc_app.py"
        app_file.write_text(
            textwrap.dedent(
                '''
                from flask import Flask
                app = Flask(__name__)

                @app.route("/api/users", methods=["GET", "POST"])
                def users():
                    """User management endpoint."""
                    return {"users": []}

                @app.route("/api/users/<int:user_id>", methods=["GET", "PUT", "DELETE"])
                def user_detail(user_id):
                    """Individual user operations."""
                    return {"user": user_id}

                @app.route("/health")
                def health():
                    """Health check endpoint."""
                    return {"status": "ok"}
                '''
            )
        )
        
        iterations = 5
        times = []
        
        for _ in range(iterations):
            start = time.perf_counter()
            documentator = APIDocumentator()
            result = documentator.analyze_app(str(app_file))
            markdown = result.generate_markdown()
            openapi_spec = result.generate_openapi_spec()
            end = time.perf_counter()
            times.append(end - start)
        
        return {
            "mean_time": statistics.mean(times),
            "median_time": statistics.median(times),
            "min_time": min(times),
            "max_time": max(times),
            "markdown_length": len(markdown),
            "openapi_paths": len(openapi_spec.get("paths", {})),
            "iterations": iterations
        }

    @staticmethod
    def benchmark_tornado_plugin(tmp_path) -> Dict[str, float]:
        """Benchmark Tornado plugin performance."""
        app_file = tmp_path / "tornado_app.py"
        app_file.write_text(
            textwrap.dedent(
                '''
                import tornado.web

                class MainHandler(tornado.web.RequestHandler):
                    def get(self):
                        pass
                    def post(self):
                        pass

                class UserHandler(tornado.web.RequestHandler):
                    def get(self, user_id):
                        pass
                    def put(self, user_id):
                        pass
                    def delete(self, user_id):
                        pass

                class AdminHandler(tornado.web.RequestHandler):
                    def get(self):
                        pass

                application = tornado.web.Application([
                    (r"/", MainHandler),
                    (r"/user/([^/]+)", UserHandler),
                    (r"/admin", AdminHandler),
                ])
                '''
            )
        )
        
        iterations = 10
        times = []
        
        for _ in range(iterations):
            start = time.perf_counter()
            discoverer = RouteDiscoverer(str(app_file))
            routes = discoverer.discover()
            end = time.perf_counter()
            times.append(end - start)
        
        return {
            "mean_time": statistics.mean(times),
            "median_time": statistics.median(times),
            "min_time": min(times),
            "max_time": max(times),
            "route_count": len(routes),
            "iterations": iterations
        }


class TestPerformanceBenchmarks:
    """Test class for performance benchmarks."""

    def test_benchmark_route_discovery_simple_performance(self, tmp_path):
        """Test that simple route discovery meets performance targets."""
        results = PerformanceBenchmarks.benchmark_route_discovery_simple(tmp_path)
        
        # Performance targets
        assert results["mean_time"] < 0.1, f"Mean time {results['mean_time']:.3f}s exceeds 0.1s target"
        assert results["max_time"] < 0.2, f"Max time {results['max_time']:.3f}s exceeds 0.2s target"
        assert results["route_count"] == 2, f"Expected 2 routes, got {results['route_count']}"
        
        print(f"Simple discovery: {results['mean_time']:.3f}s avg, {results['route_count']} routes")

    def test_benchmark_route_discovery_complex_performance(self, tmp_path):
        """Test that complex route discovery meets performance targets."""
        results = PerformanceBenchmarks.benchmark_route_discovery_complex(tmp_path)
        
        # Performance targets for complex app
        assert results["mean_time"] < 1.0, f"Mean time {results['mean_time']:.3f}s exceeds 1.0s target"
        assert results["max_time"] < 2.0, f"Max time {results['max_time']:.3f}s exceeds 2.0s target"
        assert results["route_count"] == 150, f"Expected 150 routes, got {results['route_count']}"
        
        print(f"Complex discovery: {results['mean_time']:.3f}s avg, {results['route_count']} routes")

    def test_benchmark_full_documentation_generation_performance(self, tmp_path):
        """Test that full documentation generation meets performance targets."""
        results = PerformanceBenchmarks.benchmark_full_documentation_generation(tmp_path)
        
        # Performance targets for full pipeline
        assert results["mean_time"] < 0.5, f"Mean time {results['mean_time']:.3f}s exceeds 0.5s target"
        assert results["max_time"] < 1.0, f"Max time {results['max_time']:.3f}s exceeds 1.0s target"
        assert results["openapi_paths"] > 0, "Should generate OpenAPI paths"
        assert results["markdown_length"] > 100, "Should generate substantial markdown content"
        
        print(f"Full pipeline: {results['mean_time']:.3f}s avg, {results['markdown_length']} chars")

    def test_benchmark_tornado_plugin_performance(self, tmp_path):
        """Test that Tornado plugin meets performance targets."""
        results = PerformanceBenchmarks.benchmark_tornado_plugin(tmp_path)
        
        # Performance targets for Tornado plugin
        assert results["mean_time"] < 0.1, f"Mean time {results['mean_time']:.3f}s exceeds 0.1s target"
        assert results["max_time"] < 0.2, f"Max time {results['max_time']:.3f}s exceeds 0.2s target"
        assert results["route_count"] == 3, f"Expected 3 routes, got {results['route_count']}"
        
        print(f"Tornado plugin: {results['mean_time']:.3f}s avg, {results['route_count']} routes")

    @pytest.mark.slow
    def test_performance_regression_detection(self, tmp_path):
        """Test for performance regression detection across multiple runs."""
        # Run benchmarks multiple times to detect variance
        simple_results = []
        complex_results = []
        
        for run in range(3):
            simple_results.append(
                PerformanceBenchmarks.benchmark_route_discovery_simple(tmp_path)["mean_time"]
            )
            complex_results.append(
                PerformanceBenchmarks.benchmark_route_discovery_complex(tmp_path)["mean_time"]
            )
        
        # Check for reasonable variance (coefficient of variation < 50%)
        simple_cv = statistics.stdev(simple_results) / statistics.mean(simple_results)
        complex_cv = statistics.stdev(complex_results) / statistics.mean(complex_results)
        
        assert simple_cv < 0.5, f"Simple benchmark variance too high: {simple_cv:.2f}"
        assert complex_cv < 0.5, f"Complex benchmark variance too high: {complex_cv:.2f}"
        
        print(f"Variance check - Simple CV: {simple_cv:.3f}, Complex CV: {complex_cv:.3f}")


@pytest.mark.benchmark
class TestPerformanceProfiler:
    """Profiling tests to identify performance bottlenecks."""

    def test_profile_ast_parsing_overhead(self, tmp_path):
        """Profile AST parsing overhead in route discovery."""
        app_file = tmp_path / "profile_app.py"
        app_file.write_text(
            textwrap.dedent(
                '''
                from flask import Flask
                app = Flask(__name__)

                @app.route("/test")
                def test():
                    return "test"
                '''
            )
        )
        
        # Profile with and without AST caching
        import openapi_doc_generator.utils as utils
        
        # Clear any existing cache
        if hasattr(utils, '_ast_cache'):
            utils._ast_cache.clear()
        
        # Time first run (cache miss)
        start = time.perf_counter()
        discoverer1 = RouteDiscoverer(str(app_file))
        routes1 = discoverer1.discover()
        time_first = time.perf_counter() - start
        
        # Time second run (cache hit)
        start = time.perf_counter()
        discoverer2 = RouteDiscoverer(str(app_file))
        routes2 = discoverer2.discover()
        time_second = time.perf_counter() - start
        
        # Cache should provide some improvement
        improvement = (time_first - time_second) / time_first
        print(f"AST caching improvement: {improvement:.1%} ({time_first:.4f}s -> {time_second:.4f}s)")
        
        # Basic sanity checks
        assert len(routes1) == len(routes2), "Cache should not affect results"
        assert time_second <= time_first * 1.1, "Second run should not be significantly slower"


def run_performance_suite(output_file: str = None):
    """Run the complete performance benchmark suite and optionally save results."""
    import json
    from datetime import datetime
    
    # Create a temporary directory for testing
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        benchmarks = PerformanceBenchmarks()
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "benchmarks": {
                "simple_discovery": benchmarks.benchmark_route_discovery_simple(tmp_path),
                "complex_discovery": benchmarks.benchmark_route_discovery_complex(tmp_path),
                "full_pipeline": benchmarks.benchmark_full_documentation_generation(tmp_path),
                "tornado_plugin": benchmarks.benchmark_tornado_plugin(tmp_path),
            }
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Performance results saved to {output_file}")
        
        return results


if __name__ == "__main__":
    # Run benchmarks standalone
    results = run_performance_suite("performance_results.json")
    print("Performance Benchmark Results:")
    for name, result in results["benchmarks"].items():
        print(f"  {name}: {result['mean_time']:.3f}s avg ({result.get('route_count', 'N/A')} routes)")