"""Load testing and performance benchmarks."""

import tempfile
import time
from pathlib import Path

import pytest
from locust import HttpUser, task
from locust.env import Environment
from locust.stats import StatsCSVFileWriter

from openapi_doc_generator.cli import main


@pytest.mark.performance
def test_route_discovery_performance():
    """Benchmark route discovery performance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create test apps of different sizes
        test_cases = [
            ("small", 10),
            ("medium", 50), 
            ("large", 200)
        ]
        
        results = {}
        
        for size_name, route_count in test_cases:
            app_file = tmpdir_path / f"{size_name}_app.py"
            
            # Generate routes
            routes = []
            for i in range(route_count):
                routes.append(f'''
@app.route("/api/endpoint{i}", methods=["GET", "POST"])
def endpoint{i}():
    """Endpoint {i} documentation."""
    return {{"id": {i}}}
''')
            
            app_content = f"""
from flask import Flask
app = Flask(__name__)

{chr(10).join(routes)}

if __name__ == "__main__":
    app.run()
"""
            app_file.write_text(app_content)
            
            # Measure performance
            start_time = time.perf_counter()
            
            result = main([
                "--app", str(app_file),
                "--format", "openapi",
                "--output", str(tmpdir_path / f"{size_name}_output.json")
            ])
            
            end_time = time.perf_counter()
            processing_time = end_time - start_time
            
            assert result == 0
            results[size_name] = {
                "routes": route_count,
                "time": processing_time,
                "routes_per_second": route_count / processing_time
            }
        
        # Performance assertions
        assert results["small"]["time"] < 1.0  # <1s for 10 routes
        assert results["medium"]["time"] < 5.0  # <5s for 50 routes
        assert results["large"]["time"] < 15.0  # <15s for 200 routes
        
        # Scaling should be roughly linear
        small_rps = results["small"]["routes_per_second"]
        medium_rps = results["medium"]["routes_per_second"]
        
        # Performance shouldn't degrade more than 50%
        assert medium_rps > small_rps * 0.5


@pytest.mark.performance
def test_memory_usage_benchmarks():
    """Test memory usage during documentation generation."""
    import psutil
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create a large application
        app_file = tmpdir_path / "memory_test_app.py"
        
        # Generate many routes with complex docstrings
        routes = []
        for i in range(100):
            routes.append(f'''
@app.route("/api/complex_endpoint{i}", methods=["GET", "POST", "PUT", "DELETE"])
def complex_endpoint{i}():
    """
    Complex endpoint {i} with extensive documentation.
    
    This endpoint handles multiple operations for resource {i}.
    It supports various HTTP methods and complex data structures.
    
    Returns:
        dict: A complex response with nested data structures
    
    Raises:
        ValueError: When input validation fails
        RuntimeError: When processing fails
    """
    return {{
        "id": {i},
        "data": {{"nested": {{"deep": {{"value": "test"}}}}}},
        "metadata": {{"created": "2023-01-01", "updated": "2023-01-02"}}
    }}
''')
        
        app_content = f"""
from flask import Flask
app = Flask(__name__)

{chr(10).join(routes)}

if __name__ == "__main__":
    app.run()
"""
        app_file.write_text(app_content)
        
        # Monitor memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        result = main([
            "--app", str(app_file),
            "--format", "openapi",
            "--output", str(tmpdir_path / "memory_test_output.json")
        ])
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        assert result == 0
        
        # Memory usage should be reasonable
        assert memory_increase < 500  # Less than 500MB increase
        
        print(f"Memory usage: {initial_memory:.1f}MB -> {peak_memory:.1f}MB "
              f"(+{memory_increase:.1f}MB)")


@pytest.mark.performance 
@pytest.mark.slow
def test_concurrent_processing_performance():
    """Test performance with concurrent processing."""
    import concurrent.futures
    import threading
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create multiple test apps
        apps = []
        for i in range(5):
            app_file = tmpdir_path / f"concurrent_app_{i}.py"
            app_content = f"""
from flask import Flask
app = Flask(__name__)

@app.route("/api/test{i}")
def test{i}():
    '''Test endpoint {i}.'''
    return {{"app": {i}}}

if __name__ == "__main__":
    app.run()
"""
            app_file.write_text(app_content)
            apps.append(str(app_file))
        
        # Test sequential processing
        start_time = time.perf_counter()
        
        for i, app_file in enumerate(apps):
            result = main([
                "--app", app_file,
                "--format", "openapi",
                "--output", str(tmpdir_path / f"sequential_{i}.json")
            ])
            assert result == 0
        
        sequential_time = time.perf_counter() - start_time
        
        # Test concurrent processing (simulate with threads)
        def process_app(app_info):
            i, app_file = app_info
            return main([
                "--app", app_file,
                "--format", "openapi", 
                "--output", str(tmpdir_path / f"concurrent_{i}.json")
            ])
        
        start_time = time.perf_counter()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(process_app, (i, app_file))
                for i, app_file in enumerate(apps)
            ]
            
            results = [future.result() for future in futures]
        
        concurrent_time = time.perf_counter() - start_time
        
        # All should succeed
        assert all(result == 0 for result in results)
        
        # Concurrent should be faster (allowing some overhead)
        speedup = sequential_time / concurrent_time
        print(f"Speedup: {speedup:.2f}x (sequential: {sequential_time:.2f}s, "
              f"concurrent: {concurrent_time:.2f}s)")
        
        # Should see some speedup with concurrent processing
        assert speedup > 1.2  # At least 20% improvement


class APILoadUser(HttpUser):
    """Locust user for load testing API documentation endpoints."""
    
    wait_time = lambda self: 1  # 1 second between requests
    
    @task(3)
    def get_openapi_spec(self):
        """Test OpenAPI spec endpoint."""
        self.client.get("/openapi.json")
    
    @task(2)
    def get_documentation(self):
        """Test documentation endpoint."""
        self.client.get("/docs")
    
    @task(1)
    def get_health(self):
        """Test health endpoint."""
        self.client.get("/health")


@pytest.mark.performance
@pytest.mark.requires_network
def test_api_load_testing():
    """Run load tests against API endpoints."""
    # This would typically run against a real server
    # For testing purposes, we'll simulate the load test setup
    
    env = Environment(user_classes=[APILoadUser])
    env.create_local_runner()
    
    # Configure stats collection
    stats_path = "/tmp/locust_stats"
    stats_writer = StatsCSVFileWriter(
        env, 
        stats_path,
        full_history=True
    )
    
    # Simulate a short load test
    env.runner.start(user_count=10, spawn_rate=2)
    
    # Run for 10 seconds
    time.sleep(10)
    
    env.runner.stop()
    
    # Verify test ran successfully
    assert env.runner.state == "stopped"
    assert len(env.stats.entries) > 0
    
    # Basic performance assertions
    total_stats = env.stats.total
    assert total_stats.num_failures == 0  # No failures
    assert total_stats.avg_response_time < 1000  # < 1s average response