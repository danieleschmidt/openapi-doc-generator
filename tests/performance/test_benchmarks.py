"""Performance benchmark tests."""

import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from openapi_doc_generator.discovery import RouteDiscoverer
from openapi_doc_generator.documentator import APIDocumentator


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmark tests for critical paths."""

    def test_route_discovery_performance(self, sample_flask_app, temp_dir, performance_timer):
        """Benchmark route discovery performance."""
        # Create test file
        app_file = temp_dir / "perf_test.py"
        app_file.write_text(sample_flask_app)
        
        discovery = RouteDiscovery()
        
        # Warm up
        discovery.discover_routes(str(app_file))
        
        # Benchmark
        performance_timer.start()
        for _ in range(10):  # Run multiple times for better measurement
            routes = discovery.discover_routes(str(app_file))
        performance_timer.stop()
        
        # Performance assertions
        avg_time = performance_timer.elapsed / 10
        assert avg_time < 0.1  # Should average less than 100ms per discovery
        assert routes is not None

    def test_large_file_performance(self, temp_dir, performance_timer):
        """Test performance with large application files."""
        # Generate a large Flask application
        large_app_content = '''
from flask import Flask, jsonify

app = Flask(__name__)

'''
        
        # Add many routes to simulate large application
        for i in range(100):
            large_app_content += f'''
@app.route("/api/endpoint_{i}", methods=["GET", "POST"])
def endpoint_{i}():
    """Endpoint {i} documentation."""
    return jsonify({{"endpoint": {i}}})

'''
        
        large_app_file = temp_dir / "large_app.py"
        large_app_file.write_text(large_app_content)
        
        # Benchmark
        performance_timer.start()
        
        documentator = APIDocumentator()
        result = documentator.analyze_app(str(large_app_file))
        spec = result.generate_openapi_spec()
        
        performance_timer.stop()
        
        # Should handle large files reasonably
        assert performance_timer.elapsed < 10.0  # Less than 10 seconds
        assert len(spec.get("paths", {})) >= 90  # Should find most routes

    def test_ast_parsing_cache_performance(self, sample_flask_app, temp_dir, performance_timer):
        """Test AST parsing cache effectiveness."""
        app_file = temp_dir / "cache_test.py"
        app_file.write_text(sample_flask_app)
        
        documentator = APIDocumentator()
        
        # First run - cache miss
        performance_timer.start()
        result1 = documentator.analyze_app(str(app_file))
        first_time = time.perf_counter()
        
        # Second run - should use cache
        result2 = documentator.analyze_app(str(app_file))
        performance_timer.stop()
        
        second_time = time.perf_counter()
        
        # Cache should make second run faster
        cache_time = second_time - first_time
        total_time = performance_timer.elapsed
        first_run_time = first_time - (performance_timer.end_time - performance_timer.elapsed)
        
        # Second run should be significantly faster
        assert cache_time < first_run_time * 0.8  # At least 20% faster

    def test_memory_usage_stability(self, temp_dir):
        """Test memory usage doesn't grow excessively."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process multiple files
        for i in range(20):
            app_file = temp_dir / f"test_app_{i}.py"
            app_file.write_text(f'''
from flask import Flask
app = Flask(__name__)

@app.route("/api/test_{i}")
def test_{i}():
    return "test {i}"
''')
            
            documentator = APIDocumentator()
            result = documentator.analyze_app(str(app_file))
            spec = result.generate_openapi_spec()
        
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 50MB)
        assert memory_growth < 50 * 1024 * 1024  # 50MB

    def test_concurrent_processing_performance(self, temp_dir, performance_timer):
        """Test performance with concurrent processing."""
        import concurrent.futures
        
        # Create multiple test files
        test_files = []
        for i in range(5):
            app_file = temp_dir / f"concurrent_test_{i}.py"
            app_file.write_text(f'''
from flask import Flask
app = Flask(__name__)

@app.route("/api/endpoint_{i}")
def endpoint_{i}():
    return "endpoint {i}"
''')
            test_files.append(str(app_file))
        
        def process_file(file_path):
            documentator = APIDocumentator()
            result = documentator.analyze_app(file_path)
            return result.generate_openapi_spec()
        
        # Sequential processing
        performance_timer.start()
        sequential_results = []
        for file_path in test_files:
            result = process_file(file_path)
            sequential_results.append(result)
        sequential_time = time.perf_counter()
        
        # Concurrent processing
        concurrent_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_file = {executor.submit(process_file, file_path): file_path 
                             for file_path in test_files}
            for future in concurrent.futures.as_completed(future_to_file):
                result = future.result()
                concurrent_results.append(result)
        
        performance_timer.stop()
        concurrent_time = performance_timer.elapsed - (sequential_time - performance_timer.start_time)
        
        # Verify results are equivalent
        assert len(sequential_results) == len(concurrent_results)
        
        # Concurrent should be faster (or at least not significantly slower)
        # Note: This might not always be true due to Python GIL, but worth measuring
        assert concurrent_time <= sequential_time * 1.5  # Allow some overhead

    @pytest.mark.slow
    def test_stress_testing(self, temp_dir, performance_timer):
        """Stress test with many operations."""
        documentator = APIDocumentator()
        
        # Create a moderately complex app
        app_content = '''
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/api/users", methods=["GET", "POST", "PUT", "DELETE"])
def users():
    """User management with multiple methods."""
    return jsonify({"users": []})

@app.route("/api/users/<int:user_id>", methods=["GET", "PUT", "DELETE"])
def user_detail(user_id: int):
    """Individual user operations."""
    return jsonify({"user_id": user_id})

@app.route("/api/posts/<int:post_id>/comments", methods=["GET", "POST"])
def post_comments(post_id: int):
    """Post comments management."""
    return jsonify({"comments": []})
'''
        
        app_file = temp_dir / "stress_test.py"
        app_file.write_text(app_content)
        
        # Run many iterations
        performance_timer.start()
        
        for i in range(50):  # 50 iterations
            result = documentator.analyze_app(str(app_file))
            spec = result.generate_openapi_spec()
            markdown = result.generate_markdown()
            
            # Verify output quality doesn't degrade
            assert "openapi" in spec
            assert len(spec.get("paths", {})) >= 3
        
        performance_timer.stop()
        
        # Should complete stress test in reasonable time
        assert performance_timer.elapsed < 30.0  # Less than 30 seconds
        
        # Average time per iteration should be reasonable
        avg_time = performance_timer.elapsed / 50
        assert avg_time < 0.6  # Less than 600ms per iteration