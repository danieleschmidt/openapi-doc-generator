"""
Comprehensive tests for Generation 2 & 3 enhanced modules.
"""

import asyncio
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.openapi_doc_generator.enhanced_error_handling import (
    EnhancedErrorHandler, ErrorCategory, ErrorSeverity, ErrorContext,
    get_error_handler, with_error_handling
)
from src.openapi_doc_generator.enhanced_validation import (
    InputValidator, ValidationError, get_validator
)
from src.openapi_doc_generator.enhanced_monitoring import (
    EnhancedMonitor, HealthStatus, MetricType, get_monitor, monitor_operation
)
from src.openapi_doc_generator.performance_optimizer import (
    PerformanceOptimizer, OptimizationConfig, AdvancedCache, CacheStrategy,
    get_optimizer, optimized
)
from src.openapi_doc_generator.auto_scaler import (
    IntelligentAutoScaler, ResourceLimits, ScalingDirection, ScalingTrigger,
    get_auto_scaler
)


class TestEnhancedErrorHandling:
    """Test enhanced error handling system."""
    
    def test_error_handler_initialization(self):
        """Test error handler initialization."""
        handler = EnhancedErrorHandler()
        assert handler.error_count == 0
        assert handler.recovery_attempts == 0
        assert len(handler.error_history) == 0
    
    def test_error_categorization(self):
        """Test error categorization."""
        handler = EnhancedErrorHandler()
        context = ErrorContext(operation="test")
        
        # Test different error types
        file_error = FileNotFoundError("test file not found")
        category = handler._categorize_error(file_error, context)
        assert category == ErrorCategory.FILESYSTEM
        
        import_error = ImportError("test module not found")
        context.component = "plugin"
        category = handler._categorize_error(import_error, context)
        assert category == ErrorCategory.PLUGIN_LOADING
    
    def test_severity_assessment(self):
        """Test error severity assessment."""
        handler = EnhancedErrorHandler()
        context = ErrorContext(operation="test")
        
        # Test critical error
        memory_error = MemoryError("out of memory")
        severity = handler._assess_severity(memory_error, ErrorCategory.FILESYSTEM, context)
        assert severity == ErrorSeverity.CRITICAL
        
        # Test low severity plugin error
        import_error = ImportError("plugin not found")
        severity = handler._assess_severity(import_error, ErrorCategory.PLUGIN_LOADING, context)
        assert severity == ErrorSeverity.LOW
    
    def test_error_context_manager(self):
        """Test error context manager."""
        handler = EnhancedErrorHandler()
        context = ErrorContext(operation="test_operation")
        
        with pytest.raises(ValueError):
            with handler.error_context(context):
                raise ValueError("test error")
        
        assert handler.error_count == 1
        assert len(handler.error_history) == 1
    
    def test_global_error_handler(self):
        """Test global error handler access."""
        handler1 = get_error_handler()
        handler2 = get_error_handler()
        assert handler1 is handler2  # Should be same instance
    
    def test_error_handling_decorator(self):
        """Test error handling decorator."""
        @with_error_handling("test_operation")
        def test_function():
            raise ValueError("test error")
        
        with pytest.raises(ValueError):
            test_function()
        
        # Check that error was recorded
        handler = get_error_handler()
        assert handler.error_count > 0


class TestEnhancedValidation:
    """Test enhanced validation system."""
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = InputValidator()
        assert len(validator.allowed_extensions) > 0
        assert validator.max_file_size > 0
    
    def test_file_path_validation_success(self):
        """Test successful file path validation."""
        validator = InputValidator()
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as f:
            f.write(b"test content")
            temp_path = f.name
        
        try:
            validated_path = validator.validate_file_path(temp_path)
            assert validated_path.exists()
            assert validated_path.is_file()
        finally:
            os.unlink(temp_path)
    
    def test_file_path_validation_failure(self):
        """Test file path validation failure cases."""
        validator = InputValidator()
        
        # Test non-existent file
        with pytest.raises(ValidationError):
            validator.validate_file_path("/nonexistent/file.py")
        
        # Test empty path
        with pytest.raises(ValidationError):
            validator.validate_file_path("")
        
        # Test path traversal
        with pytest.raises(ValidationError):
            validator.validate_file_path("../../../etc/passwd")
    
    def test_format_validation(self):
        """Test format validation."""
        validator = InputValidator()
        
        # Test valid format
        result = validator.validate_format("openapi", ["openapi", "markdown"])
        assert result == "openapi"
        
        # Test invalid format
        with pytest.raises(ValidationError):
            validator.validate_format("invalid", ["openapi", "markdown"])
    
    def test_api_title_validation(self):
        """Test API title validation."""
        validator = InputValidator()
        
        # Test valid title
        result = validator.validate_api_title("My API")
        assert result == "My API"
        
        # Test empty title
        with pytest.raises(ValidationError):
            validator.validate_api_title("")
        
        # Test dangerous characters
        with pytest.raises(ValidationError):
            validator.validate_api_title("API <script>alert('xss')</script>")
    
    def test_global_validator(self):
        """Test global validator access."""
        validator1 = get_validator()
        validator2 = get_validator()
        assert validator1 is validator2  # Should be same instance


class TestEnhancedMonitoring:
    """Test enhanced monitoring system."""
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = EnhancedMonitor()
        assert len(monitor.health_checks) > 0
        assert len(monitor.alerts) > 0
        assert not monitor.monitoring_active
    
    @pytest.mark.asyncio
    async def test_health_checks(self):
        """Test health check execution."""
        monitor = EnhancedMonitor()
        
        # Run health checks
        results = await monitor.run_health_checks()
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Check that results are boolean
        for name, result in results.items():
            assert isinstance(result, bool)
    
    def test_health_status(self):
        """Test health status calculation."""
        monitor = EnhancedMonitor()
        
        # Initially should be unknown
        status = monitor.get_health_status()
        assert status == HealthStatus.UNKNOWN
        
        # Set some health results
        monitor.health_results = {
            "test_check": True,
            "another_check": True
        }
        status = monitor.get_health_status()
        assert status == HealthStatus.HEALTHY
    
    def test_metric_recording(self):
        """Test metric recording."""
        monitor = EnhancedMonitor()
        
        monitor.record_metric("test_metric", 42.0, MetricType.GAUGE)
        assert "test_metric" in monitor.metrics
        assert len(monitor.metrics["test_metric"]) == 1
    
    def test_metrics_summary(self):
        """Test metrics summary generation."""
        monitor = EnhancedMonitor()
        
        summary = monitor.get_metrics_summary()
        assert "timestamp" in summary
        assert "uptime_seconds" in summary
        assert "health_status" in summary
    
    def test_global_monitor(self):
        """Test global monitor access."""
        monitor1 = get_monitor()
        monitor2 = get_monitor()
        assert monitor1 is monitor2  # Should be same instance
    
    def test_monitor_operation_decorator(self):
        """Test monitor operation decorator."""
        @monitor_operation("test_operation")
        def test_function():
            return "success"
        
        result = test_function()
        assert result == "success"
        
        # Check that operation was recorded
        monitor = get_monitor()
        assert "test_operation" in monitor.operation_times


class TestPerformanceOptimizer:
    """Test performance optimization system."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        config = OptimizationConfig()
        optimizer = PerformanceOptimizer(config)
        
        assert optimizer.config == config
        assert optimizer.cache is not None
        assert optimizer.parallel_processor is not None
        assert optimizer.memory_optimizer is not None
    
    def test_advanced_cache(self):
        """Test advanced cache functionality."""
        cache = AdvancedCache(CacheStrategy.LRU, max_size=3)
        
        # Test basic operations
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.get("nonexistent") is None
        
        # Test LRU eviction
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        cache.put("key4", "value4")  # Should evict key1
        
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
    
    def test_cache_strategies(self):
        """Test different cache strategies."""
        # Test TTL strategy
        ttl_cache = AdvancedCache(CacheStrategy.TTL, ttl=0.1)
        ttl_cache.put("key", "value")
        assert ttl_cache.get("key") == "value"
        
        # Wait for TTL expiration
        import time
        time.sleep(0.2)
        assert ttl_cache.get("key") is None
    
    def test_cache_statistics(self):
        """Test cache statistics."""
        cache = AdvancedCache(CacheStrategy.LRU)
        
        # Generate some cache activity
        cache.put("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
    
    @pytest.mark.asyncio
    async def test_batch_optimization(self):
        """Test batch operation optimization."""
        optimizer = PerformanceOptimizer()
        
        # Create simple test operations
        operations = [
            lambda: "result1",
            lambda: "result2",
            lambda: "result3"
        ]
        
        results = await optimizer.optimize_batch_operation(operations, "test_batch")
        assert len(results) == 3
        assert "result1" in results
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        optimizer = PerformanceOptimizer()
        
        summary = optimizer.get_performance_summary()
        assert "config" in summary
        assert "cache_stats" in summary
        assert "operation_profiles" in summary
        assert "recent_metrics" in summary
    
    def test_global_optimizer(self):
        """Test global optimizer access."""
        optimizer1 = get_optimizer()
        optimizer2 = get_optimizer()
        assert optimizer1 is optimizer2  # Should be same instance
    
    def test_optimized_decorator(self):
        """Test optimized decorator."""
        @optimized("test_operation")
        def test_function(x):
            return x * 2
        
        result = test_function(5)
        assert result == 10
        
        # Check that operation was profiled
        optimizer = get_optimizer()
        assert "test_operation" in optimizer.operation_profiles


class TestAutoScaler:
    """Test intelligent auto-scaling system."""
    
    def test_auto_scaler_initialization(self):
        """Test auto-scaler initialization."""
        limits = ResourceLimits(min_workers=1, max_workers=8)
        scaler = IntelligentAutoScaler(limits)
        
        assert scaler.limits == limits
        assert len(scaler.scaling_rules) > 0
        assert scaler.scaling_enabled
    
    def test_resource_limits(self):
        """Test resource limits configuration."""
        limits = ResourceLimits(min_workers=2, max_workers=16)
        assert limits.min_workers == 2
        assert limits.max_workers == 16
    
    def test_scaling_rule_creation(self):
        """Test scaling rule creation."""
        scaler = IntelligentAutoScaler()
        rules = scaler._create_default_rules()
        
        assert len(rules) > 0
        for rule in rules:
            assert rule.trigger in ScalingTrigger
            assert rule.threshold_up > 0
            assert rule.threshold_down >= 0
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self):
        """Test metrics collection."""
        scaler = IntelligentAutoScaler()
        
        metrics = await scaler._collect_current_metrics()
        assert "cpu_utilization" in metrics
        assert "memory_utilization" in metrics
        assert "response_time" in metrics
    
    def test_scaling_summary(self):
        """Test scaling summary generation."""
        scaler = IntelligentAutoScaler()
        
        summary = scaler.get_scaling_summary()
        assert "enabled" in summary
        assert "resource_limits" in summary
        assert "current_config" in summary
        assert "recent_actions" in summary
    
    def test_global_auto_scaler(self):
        """Test global auto-scaler access."""
        scaler1 = get_auto_scaler()
        scaler2 = get_auto_scaler()
        assert scaler1 is scaler2  # Should be same instance


class TestIntegration:
    """Test integration between enhanced modules."""
    
    def test_error_handler_with_monitor(self):
        """Test error handler integration with monitor."""
        error_handler = get_error_handler()
        monitor = get_monitor()
        
        # Generate an error
        context = ErrorContext(operation="integration_test")
        
        try:
            with error_handler.error_context(context):
                raise ValueError("integration test error")
        except ValueError:
            pass
        
        # Check that error was recorded
        assert error_handler.error_count > 0
        
        # Check monitor error summary
        error_summary = error_handler.get_error_summary()
        assert error_summary["total_errors"] > 0
    
    def test_optimizer_with_monitor(self):
        """Test optimizer integration with monitor."""
        optimizer = get_optimizer()
        monitor = get_monitor()
        
        # Record some performance metrics
        optimizer._record_operation_performance("integration_test", 0.1, 1)
        
        # Check that metrics were recorded
        assert "integration_test" in optimizer.operation_profiles
        
        # Check monitor has metrics
        summary = monitor.get_metrics_summary()
        assert "operations" in summary
    
    @pytest.mark.asyncio
    async def test_auto_scaler_with_optimizer(self):
        """Test auto-scaler integration with optimizer."""
        scaler = get_auto_scaler()
        optimizer = get_optimizer()
        
        # Simulate high CPU usage scenario
        with patch('psutil.cpu_percent', return_value=80.0):
            metrics = await scaler._collect_current_metrics()
            assert metrics["cpu_utilization"] == 80.0
        
        # Check that auto-scaler can access optimizer config
        summary = scaler.get_scaling_summary()
        assert "current_config" in summary
        assert "workers" in summary["current_config"]


@pytest.fixture
def cleanup_globals():
    """Fixture to cleanup global instances between tests."""
    yield
    # Reset global instances if needed
    # This ensures tests don't interfere with each other