# Telemetry and Observability Implementation

Advanced monitoring and observability setup for OpenAPI-Doc-Generator production deployments.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │────│   OpenTelemetry │────│   Collectors    │
│                 │    │      SDK        │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                              ┌────────────────────────┼────────────────────────┐
                              │                        │                        │
                    ┌─────────▼─────────┐    ┌─────────▼─────────┐    ┌─────────▼─────────┐
                    │     Prometheus    │    │     Jaeger        │    │   Elasticsearch   │
                    │    (Metrics)      │    │    (Traces)       │    │     (Logs)        │
                    └───────────────────┘    └───────────────────┘    └───────────────────┘
```

## Implementation Components

### 1. OpenTelemetry Integration

**File**: `src/openapi_doc_generator/telemetry.py`

```python
"""Advanced telemetry and observability implementation."""

import logging
import time
from typing import Optional, Dict, Any
from opentelemetry import trace, metrics
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.instrumentation.logging import LoggingInstrumentor

class TelemetryManager:
    """Centralized telemetry management."""
    
    def __init__(self, service_name: str = "openapi-doc-generator"):
        self.service_name = service_name
        self.tracer = None
        self.meter = None
        self.logger = logging.getLogger(__name__)
        
    def initialize(self, config: Dict[str, Any]):
        """Initialize telemetry with configuration."""
        # Setup tracing
        if config.get("tracing", {}).get("enabled", False):
            self._setup_tracing(config["tracing"])
        
        # Setup metrics
        if config.get("metrics", {}).get("enabled", False):
            self._setup_metrics(config["metrics"])
        
        # Setup logging
        if config.get("logging", {}).get("structured", False):
            self._setup_structured_logging(config["logging"])
    
    def _setup_tracing(self, config: Dict[str, Any]):
        """Configure distributed tracing."""
        provider = TracerProvider()
        
        # Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name=config.get("jaeger_host", "localhost"),
            agent_port=config.get("jaeger_port", 6831),
        )
        
        provider.add_span_processor(
            BatchSpanProcessor(jaeger_exporter)
        )
        
        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer(self.service_name)
    
    def _setup_metrics(self, config: Dict[str, Any]):
        """Configure metrics collection."""
        reader = PrometheusMetricReader()
        provider = MeterProvider(metric_readers=[reader])
        
        metrics.set_meter_provider(provider)
        self.meter = metrics.get_meter(self.service_name)
    
    def _setup_structured_logging(self, config: Dict[str, Any]):
        """Configure structured logging."""
        LoggingInstrumentor().instrument(set_logging_format=True)
        
        formatter = logging.Formatter(
            '{"timestamp":"%(asctime)s","level":"%(levelname)s",'
            '"logger":"%(name)s","message":"%(message)s",'
            '"trace_id":"%(otelTraceID)s","span_id":"%(otelSpanID)s"}'
        )
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel(config.get("level", "INFO"))

# Global telemetry instance
telemetry = TelemetryManager()
```

### 2. Performance Monitoring

**File**: `src/openapi_doc_generator/monitoring.py`

```python
"""Performance monitoring and metrics collection."""

import time
import psutil
import threading
from typing import Dict, Any, Optional
from contextlib import contextmanager
from dataclasses import dataclass, asdict

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    operation: str
    duration_ms: float
    memory_usage_mb: float
    cpu_percent: float
    timestamp: float
    correlation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class PerformanceMonitor:
    """Advanced performance monitoring."""
    
    def __init__(self):
        self.metrics = []
        self.active_operations = {}
        self._lock = threading.Lock()
    
    @contextmanager
    def monitor_operation(self, operation: str, correlation_id: str = None):
        """Context manager for monitoring operations."""
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_cpu = psutil.cpu_percent()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            end_cpu = psutil.cpu_percent()
            
            metrics = PerformanceMetrics(
                operation=operation,
                duration_ms=(end_time - start_time) * 1000,
                memory_usage_mb=end_memory - start_memory,
                cpu_percent=(end_cpu + start_cpu) / 2,
                timestamp=time.time(),
                correlation_id=correlation_id
            )
            
            self._record_metrics(metrics)
    
    def _record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        with self._lock:
            self.metrics.append(metrics)
            
        # Emit to telemetry system
        if hasattr(telemetry, 'meter') and telemetry.meter:
            duration_histogram = telemetry.meter.create_histogram(
                "operation_duration_ms",
                description="Operation duration in milliseconds"
            )
            duration_histogram.record(
                metrics.duration_ms,
                {"operation": metrics.operation}
            )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        if not self.metrics:
            return {}
        
        operations = {}
        for metric in self.metrics:
            if metric.operation not in operations:
                operations[metric.operation] = []
            operations[metric.operation].append(metric.duration_ms)
        
        summary = {}
        for op, durations in operations.items():
            summary[op] = {
                "count": len(durations),
                "avg_duration_ms": sum(durations) / len(durations),
                "min_duration_ms": min(durations),
                "max_duration_ms": max(durations),
                "p95_duration_ms": sorted(durations)[int(0.95 * len(durations))]
            }
        
        return summary

# Global performance monitor
performance_monitor = PerformanceMonitor()
```

### 3. Health Check Endpoint

**File**: `src/openapi_doc_generator/health_check.py`

```python
"""Health check and readiness probe implementation."""

import json
import time
import psutil
from typing import Dict, Any
from http.server import HTTPServer, BaseHTTPRequestHandler

class HealthCheckHandler(BaseHTTPRequestHandler):
    """HTTP handler for health checks."""
    
    def do_GET(self):
        """Handle GET requests for health checks."""
        if self.path == "/health":
            self._handle_health_check()
        elif self.path == "/ready":
            self._handle_readiness_check()
        elif self.path == "/metrics":
            self._handle_metrics()
        else:
            self._send_error(404, "Not Found")
    
    def _handle_health_check(self):
        """Handle liveness probe."""
        health_data = {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "0.1.0",
            "uptime": time.time() - start_time
        }
        self._send_json_response(200, health_data)
    
    def _handle_readiness_check(self):
        """Handle readiness probe."""
        ready = self._check_dependencies()
        status_code = 200 if ready["ready"] else 503
        self._send_json_response(status_code, ready)
    
    def _handle_metrics(self):
        """Handle metrics endpoint."""
        metrics = {
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent
            },
            "performance": performance_monitor.get_metrics_summary()
        }
        self._send_json_response(200, metrics)
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """Check system dependencies and readiness."""
        checks = {
            "python_version": True,  # Always true if running
            "memory_available": psutil.virtual_memory().percent < 90,
            "disk_space": psutil.disk_usage('/').percent < 90
        }
        
        return {
            "ready": all(checks.values()),
            "checks": checks,
            "timestamp": time.time()
        }
    
    def _send_json_response(self, status_code: int, data: Dict[str, Any]):
        """Send JSON response."""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _send_error(self, status_code: int, message: str):
        """Send error response."""
        self.send_error(status_code, message)

def start_health_server(host: str = "0.0.0.0", port: int = 8080):
    """Start health check server."""
    global start_time
    start_time = time.time()
    
    server = HTTPServer((host, port), HealthCheckHandler)
    server.serve_forever()
```

### 4. Configuration

**File**: `telemetry.yaml`

```yaml
# Telemetry configuration
telemetry:
  service_name: "openapi-doc-generator"
  
  tracing:
    enabled: true
    jaeger_host: "jaeger"
    jaeger_port: 6831
    sample_rate: 0.1
  
  metrics:
    enabled: true
    prometheus_port: 9090
    collection_interval: 30
  
  logging:
    structured: true
    level: "INFO"
    correlation_enabled: true

# Health check configuration
health:
  port: 8080
  host: "0.0.0.0"
  endpoints:
    - path: "/health"
      type: "liveness"
    - path: "/ready"
      type: "readiness"
    - path: "/metrics"
      type: "metrics"
```

## Deployment Integration

### Docker Compose Observability Stack

**File**: `docker-compose.observability.yml`

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8080:8080"
    environment:
      - TELEMETRY_CONFIG=/app/telemetry.yaml
    volumes:
      - ./telemetry.yaml:/app/telemetry.yaml
    depends_on:
      - jaeger
      - prometheus

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"
      - "6831:6831/udp"
    environment:
      - COLLECTOR_OTLP_ENABLED=true

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
```

## Usage Examples

### Application Integration
```python
# In main application
from openapi_doc_generator.telemetry import telemetry
from openapi_doc_generator.monitoring import performance_monitor

# Initialize telemetry
telemetry.initialize(config)

# Monitor operations
with performance_monitor.monitor_operation("route_discovery"):
    # Perform route discovery
    routes = discover_routes(app_file)

# Create spans for tracing
with telemetry.tracer.start_as_current_span("generate_docs") as span:
    span.set_attribute("app_file", app_file)
    docs = generate_documentation(routes)
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: openapi-doc-generator
spec:
  template:
    spec:
      containers:
      - name: app
        image: openapi-doc-generator:latest
        ports:
        - containerPort: 8080
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

## Monitoring Dashboards

Pre-configured Grafana dashboards are included for:
- Application performance metrics
- System resource utilization
- Error rates and latency percentiles
- Distributed tracing visualization
- Custom business metrics

## Next Steps

1. Implement telemetry in core modules
2. Configure production monitoring stack
3. Set up alerting rules
4. Create custom dashboards
5. Integrate with incident management