# Production Monitoring & Observability

## Overview

This document outlines the monitoring and observability strategy for the OpenAPI Documentation Generator, tailored for production deployments and operational excellence.

## Health Check Endpoints

### Application Health
The application includes built-in health monitoring via `health_server.py`:

```python
# Health check endpoint available at http://localhost:8000/health
{
  "status": "healthy",
  "timestamp": "2025-07-29T00:00:00Z",
  "version": "1.0.0",
  "checks": {
    "ast_parsing": "ok",
    "template_engine": "ok",
    "filesystem": "ok"
  }
}
```

### Docker Health Checks
Dockerfile includes comprehensive health checks:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"
```

## Performance Monitoring

### Built-in Metrics
The application includes performance monitoring in `monitoring.py`:

- **Response Time**: API documentation generation latency
- **Memory Usage**: Peak memory consumption during processing
- **CPU Utilization**: Processing time for AST analysis
- **Cache Hit Rates**: Template and AST caching efficiency

### Metrics Collection
```python
# Example metrics collection
metrics = {
    "generation_time_ms": 1250,
    "ast_nodes_processed": 500,
    "template_renders": 15,
    "cache_hits": 12,
    "cache_misses": 3,
    "memory_peak_mb": 45
}
```

## Logging Configuration

### Structured Logging
Application uses structured JSON logging via `logging_config.py`:

```json
{
  "timestamp": "2025-07-29T00:00:00Z",
  "level": "INFO",
  "logger": "openapi_doc_generator",
  "message": "Generated documentation for FastAPI application",
  "extra": {
    "framework": "fastapi",
    "routes_discovered": 25,
    "generation_time_ms": 1250,
    "output_format": "markdown"
  }
}
```

### Log Levels & Categories
- **DEBUG**: AST parsing details, template rendering
- **INFO**: Generation success, performance metrics
- **WARNING**: Deprecated patterns, missing documentation
- **ERROR**: Generation failures, configuration issues
- **CRITICAL**: System-level failures

## Monitoring Stack Integration

### Prometheus Metrics
Create `prometheus_metrics.py` for production monitoring:

```python
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
generation_requests = Counter('doc_generation_requests_total', 'Total documentation generation requests')
generation_duration = Histogram('doc_generation_duration_seconds', 'Time spent generating documentation')
active_generations = Gauge('doc_generation_active', 'Number of active documentation generations')
```

### Grafana Dashboard
Recommended dashboard panels:

1. **Generation Success Rate**: Success/failure ratio over time
2. **Response Time Distribution**: P50, P95, P99 latencies
3. **Memory Usage**: Peak and average memory consumption
4. **Error Rate**: Error frequency by type
5. **Cache Performance**: Hit/miss ratios
6. **Framework Usage**: Distribution of detected frameworks

### Log Aggregation
Integration with log aggregation systems:

- **Elasticsearch/OpenSearch**: For log search and analysis
- **Fluentd/Fluent Bit**: For log collection and forwarding
- **Loki**: For lightweight log aggregation

Example Fluentd configuration:
```xml
<source>
  @type tail
  path /var/log/openapi-doc-generator/*.log
  pos_file /var/log/fluentd/openapi-doc-generator.log.pos
  tag openapi.doc.generator
  format json
</source>
```

## Alerting Strategy

### Critical Alerts (Immediate Response)
- **Service Down**: Health check failures
- **High Error Rate**: >5% error rate over 5 minutes
- **Memory Leak**: Sustained memory growth >80% for 10 minutes
- **Response Time**: P95 latency >10 seconds

### Warning Alerts (Investigation Required)
- **Moderate Error Rate**: >2% error rate over 15 minutes
- **Cache Miss Rate**: <50% cache hit rate over 30 minutes
- **Slow Generation**: P95 latency >5 seconds over 15 minutes
- **High CPU**: >80% CPU utilization for 15 minutes

### Example Alert Configuration (Prometheus/Alertmanager)
```yaml
groups:
  - name: openapi-doc-generator
    rules:
      - alert: HighErrorRate
        expr: rate(doc_generation_errors_total[5m]) / rate(doc_generation_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate in documentation generation"
          description: "Error rate is {{ $value | humanizePercentage }} over the last 5 minutes"
      
      - alert: SlowGeneration
        expr: histogram_quantile(0.95, rate(doc_generation_duration_seconds_bucket[15m])) > 10
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Slow documentation generation"
          description: "95th percentile latency is {{ $value }}s over the last 15 minutes"
```

## Application Performance Monitoring (APM)

### Distributed Tracing
For microservice environments, implement distributed tracing:

```python
import opentelemetry
from opentelemetry.trace import get_tracer

tracer = get_tracer(__name__)

def generate_documentation(app_path: str) -> DocumentationResult:
    with tracer.start_as_current_span("generate_documentation") as span:
        span.set_attribute("app.path", app_path)
        span.set_attribute("app.framework", detected_framework)
        
        # Document generation logic
        result = process_application(app_path)
        
        span.set_attribute("doc.routes_count", len(result.routes))
        span.set_attribute("doc.generation_time_ms", result.generation_time)
        
        return result
```

### Custom Metrics Collection
```python
def collect_custom_metrics():
    """Collect application-specific metrics"""
    return {
        "routes_discovered": len(discovered_routes),
        "plugins_loaded": len(active_plugins),
        "templates_cached": len(template_cache),
        "ast_cache_size": sys.getsizeof(ast_cache),
        "total_processing_time": processing_timer.elapsed(),
        "framework_distribution": Counter(detected_frameworks)
    }
```

## Infrastructure Monitoring

### Container Monitoring
Monitor container-specific metrics:

- **Container Resource Usage**: CPU, memory, disk I/O
- **Container Health**: Restart counts, exit codes
- **Image Security**: Vulnerability scan results
- **Network Performance**: Request/response times

### Kubernetes Monitoring (if applicable)
```yaml
apiVersion: v1
kind: Service
metadata:
  name: openapi-doc-generator-metrics
  labels:
    app: openapi-doc-generator
spec:
  ports:
  - port: 8080
    name: metrics
  selector:
    app: openapi-doc-generator
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: openapi-doc-generator
spec:
  selector:
    matchLabels:
      app: openapi-doc-generator
  endpoints:
  - port: metrics
```

## Security Monitoring

### Security Event Logging
Log security-relevant events:

```python
security_logger = logging.getLogger("security")

def log_security_event(event_type: str, details: dict):
    security_logger.warning(
        "Security event detected",
        extra={
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details,
            "source_ip": request.remote_addr if request else None
        }
    )
```

### Security Metrics
- **Failed Authentication Attempts**: If auth is implemented
- **Suspicious Input Patterns**: Potential injection attempts
- **File Access Violations**: Unauthorized file access attempts
- **Rate Limiting Triggers**: Excessive request patterns

## Business Metrics

### Usage Analytics
Track business-relevant metrics:

- **Documentation Generation Volume**: Requests per day/hour
- **Framework Popularity**: Most frequently documented frameworks
- **Feature Usage**: Which output formats are most popular
- **User Success Rate**: Successful vs. failed generations

### Dashboard KPIs
Key performance indicators for stakeholders:

1. **Availability**: 99.9% uptime target
2. **Performance**: <3 second average generation time
3. **Success Rate**: >95% successful generations
4. **User Satisfaction**: Based on error rates and feedback

## Runbook Integration

### Automated Remediation
For common issues, implement automated responses:

```yaml
# Example: Auto-restart on memory issues
- alert: HighMemoryUsage
  expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9
  for: 5m
  annotations:
    action: "restart_container"
    runbook_url: "https://docs.company.com/runbooks/memory-issues"
```

### Escalation Procedures
1. **Level 1**: Automated remediation attempts
2. **Level 2**: On-call engineer notification
3. **Level 3**: Senior engineer escalation
4. **Level 4**: Management and stakeholder notification

## Configuration Management

### Environment-Specific Settings
```python
MONITORING_CONFIG = {
    "development": {
        "log_level": "DEBUG",
        "metrics_enabled": False,
        "health_check_interval": 60
    },
    "staging": {
        "log_level": "INFO",
        "metrics_enabled": True,
        "health_check_interval": 30
    },
    "production": {
        "log_level": "WARNING",
        "metrics_enabled": True,
        "health_check_interval": 15,
        "alert_thresholds": {
            "error_rate": 0.02,
            "response_time_p95": 5.0,
            "memory_usage": 0.8
        }
    }
}
```

## Monitoring Checklist

### Pre-Deployment
- [ ] Health check endpoints configured
- [ ] Logging configuration verified
- [ ] Metrics collection enabled
- [ ] Alert rules defined and tested
- [ ] Dashboard panels created
- [ ] Runbooks documented

### Post-Deployment
- [ ] Health checks responding correctly
- [ ] Metrics being collected
- [ ] Logs flowing to aggregation system
- [ ] Alerts firing appropriately
- [ ] Dashboard displaying accurate data
- [ ] Performance baseline established

## Continuous Improvement

### Monitoring Evolution
1. **Weekly**: Review alert noise and adjust thresholds
2. **Monthly**: Analyze performance trends and capacity planning
3. **Quarterly**: Evaluate monitoring tool effectiveness
4. **Annually**: Comprehensive monitoring strategy review

### Feedback Loop
- Incorporate user feedback into monitoring strategy
- Use monitoring data to inform feature development
- Continuously refine alerting based on operational experience
- Share monitoring insights with development team for optimization