# Metrics Dashboard Configuration

## Overview

This document defines the metrics dashboards for monitoring the OpenAPI Doc Generator application performance, health, and business metrics.

## Dashboard Architecture

### Dashboard Hierarchy
```
Production Dashboards/
├── Executive Summary
├── Application Performance
├── Infrastructure Health
├── Security Posture
├── Business Metrics
└── Troubleshooting
```

## Executive Summary Dashboard

### Key Performance Indicators (KPIs)
```json
{
  "dashboard": "executive-summary",
  "refresh": "5m",
  "panels": [
    {
      "title": "Service Availability",
      "type": "singlestat",
      "metric": "up{job='openapi-doc-generator'}",
      "target": "99.9%",
      "format": "percentage"
    },
    {
      "title": "Documentation Success Rate",
      "type": "singlestat", 
      "metric": "rate(openapi_doc_generator_successful_generations_total[24h])",
      "target": "95%",
      "format": "percentage"
    },
    {
      "title": "Active Users (24h)",
      "type": "singlestat",
      "metric": "count(increase(openapi_doc_generator_requests_total[24h]) by (user))",
      "format": "number"
    },
    {
      "title": "Error Rate",
      "type": "graph",
      "metric": "rate(openapi_doc_generator_errors_total[5m])",
      "threshold": "1%",
      "timeRange": "24h"
    }
  ]
}
```

### Business Impact Metrics
- Documentation generations per day
- User adoption trends
- Feature usage distribution
- Cost per documentation generation

## Application Performance Dashboard

### Response Time Metrics
```json
{
  "title": "Response Time Distribution",
  "type": "heatmap",
  "targets": [
    {
      "expr": "histogram_quantile(0.50, rate(openapi_doc_generator_request_duration_seconds_bucket[5m]))",
      "legendFormat": "50th percentile"
    },
    {
      "expr": "histogram_quantile(0.95, rate(openapi_doc_generator_request_duration_seconds_bucket[5m]))",
      "legendFormat": "95th percentile"
    },
    {
      "expr": "histogram_quantile(0.99, rate(openapi_doc_generator_request_duration_seconds_bucket[5m]))",
      "legendFormat": "99th percentile"
    }
  ]
}
```

### Throughput Metrics
```json
{
  "title": "Request Throughput",
  "type": "graph",
  "targets": [
    {
      "expr": "rate(openapi_doc_generator_requests_total[1m])",
      "legendFormat": "Requests/sec"
    },
    {
      "expr": "rate(openapi_doc_generator_successful_generations_total[1m])",
      "legendFormat": "Successful generations/sec"
    }
  ]
}
```

### Error Analysis
```json
{
  "title": "Error Rate by Type",
  "type": "piechart",
  "targets": [
    {
      "expr": "sum(rate(openapi_doc_generator_errors_total[5m])) by (error_type)",
      "legendFormat": "{{ error_type }}"
    }
  ]
}
```

### Cache Performance
```json
{
  "title": "AST Cache Performance",
  "type": "graph",
  "targets": [
    {
      "expr": "(rate(openapi_doc_generator_ast_cache_hits_total[5m]) / rate(openapi_doc_generator_ast_cache_requests_total[5m])) * 100",
      "legendFormat": "Cache Hit Rate %"
    },
    {
      "expr": "rate(openapi_doc_generator_ast_cache_misses_total[5m])",
      "legendFormat": "Cache Misses/sec"
    }
  ]
}
```

## Infrastructure Health Dashboard

### Resource Utilization
```json
{
  "title": "CPU and Memory Usage",
  "type": "graph",
  "targets": [
    {
      "expr": "rate(process_cpu_seconds_total{job='openapi-doc-generator'}[5m]) * 100",
      "legendFormat": "CPU Usage %"
    },
    {
      "expr": "(process_resident_memory_bytes{job='openapi-doc-generator'} / 1024 / 1024)",
      "legendFormat": "Memory Usage MB"
    }
  ]
}
```

### Container Health
```json
{
  "title": "Container Metrics",
  "type": "table",
  "targets": [
    {
      "expr": "up{job='openapi-doc-generator'}",
      "legendFormat": "Status"
    },
    {
      "expr": "time() - process_start_time_seconds{job='openapi-doc-generator'}",
      "legendFormat": "Uptime (seconds)"
    },
    {
      "expr": "process_open_fds{job='openapi-doc-generator'}",
      "legendFormat": "Open File Descriptors"
    }
  ]
}
```

### Network Metrics
```json
{
  "title": "Network I/O",
  "type": "graph",
  "targets": [
    {
      "expr": "rate(container_network_receive_bytes_total{name='openapi-doc-generator'}[5m])",
      "legendFormat": "Bytes Received/sec"
    },
    {
      "expr": "rate(container_network_transmit_bytes_total{name='openapi-doc-generator'}[5m])",
      "legendFormat": "Bytes Transmitted/sec"
    }
  ]
}
```

## Security Posture Dashboard

### Security Metrics
```json
{
  "title": "Security Events",
  "type": "graph",
  "targets": [
    {
      "expr": "rate(openapi_doc_generator_security_events_total[5m])",
      "legendFormat": "Security Events/sec"
    },
    {
      "expr": "openapi_doc_generator_vulnerabilities_total",
      "legendFormat": "Known Vulnerabilities"
    }
  ]
}
```

### Authentication Metrics
```json
{
  "title": "Authentication Events",
  "type": "graph",
  "targets": [
    {
      "expr": "rate(openapi_doc_generator_auth_attempts_total[5m])",
      "legendFormat": "Auth Attempts/sec"
    },
    {
      "expr": "rate(openapi_doc_generator_auth_failures_total[5m])",
      "legendFormat": "Auth Failures/sec"
    }
  ]
}
```

## Business Metrics Dashboard

### Usage Analytics
```json
{
  "title": "Framework Usage Distribution",
  "type": "piechart",
  "targets": [
    {
      "expr": "sum(increase(openapi_doc_generator_framework_detections_total[24h])) by (framework)",
      "legendFormat": "{{ framework }}"
    }
  ]
}
```

### Output Format Preferences
```json
{
  "title": "Output Format Usage",
  "type": "bargraph",
  "targets": [
    {
      "expr": "sum(increase(openapi_doc_generator_output_formats_total[24h])) by (format)",
      "legendFormat": "{{ format }}"
    }
  ]
}
```

### Performance Trends
```json
{
  "title": "Documentation Generation Time Trends",
  "type": "graph",
  "targets": [
    {
      "expr": "histogram_quantile(0.50, rate(openapi_doc_generator_generation_duration_seconds_bucket[1h]))",
      "legendFormat": "Median Generation Time"
    },
    {
      "expr": "avg(rate(openapi_doc_generator_routes_discovered_total[1h]))",
      "legendFormat": "Average Routes per Generation"
    }
  ]
}
```

## Troubleshooting Dashboard

### Real-time Debugging
```json
{
  "title": "Recent Errors",
  "type": "logs",
  "targets": [
    {
      "expr": "{job='openapi-doc-generator'} |= 'ERROR'",
      "limit": 100
    }
  ]
}
```

### Performance Debugging
```json
{
  "title": "Slow Requests",
  "type": "table",
  "targets": [
    {
      "expr": "topk(10, histogram_quantile(0.99, rate(openapi_doc_generator_request_duration_seconds_bucket[5m])))",
      "legendFormat": "{{ endpoint }}"
    }
  ]
}
```

### Resource Debugging
```json
{
  "title": "Memory Usage by Component",
  "type": "graph",
  "targets": [
    {
      "expr": "openapi_doc_generator_memory_usage_bytes by (component)",
      "legendFormat": "{{ component }}"
    }
  ]
}
```

## Dashboard Configuration

### Grafana Dashboard JSON
```json
{
  "dashboard": {
    "id": null,
    "title": "OpenAPI Doc Generator - Application Performance",
    "tags": ["openapi", "performance"],
    "timezone": "UTC",
    "refresh": "30s",
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "templating": {
      "list": [
        {
          "name": "environment",
          "type": "query",
          "query": "label_values(openapi_doc_generator_requests_total, environment)"
        },
        {
          "name": "instance",
          "type": "query", 
          "query": "label_values(openapi_doc_generator_requests_total{environment='$environment'}, instance)"
        }
      ]
    }
  }
}
```

### Alert Integration
```json
{
  "alert": {
    "conditions": [
      {
        "query": {
          "params": ["A", "5m", "now"]
        },
        "reducer": {
          "type": "avg",
          "params": []
        },
        "evaluator": {
          "params": [0.05],
          "type": "gt"
        }
      }
    ],
    "executionErrorState": "alerting",
    "noDataState": "no_data",
    "frequency": "10s",
    "handler": 1,
    "name": "High Error Rate Alert",
    "message": "Error rate has exceeded 5% threshold"
  }
}
```

## Automation and Deployment

### Dashboard as Code
```yaml
# dashboards.yaml
dashboards:
  - name: "executive-summary"
    path: "./dashboards/executive-summary.json"
    folder: "OpenAPI Doc Generator"
  - name: "application-performance" 
    path: "./dashboards/application-performance.json"
    folder: "OpenAPI Doc Generator"
  - name: "infrastructure-health"
    path: "./dashboards/infrastructure-health.json"
    folder: "OpenAPI Doc Generator"
```

### Automated Deployment
```bash
#!/bin/bash
# deploy-dashboards.sh

GRAFANA_URL="http://grafana:3000"
API_KEY="your-api-key"

for dashboard in dashboards/*.json; do
    curl -X POST \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d @"$dashboard" \
        "$GRAFANA_URL/api/dashboards/db"
done
```

### Dashboard Versioning
```bash
# Version control for dashboards
git add dashboards/
git commit -m "Update application performance dashboard"
git tag dashboard-v1.2.0
```

## Maintenance and Updates

### Dashboard Maintenance
- Weekly review of dashboard relevance
- Monthly optimization of query performance
- Quarterly review of metrics and KPIs
- Annual dashboard architecture review

### Performance Optimization
```json
{
  "query_optimization": {
    "recording_rules": [
      {
        "record": "openapi:error_rate_5m",
        "expr": "rate(openapi_doc_generator_errors_total[5m]) / rate(openapi_doc_generator_requests_total[5m])"
      }
    ],
    "caching": {
      "enabled": true,
      "ttl": "5m"
    }
  }
}
```

### Access Control
```yaml
# dashboard-permissions.yaml
permissions:
  viewers: ["developers", "operations"]
  editors: ["senior-engineers", "platform-team"]
  admins: ["engineering-managers"]
```

This comprehensive dashboard configuration provides complete visibility into the OpenAPI Doc Generator's performance, health, and business impact.