# Alerting and Incident Response Runbook

## Overview

This runbook provides procedures for monitoring alerts, incident response, and system recovery for the OpenAPI Doc Generator.

## Alert Definitions

### Critical Alerts

#### High Error Rate
- **Condition**: Error rate > 5% over 5 minutes
- **Impact**: Service degradation affecting users
- **Escalation**: Immediate notification to on-call engineer

```yaml
alert: HighErrorRate
expr: |
  (
    rate(openapi_doc_generator_errors_total[5m]) /
    rate(openapi_doc_generator_requests_total[5m])
  ) * 100 > 5
for: 5m
labels:
  severity: critical
annotations:
  summary: "High error rate detected"
  description: "Error rate is {{ $value }}% over the last 5 minutes"
```

#### Service Unavailable
- **Condition**: Service down for > 1 minute
- **Impact**: Complete service outage
- **Escalation**: Immediate page to on-call team

```yaml
alert: ServiceDown
expr: up{job="openapi-doc-generator"} == 0
for: 1m
labels:
  severity: critical
annotations:
  summary: "Service is down"
  description: "OpenAPI Doc Generator service has been down for more than 1 minute"
```

#### High Memory Usage
- **Condition**: Memory usage > 90% for 10 minutes
- **Impact**: Potential service crashes
- **Escalation**: Warning to development team

```yaml
alert: HighMemoryUsage
expr: |
  (
    process_memory_rss_bytes{job="openapi-doc-generator"} /
    container_spec_memory_limit_bytes
  ) * 100 > 90
for: 10m
labels:
  severity: warning
annotations:
  summary: "High memory usage detected"
  description: "Memory usage is {{ $value }}% of available memory"
```

### Warning Alerts

#### Slow Response Time
- **Condition**: 95th percentile response time > 10s
- **Impact**: Poor user experience
- **Escalation**: Notification to development team

```yaml
alert: SlowResponseTime
expr: |
  histogram_quantile(0.95,
    rate(openapi_doc_generator_request_duration_seconds_bucket[5m])
  ) > 10
for: 15m
labels:
  severity: warning
annotations:
  summary: "Slow response times detected"
  description: "95th percentile response time is {{ $value }}s"
```

#### Low Cache Hit Rate
- **Condition**: AST cache hit rate < 50%
- **Impact**: Increased processing time
- **Escalation**: Development team notification

```yaml
alert: LowCacheHitRate
expr: |
  (
    rate(openapi_doc_generator_ast_cache_hits_total[15m]) /
    rate(openapi_doc_generator_ast_cache_requests_total[15m])
  ) * 100 < 50
for: 15m
labels:
  severity: warning
annotations:
  summary: "Low AST cache hit rate"
  description: "Cache hit rate is {{ $value }}%"
```

## Incident Response Procedures

### Severity Levels

#### SEV1 - Critical
- **Response Time**: 15 minutes
- **Examples**: Service completely down, data loss, security breach
- **Actions**: Immediate investigation and resolution

#### SEV2 - High
- **Response Time**: 1 hour
- **Examples**: Significant performance degradation, partial outage
- **Actions**: Investigation within business hours

#### SEV3 - Medium
- **Response Time**: 4 hours
- **Examples**: Minor performance issues, non-critical bugs
- **Actions**: Scheduled resolution

#### SEV4 - Low
- **Response Time**: Next business day
- **Examples**: Documentation issues, enhancement requests
- **Actions**: Planned work

### Incident Response Workflow

#### 1. Alert Detection
```bash
# Automated alert triggers
# - Prometheus AlertManager
# - Grafana notifications
# - PagerDuty integration
# - Slack notifications
```

#### 2. Initial Assessment
```bash
# Check service health
curl -f http://openapi-doc-generator:8080/health || echo "Service down"

# Check recent logs
docker logs --tail=100 openapi-doc-generator

# Check resource usage
docker stats openapi-doc-generator
```

#### 3. Triage and Classification
- Determine severity level
- Identify affected components
- Estimate impact scope
- Assign incident commander

#### 4. Investigation
```bash
# Check application metrics
curl http://openapi-doc-generator:8080/metrics

# Analyze error patterns
grep "ERROR" /var/log/openapi-doc-generator/*.log | tail -50

# Check dependency health
curl -f http://dependency-service:8080/health
```

#### 5. Resolution
- Apply immediate fixes
- Deploy hotfixes if needed
- Verify resolution
- Update stakeholders

#### 6. Post-Incident Review
- Conduct blameless postmortem
- Document lessons learned
- Update runbooks
- Implement preventive measures

## Common Troubleshooting Scenarios

### High Error Rate

#### Symptoms
- Increased error logs
- User complaints
- Alert notifications

#### Investigation Steps
```bash
# Check error distribution
grep -c "ERROR" /var/log/openapi-doc-generator/*.log

# Analyze error types
grep "ERROR" /var/log/openapi-doc-generator/*.log | \
  awk '{print $NF}' | sort | uniq -c | sort -nr

# Check recent deployments
git log --oneline -10
```

#### Common Causes
- Invalid input files
- Dependency failures
- Configuration errors
- Resource exhaustion

#### Resolution Steps
1. Identify error pattern
2. Check recent changes
3. Rollback if necessary
4. Apply targeted fix
5. Monitor recovery

### Memory Leaks

#### Symptoms
- Gradually increasing memory usage
- Out of memory errors
- Container restarts

#### Investigation Steps
```bash
# Monitor memory over time
watch 'docker stats --no-stream openapi-doc-generator'

# Check for memory leaks in logs
grep -i "memory\|oom" /var/log/openapi-doc-generator/*.log

# Analyze memory usage patterns
curl http://openapi-doc-generator:8080/metrics | grep memory
```

#### Resolution Steps
1. Restart service for immediate relief
2. Collect memory dumps for analysis
3. Review recent code changes
4. Apply memory optimization fixes
5. Monitor for recurrence

### Performance Degradation

#### Symptoms
- Slow response times
- High CPU usage
- User complaints

#### Investigation Steps
```bash
# Check response times
curl -w "@curl-format.txt" -o /dev/null -s \
  "http://openapi-doc-generator:8080/api/analyze"

# Analyze performance metrics
curl http://openapi-doc-generator:8080/metrics | grep duration

# Check for resource contention
top -p $(pgrep -f openapi-doc-generator)
```

#### Resolution Steps
1. Identify performance bottleneck
2. Scale resources if needed
3. Optimize code paths
4. Enable performance caching
5. Monitor improvement

## Recovery Procedures

### Service Restart
```bash
# Graceful restart
docker-compose restart openapi-doc-generator

# Force restart if needed
docker-compose down
docker-compose up -d
```

### Rollback Procedure
```bash
# Identify last known good version
git tag --sort=-version:refname | head -5

# Deploy previous version
git checkout v1.2.3
docker build -t openapi-doc-generator:rollback .
docker-compose up -d
```

### Data Recovery
```bash
# Backup current state
docker exec openapi-doc-generator \
  tar -czf /tmp/backup-$(date +%Y%m%d).tar.gz /app/data

# Restore from backup
docker exec openapi-doc-generator \
  tar -xzf /tmp/backup-20231201.tar.gz -C /
```

## Monitoring Tools Integration

### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
scrape_configs:
- job_name: 'openapi-doc-generator'
  static_configs:
  - targets: ['openapi-doc-generator:8080']
  scrape_interval: 5s
  metrics_path: /metrics
```

### Grafana Dashboards
- Application Performance Dashboard
- Error Rate and Trends
- Resource Usage Monitoring
- Business Metrics Overview

### Log Aggregation
```yaml
# filebeat.yml
filebeat.inputs:
- type: log
  paths:
    - /var/log/openapi-doc-generator/*.log
  fields:
    service: openapi-doc-generator
output.elasticsearch:
  hosts: ["elasticsearch:9200"]
```

## Escalation Matrix

### On-Call Rotation
- **Primary**: Senior Engineer (15-minute response)
- **Secondary**: Platform Team Lead (30-minute response)
- **Escalation**: Engineering Manager (1-hour response)

### Contact Information
- **PagerDuty**: Integration key configured
- **Slack**: #alerts-openapi-doc-generator
- **Email**: oncall-team@company.com

### Escalation Triggers
- SEV1 incidents after 30 minutes
- SEV2 incidents after 2 hours
- Multiple related incidents
- Security-related incidents

## Prevention and Improvement

### Proactive Monitoring
- Regular health checks
- Performance trend analysis
- Capacity planning
- Security scanning

### Continuous Improvement
- Monthly alert review
- Quarterly runbook updates
- Annual disaster recovery testing
- Post-incident action tracking

### Documentation Maintenance
- Keep runbooks current
- Update contact information
- Review and test procedures
- Training for new team members

This runbook ensures rapid response to incidents and maintains high service availability for the OpenAPI Doc Generator.