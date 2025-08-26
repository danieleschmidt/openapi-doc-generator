#!/bin/bash
# Comprehensive health check for Autonomous SDLC System

set -e

# Configuration
HEALTH_ENDPOINT="${HEALTH_ENDPOINT:-http://localhost:9000/health}"
TIMEOUT="${HEALTH_CHECK_TIMEOUT:-10}"
RETRIES="${HEALTH_CHECK_RETRIES:-3}"

# Function to log with timestamp
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] HEALTHCHECK: $1"
}

# Function to check HTTP endpoint
check_http_endpoint() {
    local url="$1"
    local expected_status="${2:-200}"
    local timeout="${3:-$TIMEOUT}"
    
    local response
    response=$(curl -s -o /dev/null -w "%{http_code}" --max-time "$timeout" "$url" 2>/dev/null)
    
    if [ "$response" = "$expected_status" ]; then
        return 0
    else
        log "HTTP check failed: $url returned $response (expected $expected_status)"
        return 1
    fi
}

# Function to check if process is running
check_process() {
    local process_name="$1"
    
    if pgrep -f "$process_name" > /dev/null 2>&1; then
        return 0
    else
        log "Process check failed: $process_name not running"
        return 1
    fi
}

# Function to check memory usage
check_memory() {
    local max_memory_percent="${1:-90}"
    
    local memory_usage
    memory_usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
    
    if [ "$memory_usage" -lt "$max_memory_percent" ]; then
        return 0
    else
        log "Memory check failed: ${memory_usage}% usage (max: ${max_memory_percent}%)"
        return 1
    fi
}

# Function to check disk space
check_disk() {
    local max_disk_percent="${1:-95}"
    local mount_point="${2:-/app}"
    
    local disk_usage
    disk_usage=$(df "$mount_point" | tail -1 | awk '{print $5}' | sed 's/%//')
    
    if [ "$disk_usage" -lt "$max_disk_percent" ]; then
        return 0
    else
        log "Disk check failed: ${disk_usage}% usage on $mount_point (max: ${max_disk_percent}%)"
        return 1
    fi
}

# Main health check function
perform_health_check() {
    local check_type="${1:-basic}"
    
    log "Starting $check_type health check..."
    
    case "$check_type" in
        "basic")
            # Basic HTTP health check
            if check_http_endpoint "$HEALTH_ENDPOINT"; then
                log "✓ Basic health check passed"
                return 0
            else
                log "❌ Basic health check failed"
                return 1
            fi
            ;;
        
        "detailed")
            local failed_checks=0
            
            # HTTP health check
            if ! check_http_endpoint "$HEALTH_ENDPOINT"; then
                ((failed_checks++))
            fi
            
            # Process check
            if ! check_process "python3"; then
                ((failed_checks++))
            fi
            
            # Memory check
            if ! check_memory 90; then
                ((failed_checks++))
            fi
            
            # Disk check
            if ! check_disk 95 "/app"; then
                ((failed_checks++))
            fi
            
            # Component-specific checks
            if [ "${AUTONOMOUS_FEATURES_ENABLED:-true}" = "true" ]; then
                # Check metrics endpoint
                if ! check_http_endpoint "http://localhost:8080/metrics"; then
                    log "⚠️  Metrics endpoint not available"
                    ((failed_checks++))
                fi
                
                # Check if autonomous components are responsive
                python3 -c "
import sys
sys.path.insert(0, '/app/src')
try:
    from openapi_doc_generator import AUTONOMOUS_SDLC_LOADED
    if not AUTONOMOUS_SDLC_LOADED:
        print('Autonomous components not loaded')
        sys.exit(1)
    print('✓ Autonomous components loaded')
except Exception as e:
    print(f'❌ Autonomous component check failed: {e}')
    sys.exit(1)
" || ((failed_checks++))
            fi
            
            if [ "$failed_checks" -eq 0 ]; then
                log "✓ Detailed health check passed"
                return 0
            else
                log "❌ Detailed health check failed ($failed_checks checks failed)"
                return 1
            fi
            ;;
        
        "deep")
            # Deep health check including component functionality
            python3 -c "
import sys
import asyncio
sys.path.insert(0, '/app/src')

async def deep_health_check():
    try:
        from openapi_doc_generator import AUTONOMOUS_SDLC_LOADED
        
        if not AUTONOMOUS_SDLC_LOADED:
            print('❌ Autonomous components not loaded')
            return False
        
        # Test each component
        from openapi_doc_generator.ai_documentation_agent import create_ai_documentation_agent
        from openapi_doc_generator.advanced_security_guardian import create_security_guardian
        from openapi_doc_generator.quantum_performance_engine import create_quantum_performance_engine
        from openapi_doc_generator.autonomous_reliability_engine import create_reliability_engine
        
        # Quick functionality tests
        ai_agent = create_ai_documentation_agent()
        security = create_security_guardian()
        performance = create_quantum_performance_engine()
        reliability = create_reliability_engine()
        
        # Test security status
        security_status = security.get_security_status()
        if security_status['status'] != 'active':
            print(f'❌ Security guardian not active: {security_status}')
            return False
        
        # Test performance reporting
        perf_report = performance.get_performance_report()
        if not isinstance(perf_report, dict):
            print('❌ Performance engine not responding')
            return False
        
        # Test reliability reporting
        reliability_report = reliability.get_reliability_report()
        if 'system_resilience_score' not in reliability_report:
            print('❌ Reliability engine not responding')
            return False
        
        print('✓ Deep health check passed - all components functional')
        return True
        
    except Exception as e:
        print(f'❌ Deep health check failed: {e}')
        return False

result = asyncio.run(deep_health_check())
sys.exit(0 if result else 1)
" 
            return $?
            ;;
        
        *)
            log "❌ Unknown health check type: $check_type"
            return 1
            ;;
    esac
}

# Retry logic
attempt_health_check() {
    local check_type="${1:-basic}"
    local attempt=1
    
    while [ $attempt -le $RETRIES ]; do
        log "Health check attempt $attempt/$RETRIES..."
        
        if perform_health_check "$check_type"; then
            log "✅ Health check successful on attempt $attempt"
            return 0
        fi
        
        if [ $attempt -lt $RETRIES ]; then
            log "❌ Attempt $attempt failed, retrying in 2 seconds..."
            sleep 2
        fi
        
        ((attempt++))
    done
    
    log "💀 All $RETRIES health check attempts failed"
    return 1
}

# Main execution
main() {
    local check_type="${1:-basic}"
    
    # Validate check type
    case "$check_type" in
        "basic"|"detailed"|"deep")
            ;;
        *)
            log "Usage: $0 [basic|detailed|deep]"
            exit 1
            ;;
    esac
    
    # Perform health check with retries
    if attempt_health_check "$check_type"; then
        exit 0
    else
        exit 1
    fi
}

# Script entry point
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi