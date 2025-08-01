#!/bin/bash

# Terragon Continuous Autonomous Execution Script
# Runs continuous value discovery and execution cycles

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$SCRIPT_DIR/autonomous-execution.log"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check if we're in a git repository
if [ ! -d "$REPO_ROOT/.git" ]; then
    log "ERROR: Not in a git repository"
    exit 1
fi

# Change to repository root
cd "$REPO_ROOT"

log "Starting Terragon continuous autonomous execution"
log "Repository: $(basename "$REPO_ROOT")"
log "Working directory: $(pwd)"

# Function to run autonomous cycle
run_autonomous_cycle() {
    local cycle_start=$(date +%s)
    log "Starting autonomous cycle $1"
    
    # Discovery phase
    log "Phase 1: Value discovery"
    if python3 .terragon/backlog-discovery.py; then
        log "✅ Discovery completed successfully"
        
        # Check if items were discovered
        if [ -f ".terragon/discovered-items.json" ]; then
            item_count=$(python3 -c "import json; print(len(json.load(open('.terragon/discovered-items.json'))))" 2>/dev/null || echo "0")
            log "📊 Discovered $item_count work items"
            
            if [ "$item_count" -gt 0 ]; then
                # Execution phase
                log "Phase 2: Autonomous execution"
                if python3 .terragon/autonomous-executor.py; then
                    log "✅ Execution completed successfully"
                else
                    log "⚠️  Execution completed with issues"
                fi
            else
                log "ℹ️  No work items to execute - repository in excellent shape"
            fi
        else
            log "⚠️  No discovery results found"
        fi
    else
        log "❌ Discovery failed"
        return 1
    fi
    
    local cycle_end=$(date +%s)
    local cycle_duration=$((cycle_end - cycle_start))
    log "Cycle $1 completed in ${cycle_duration}s"
    
    return 0
}

# Function to generate status report
generate_status_report() {
    log "Generating status report"
    
    local report_file="$SCRIPT_DIR/execution-status-$(date +%Y%m%d-%H%M%S).json"
    
    cat > "$report_file" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)",
  "repository": "$(basename "$REPO_ROOT")",
  "execution_mode": "continuous",
  "git_branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "autonomous_status": {
    "backlog_file_exists": $(test -f "AUTONOMOUS_BACKLOG.md" && echo "true" || echo "false"),
    "metrics_file_exists": $(test -f ".terragon/value-metrics.json" && echo "true" || echo "false"),
    "discovered_items_exists": $(test -f ".terragon/discovered-items.json" && echo "true" || echo "false")
  }
}
EOF
    
    if [ -f ".terragon/value-metrics.json" ]; then
        # Merge metrics into status report
        python3 -c "
import json
status = json.load(open('$report_file'))
metrics = json.load(open('.terragon/value-metrics.json'))
status['value_metrics'] = {
    'completed_tasks': metrics.get('valueDelivered', {}).get('completedTasks', 0),
    'total_score': metrics.get('valueDelivered', {}).get('totalScore', 0),
    'backlog_items': metrics.get('backlogMetrics', {}).get('totalItems', 0)
}
json.dump(status, open('$report_file', 'w'), indent=2)
"
    fi
    
    log "Status report generated: $report_file"
}

# Main execution
main() {
    local mode="${1:-single}"
    local max_cycles="${2:-1}"
    local sleep_duration="${3:-3600}"  # 1 hour default
    
    log "Execution mode: $mode"
    log "Max cycles: $max_cycles"
    
    case "$mode" in
        "single")
            run_autonomous_cycle 1
            generate_status_report
            ;;
        "continuous")
            log "Starting continuous execution (sleep: ${sleep_duration}s between cycles)"
            local cycle=1
            while [ $cycle -le $max_cycles ] || [ $max_cycles -eq -1 ]; do
                run_autonomous_cycle $cycle
                
                if [ $cycle -lt $max_cycles ] || [ $max_cycles -eq -1 ]; then
                    log "Sleeping for ${sleep_duration}s before next cycle..."
                    sleep $sleep_duration
                fi
                
                cycle=$((cycle + 1))
            done
            generate_status_report
            ;;
        "discover-only")
            log "Running discovery-only mode"
            python3 .terragon/backlog-discovery.py
            generate_status_report
            ;;
        *)
            log "Unknown mode: $mode"
            echo "Usage: $0 [single|continuous|discover-only] [max_cycles] [sleep_duration]"
            exit 1
            ;;
    esac
    
    log "Autonomous execution completed"
}

# Handle signals for graceful shutdown
trap 'log "Received signal, shutting down gracefully..."; exit 0' SIGTERM SIGINT

# Run main function with all arguments
main "$@"