#!/bin/bash
set -e

# Autonomous SDLC Production Entrypoint
# Quantum-enhanced startup with comprehensive health checks

echo "🚀 Starting Autonomous SDLC System..."

# Function to log with timestamp
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if service is ready
wait_for_service() {
    local service_name="$1"
    local host="$2"
    local port="$3"
    local max_attempts="${4:-30}"
    
    log "Waiting for $service_name at $host:$port..."
    
    for i in $(seq 1 $max_attempts); do
        if nc -z "$host" "$port" 2>/dev/null; then
            log "✓ $service_name is ready"
            return 0
        fi
        log "⏳ Waiting for $service_name... (attempt $i/$max_attempts)"
        sleep 2
    done
    
    log "❌ $service_name failed to start within $(($max_attempts * 2)) seconds"
    return 1
}

# Pre-start checks and setup
log "🔧 Running pre-start setup..."

# Create required directories
mkdir -p /app/logs /app/cache /app/tmp
chmod 755 /app/logs /app/cache /app/tmp

# Run pre-start script if it exists
if [ -f "/app/prestart.sh" ]; then
    log "📋 Running pre-start script..."
    /app/prestart.sh
fi

# Wait for dependencies if in Docker Compose environment
if [ "${WAIT_FOR_SERVICES:-false}" = "true" ]; then
    log "⏳ Waiting for dependent services..."
    
    # Wait for Redis if configured
    if [ -n "${REDIS_URL}" ]; then
        redis_host=$(echo $REDIS_URL | sed -E 's|redis://([^:/]+).*|\1|')
        redis_port=$(echo $REDIS_URL | sed -E 's|redis://[^:]+:([0-9]+).*|\1|')
        wait_for_service "Redis" "${redis_host:-redis}" "${redis_port:-6379}"
    fi
    
    # Wait for PostgreSQL if configured  
    if [ -n "${DATABASE_URL}" ]; then
        db_host=$(echo $DATABASE_URL | sed -E 's|postgresql://[^@]+@([^:/]+).*|\1|')
        db_port=$(echo $DATABASE_URL | sed -E 's|postgresql://[^@]+@[^:]+:([0-9]+).*|\1|')
        wait_for_service "PostgreSQL" "${db_host:-postgres}" "${db_port:-5432}"
    fi
fi

# Initialize autonomous components
log "🧠 Initializing Autonomous SDLC Components..."

# Set Python path
export PYTHONPATH="/app/src:${PYTHONPATH}"

# Run database migrations if needed
if [ "${RUN_MIGRATIONS:-false}" = "true" ]; then
    log "🗃️  Running database migrations..."
    python3 /app/src/openapi_doc_generator/cli.py migrate || true
fi

# Health check before starting
log "🏥 Performing initial health check..."
python3 -c "
import sys
sys.path.insert(0, '/app/src')

try:
    # Test imports
    from openapi_doc_generator.ai_documentation_agent import create_ai_documentation_agent
    from openapi_doc_generator.autonomous_reliability_engine import create_reliability_engine
    from openapi_doc_generator.advanced_security_guardian import create_security_guardian
    from openapi_doc_generator.quantum_performance_engine import create_quantum_performance_engine
    
    print('✓ All autonomous components importable')
    
    # Test basic functionality
    agent = create_ai_documentation_agent()
    security = create_security_guardian()
    performance = create_quantum_performance_engine()
    reliability = create_reliability_engine()
    
    print('✓ All autonomous components initialized')
    
except Exception as e:
    print(f'❌ Component initialization failed: {e}')
    sys.exit(1)
" || exit 1

# Determine startup mode based on environment
STARTUP_MODE="${STARTUP_MODE:-gunicorn}"

log "🎯 Starting in $STARTUP_MODE mode..."

case "$STARTUP_MODE" in
    "gunicorn")
        log "🦄 Starting with Gunicorn (Production)"
        exec gunicorn \
            --config /app/gunicorn.conf.py \
            --bind 0.0.0.0:8000 \
            --access-logfile /app/logs/access.log \
            --error-logfile /app/logs/error.log \
            --log-level "${LOG_LEVEL:-info}" \
            "openapi_doc_generator.cli:app"
        ;;
    
    "uvicorn")
        log "⚡ Starting with Uvicorn (Development)"
        exec uvicorn \
            openapi_doc_generator.cli:app \
            --host 0.0.0.0 \
            --port 8000 \
            --log-level "${LOG_LEVEL:-info}" \
            --access-log \
            --use-colors
        ;;
    
    "autonomous")
        log "🤖 Starting Autonomous SDLC System"
        exec python3 /app/start-autonomous.py
        ;;
    
    "cli")
        log "💻 Starting CLI mode"
        exec python3 /app/src/openapi_doc_generator/cli.py "$@"
        ;;
    
    "shell")
        log "🐚 Starting interactive shell"
        exec /bin/bash
        ;;
    
    *)
        log "❌ Unknown startup mode: $STARTUP_MODE"
        log "Available modes: gunicorn, uvicorn, autonomous, cli, shell"
        exit 1
        ;;
esac