# Multi-stage build for optimized image size
FROM python:3.13-slim AS builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION=0.1.0
ARG VCS_REF

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install dependencies in builder stage
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir .

# Production stage
FROM python:3.13-slim AS production

# Set labels for metadata
LABEL maintainer="Terragon Labs" \
      version="${VERSION}" \
      description="OpenAPI Documentation Generator" \
      build-date="${BUILD_DATE}" \
      vcs-ref="${VCS_REF}"

# Create non-root user for security
RUN groupadd -r openapi && useradd -r -g openapi -u 1000 openapi

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY --chown=openapi:openapi . .

# Install the application
RUN pip install --no-cache-dir --no-deps .

# Create output directory with proper permissions
RUN mkdir -p /app/output && chown -R openapi:openapi /app

# Switch to non-root user
USER openapi

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/openapi/.local/bin:$PATH"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import openapi_doc_generator; print('OK')" || exit 1

# Set default working directory for mounted volumes
WORKDIR /workspace

# Default entrypoint and command
ENTRYPOINT ["openapi-doc-generator"]
CMD ["--help"]