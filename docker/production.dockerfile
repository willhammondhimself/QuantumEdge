# QuantumEdge Production Dockerfile
# Multi-stage build for optimized production deployment

# Build stage
FROM python:3.9-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Add metadata
LABEL maintainer="QuantumEdge Team" \
      org.opencontainers.image.title="QuantumEdge" \
      org.opencontainers.image.description="Quantum-inspired portfolio optimization" \
      org.opencontainers.image.url="https://github.com/yourusername/quantumedge" \
      org.opencontainers.image.source="https://github.com/yourusername/quantumedge" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /app
COPY requirements.txt requirements-minimal.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim as production

# Create non-root user
RUN groupadd -r quantumedge && useradd -r -g quantumedge quantumedge

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Copy configuration files
COPY pyproject.toml setup.py ./

# Install the package
RUN pip install -e . --no-deps

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs && \
    chown -R quantumedge:quantumedge /app

# Switch to non-root user
USER quantumedge

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]