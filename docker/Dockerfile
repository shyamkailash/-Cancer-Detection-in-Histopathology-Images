# syntax=docker/dockerfile:1

# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements/base.txt /app/requirements/base.txt
RUN pip install --no-cache-dir --user -r requirements/base.txt

# Stage 2: Final runtime image
FROM python:3.12-slim

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

# Copy Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Ensure the local bin is in PATH
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy application code and ML packages
COPY backend/ /app/backend/
COPY ml/ /app/ml/
COPY data/metadata/ /app/data/metadata/
COPY requirements/ /app/requirements/

# Set working directory
WORKDIR /app/backend

# Change ownership
RUN chown -R appuser:appuser /app

# Copy and set up entrypoint
COPY --chown=appuser:appuser docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import django; import os; os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.settings.production'); django.setup(); from django.db import connection; connection.cursor().execute('SELECT 1')"]

# Entry point
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120"]