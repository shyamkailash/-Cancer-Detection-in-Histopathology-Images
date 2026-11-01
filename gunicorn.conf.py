"""
Gunicorn configuration for Cancer Detection Histopathology backend on Render.
Optimized for low-memory environments (Render 512 MB Free Tier limit).
"""

import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# Worker processes: Keep workers at 1 to prevent PyTorch memory duplication.
# Threads handle concurrent I/O efficiently while sharing memory.
workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
threads = int(os.environ.get("GUNICORN_THREADS", "2"))
worker_class = "gthread"

# Request lifecycle and memory recycling
max_requests = 500
max_requests_jitter = 50
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "120"))
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info").lower()
