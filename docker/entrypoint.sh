#!/bin/bash
set -e

echo "========================================"
echo "Cancer Detection Histopathology - Starting"
echo "========================================"

# Wait for PostgreSQL
echo "Waiting for PostgreSQL to be ready..."
MAX_RETRIES=30
RETRIES=0

while [ $RETRIES -lt $MAX_RETRIES ]; do
    if python -c "
import psycopg2
import os
try:
    conn = psycopg2.connect(
        dbname=os.environ.get('POSTGRES_DB', 'cancer_detection'),
        user=os.environ.get('POSTGRES_USER', 'cancer_user'),
        password=os.environ.get('POSTGRES_PASSWORD', 'cancer_password'),
        host=os.environ.get('POSTGRES_HOST', 'db'),
        port=os.environ.get('POSTGRES_PORT', '5432')
    )
    conn.close()
    exit(0)
except Exception as e:
    exit(1)
" 2>/dev/null; then
        echo "PostgreSQL is ready!"
        break
    fi
    
    RETRIES=$((RETRIES + 1))
    echo "Waiting for PostgreSQL... ($RETRIES/$MAX_RETRIES)"
    sleep 2
done

if [ $RETRIES -eq $MAX_RETRIES ]; then
    echo "ERROR: PostgreSQL not available after $MAX_RETRIES retries"
    exit 1
fi

# Run migrations
echo "Running database migrations..."
python manage.py migrate --noinput

echo "========================================"
echo "Starting application..."
echo "========================================"

# Execute the main command (e.g., gunicorn)
exec "$@"