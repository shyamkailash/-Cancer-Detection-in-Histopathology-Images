#!/usr/bin/env bash
# Render build script for Cancer Detection Histopathology backend
set -o errexit

echo "==> Python version:"
python --version

echo "==> Upgrading pip..."
python -m pip install --upgrade pip

echo "==> Installing dependencies..."
python -m pip install -r requirements/base.txt

echo "==> Verifying or downloading model checkpoints..."
python scripts/download_checkpoints.py

echo "==> Running Django database migrations..."
python backend/manage.py migrate --noinput --settings=config.settings.production

echo "==> Collecting static files..."
python backend/manage.py collectstatic --noinput --settings=config.settings.production

echo "==> Build complete!"
