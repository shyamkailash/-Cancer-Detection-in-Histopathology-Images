"""
Development settings for Cancer Detection Histopathology project.
"""

from .base import *  # noqa: F401, F403
from decouple import config

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = config('SECRET_KEY', default='django-insecure-dev-key-change-in-production-abc123xyz')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = config(
    'DEBUG',
    default=True,
    cast=lambda v: v.strip().lower() in ('true', '1', 't', 'yes', 'y') if isinstance(v, str) else bool(v)
)

ALLOWED_HOSTS = config(
    'ALLOWED_HOSTS',
    default='localhost,127.0.0.1',
    cast=lambda v: [s.strip() for s in v.split(',')]
)

# Database - PostgreSQL
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': config('POSTGRES_DB', default='cancer_detection'),
        'USER': config('POSTGRES_USER', default='cancer_user'),
        'PASSWORD': config('POSTGRES_PASSWORD', default='cancer_password'),
        'HOST': config('POSTGRES_HOST', default='localhost'),
        'PORT': config('POSTGRES_PORT', default='5432'),
    }
}

# Use verbose logging in development
LOGGING['handlers']['console']['formatter'] = 'verbose'