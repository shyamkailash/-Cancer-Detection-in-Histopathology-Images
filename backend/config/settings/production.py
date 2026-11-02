"""
Production settings for Cancer Detection Histopathology project.
"""

from .base import *  # noqa: F401, F403
from decouple import config

# Security
SECRET_KEY = config('SECRET_KEY')
DEBUG = config(
    'DEBUG',
    default=False,
    cast=lambda v: v.strip().lower() in ('true', '1', 't', 'yes', 'y') if isinstance(v, str) else bool(v)
)
ALLOWED_HOSTS = config(
    'ALLOWED_HOSTS',
    default='cancer-detection-api.onrender.com,localhost,127.0.0.1,.onrender.com',
    cast=lambda v: [s.strip() for s in v.split(',') if s.strip()]
)

# Render runs behind a reverse proxy that terminates TLS
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')

# Database - PostgreSQL (supports DATABASE_URL or individual POSTGRES_* variables)
import urllib.parse
_database_url = config('DATABASE_URL', default=None)
if _database_url:
    _parsed_db = urllib.parse.urlparse(_database_url)
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.postgresql',
            'NAME': _parsed_db.path.lstrip('/'),
            'USER': _parsed_db.username,
            'PASSWORD': _parsed_db.password,
            'HOST': _parsed_db.hostname,
            'PORT': str(_parsed_db.port or 5432),
        }
    }
else:
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.postgresql',
            'NAME': config('POSTGRES_DB', default='cancer_detection'),
            'USER': config('POSTGRES_USER', default='cancer_user'),
            'PASSWORD': config('POSTGRES_PASSWORD', default='cancer_password'),
            'HOST': config('POSTGRES_HOST', default='db'),
            'PORT': config('POSTGRES_PORT', default='5432'),
        }
    }

# Security settings
CSRF_TRUSTED_ORIGINS = config(
    'CSRF_TRUSTED_ORIGINS',
    default='https://cancer-detection-in-histopathology.vercel.app,https://cancer-detection-api.onrender.com',
    cast=lambda v: [s.strip() for s in v.split(',') if s.strip()]
)
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'
SECURE_HSTS_SECONDS = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
