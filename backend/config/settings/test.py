"""
Test settings for Cancer Detection Histopathology project.
Used by pytest and CI.
"""

from .development import *  # noqa: F401, F403

# Override for testing
DEBUG = False
SECRET_KEY = 'test-secret-key-not-for-production'

# Database - isolated in-memory SQLite for tests
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': ':memory:',
    }
}

# Disable logging during tests for cleaner output
LOGGING = {
    'version': 1,
    'disable_existing_loggers': True,
    'handlers': {
        'null': {
            'class': 'logging.NullHandler',
        },
    },
    'root': {
        'handlers': ['null'],
        'level': 'WARNING',
    },
}

# Speed up tests with faster password hasher
PASSWORD_HASHERS = [
    'django.contrib.auth.hashers.MD5PasswordHasher',
]