"""
Pytest configuration and shared fixtures for the project.
"""

import pytest


@pytest.fixture
def api_version():
    """Return the current API version."""
    return '0.1.0'


@pytest.fixture
def expected_health_keys():
    """Return the expected keys in health check response."""
    return ['status', 'version', 'timestamp', 'components', 'environment']