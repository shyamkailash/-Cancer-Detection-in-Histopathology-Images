"""
Tests for the health check API endpoint.
"""

import pytest
from rest_framework.test import APIClient


@pytest.mark.django_db
class TestHealthCheckEndpoint:
    """Tests for the /api/health/ endpoint."""

    @pytest.fixture
    def api_client(self):
        return APIClient()

    def test_health_endpoint_returns_200(self, api_client):
        """Test that health endpoint returns 200 status code."""
        response = api_client.get('/api/health/')
        assert response.status_code == 200

    def test_health_endpoint_returns_correct_structure(self, api_client):
        """Test that health endpoint returns expected JSON structure."""
        response = api_client.get('/api/health/')
        data = response.data

        assert 'status' in data
        assert 'version' in data
        assert 'timestamp' in data
        assert 'components' in data
        assert 'environment' in data

    def test_health_endpoint_status_value(self, api_client):
        """Test that status field has valid value."""
        response = api_client.get('/api/health/')
        data = response.data

        assert data['status'] in ['healthy', 'degraded', 'unhealthy']

    def test_health_endpoint_version_format(self, api_client):
        """Test that version follows semantic versioning format."""
        response = api_client.get('/api/health/')
        data = response.data

        version = data['version']
        parts = version.split('.')
        assert len(parts) >= 2
        for part in parts:
            assert part.isdigit()

    def test_health_endpoint_timestamp_format(self, api_client):
        """Test that timestamp is a valid ISO format string."""
        response = api_client.get('/api/health/')
        data = response.data

        timestamp = data['timestamp']
        assert 'T' in timestamp
        assert timestamp.endswith('Z') or '+' in timestamp

    def test_health_endpoint_contains_database_component(self, api_client):
        """Test that components contains database status."""
        response = api_client.get('/api/health/')
        data = response.data

        assert 'database' in data['components']
        assert data['components']['database'] == 'connected'

    def test_health_endpoint_environment_is_development(self, api_client, settings):
        """Test that environment reflects DEBUG setting."""
        settings.DEBUG = True
        response = api_client.get('/api/health/')
        data = response.data

        assert data['environment'] == 'development'

    def test_health_endpoint_no_auth_required(self, api_client):
        """Test that health endpoint does not require authentication."""
        response = api_client.get('/api/health/')
        assert response.status_code == 200

    def test_health_endpoint_post_method_not_allowed(self, api_client):
        """Test that POST method is not allowed on health endpoint."""
        response = api_client.post('/api/health/', {})
        assert response.status_code == 405

    def test_health_endpoint_put_method_not_allowed(self, api_client):
        """Test that PUT method is not allowed on health endpoint."""
        response = api_client.put('/api/health/', {})
        assert response.status_code == 405

    def test_health_endpoint_delete_method_not_allowed(self, api_client):
        """Test that DELETE method is not allowed on health endpoint."""
        response = api_client.delete('/api/health/')
        assert response.status_code == 405


class TestAPIRootEndpoint:
    """Tests for the /api/ root endpoint."""

    @pytest.fixture
    def api_client(self):
        return APIClient()

    def test_api_root_returns_200(self, api_client):
        """Test that API root returns 200 status code."""
        response = api_client.get('/api/')
        assert response.status_code == 200

    def test_api_root_contains_endpoints(self, api_client):
        """Test that API root contains endpoints dictionary."""
        response = api_client.get('/api/')
        data = response.data

        assert 'endpoints' in data
        assert 'health' in data['endpoints']

    def test_api_root_contains_version(self, api_client):
        """Test that API root contains version."""
        response = api_client.get('/api/')
        data = response.data

        assert 'version' in data


class TestHealthCheckIntegration:
    """Integration tests for health check with database."""

    @pytest.fixture
    def api_client(self):
        return APIClient()

    def test_database_connectivity_reflected_in_health(self, db, api_client):
        """Test that database connectivity is correctly reported when DB is available."""
        response = api_client.get('/api/health/')
        data = response.data

        assert data['components']['database'] == 'connected'
        assert data['status'] == 'healthy'
        assert response.status_code == 200

    def test_database_error_handling(self, monkeypatch, api_client):
        """Test that database failure is gracefully handled with 503 and degraded status."""
        from django.db import connection
        from django.db.utils import OperationalError

        def mock_cursor():
            raise OperationalError("database connection timeout")

        monkeypatch.setattr(connection, 'cursor', mock_cursor)
        response = api_client.get('/api/health/')
        data = response.data

        assert response.status_code == 503
        assert data['status'] == 'degraded'
        assert 'database connection timeout' in data['components']['database']