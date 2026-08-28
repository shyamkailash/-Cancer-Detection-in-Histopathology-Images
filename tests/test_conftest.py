"""
Tests for root conftest fixtures.
"""


def test_api_version_fixture(api_version):
    """Test that api_version fixture returns a valid version string."""
    assert api_version == '0.1.0'


def test_expected_health_keys_fixture(expected_health_keys):
    """Test that expected_health_keys fixture returns the standard response keys."""
    assert 'status' in expected_health_keys
    assert 'version' in expected_health_keys
    assert 'timestamp' in expected_health_keys
    assert 'components' in expected_health_keys
    assert 'environment' in expected_health_keys
