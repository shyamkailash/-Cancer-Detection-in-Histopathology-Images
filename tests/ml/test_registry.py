"""
Tests for Dataset Registry.
"""

import pytest
from ml.data.registry import DatasetRegistry


def test_default_registry_loads_pcam_and_breakhis():
    """Test that default dataset registry initializes and loads known sources."""
    registry = DatasetRegistry()
    sources = registry.list_sources()

    assert "pcam" in sources
    assert "breakhis" in sources

    pcam = registry.get_source("pcam")
    assert pcam["source_id"] == "pcam"
    assert pcam["task"] == "lymph_node_metastasis_detection"
    assert pcam["problem_type"] == "binary_classification"
    assert pcam["license"] == "CC0: Public Domain"

    breakhis = registry.get_source("breakhis")
    assert breakhis["source_id"] == "breakhis"
    assert breakhis["patient_id_available"] is True


def test_registry_unknown_source_raises_key_error():
    """Test that requesting an unknown source raises KeyError."""
    registry = DatasetRegistry()
    with pytest.raises(KeyError):
        registry.get_source("unknown_source_id_123")


def test_register_new_custom_source(tmp_path):
    """Test registering a custom source and persisting it."""
    reg_file = tmp_path / "custom_registry.json"
    registry = DatasetRegistry(registry_path=reg_file)

    meta = {
        "source_id": "custom_camelyon",
        "dataset_name": "Custom Camelyon",
        "task": "metastasis_detection",
        "problem_type": "binary_classification",
        "license": "Open",
    }
    registry.register_source("custom_camelyon", meta)
    registry.save()

    # Re-load from disk
    loaded_reg = DatasetRegistry(registry_path=reg_file)
    assert loaded_reg.has_source("custom_camelyon")
    assert loaded_reg.get_source("custom_camelyon")["dataset_name"] == "Custom Camelyon"


def test_register_source_missing_required_fields():
    """Test that registering a source without required fields raises ValueError."""
    registry = DatasetRegistry()
    with pytest.raises(ValueError):
        registry.register_source("incomplete", {"source_id": "incomplete"})
