"""
Serializers for Prediction History and Detail records.
"""

from rest_framework import serializers
from .models import PredictionRecord


class PredictionRecordSerializer(serializers.ModelSerializer):
    """Serializer for user prediction history list and detail."""

    class Meta:
        model = PredictionRecord
        fields = [
            "id",
            "model_name",
            "model_display_name",
            "predicted_class_id",
            "predicted_class_name",
            "confidence",
            "normal_probability",
            "metastasis_probability",
            "inference_time_ms",
            "has_explanation",
            "target_layer",
            "image_filename",
            "overlay_preview",
            "created_at",
        ]
        read_only_fields = fields
