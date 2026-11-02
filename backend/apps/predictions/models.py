"""
Database models for Cancer Detection predictions and user history.
"""

from django.db import models
from django.conf import settings


class PredictionRecord(models.Model):
    """
    Persistent record of a cancer classification inference run.
    Associated with the authenticated user for private history tracking.
    """
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="predictions",
        null=True,
        blank=True,
        db_index=True,
    )
    model_name = models.CharField(max_length=64, db_index=True)
    model_display_name = models.CharField(max_length=128, blank=True)
    predicted_class_id = models.IntegerField()
    predicted_class_name = models.CharField(max_length=32)
    confidence = models.FloatField()
    normal_probability = models.FloatField()
    metastasis_probability = models.FloatField()
    inference_time_ms = models.FloatField(default=0.0)
    has_explanation = models.BooleanField(default=False)
    target_layer = models.CharField(max_length=64, blank=True)
    image_filename = models.CharField(max_length=255, blank=True)
    overlay_preview = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Prediction Record"
        verbose_name_plural = "Prediction Records"

    def __str__(self):
        user_str = self.user.username if self.user else "anonymous"
        return f"[{user_str}] {self.predicted_class_name} ({self.confidence:.1%}) - {self.model_name}"
