"""
URL routing for predictions and model explainability.
"""

from django.urls import path
from .views import (
    PredictAPIView,
    ModelsListAPIView,
    DemoView,
    PredictionHistoryAPIView,
    PredictionDetailAPIView,
)

app_name = 'predictions'

urlpatterns = [
    path('predict/', PredictAPIView.as_view(), name='predict'),
    path('predict/history/', PredictionHistoryAPIView.as_view(), name='prediction-history'),
    path('predict/history/<int:pk>/', PredictionDetailAPIView.as_view(), name='prediction-detail'),
    path('models/', ModelsListAPIView.as_view(), name='models-list'),
    path('demo/', DemoView.as_view(), name='demo'),
]

