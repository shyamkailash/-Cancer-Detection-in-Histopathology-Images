"""
URL routing for predictions and model explainability.
"""

from django.urls import path
from .views import PredictAPIView, ModelsListAPIView, DemoView

app_name = 'predictions'

urlpatterns = [
    path('predict/', PredictAPIView.as_view(), name='predict'),
    path('models/', ModelsListAPIView.as_view(), name='models-list'),
    path('demo/', DemoView.as_view(), name='demo'),
]
