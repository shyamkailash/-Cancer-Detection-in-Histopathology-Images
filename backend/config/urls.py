"""
Root URL configuration for Cancer Detection Histopathology project.
"""

from django.contrib import admin
from django.urls import path, include
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response


@api_view(['GET'])
@permission_classes([AllowAny])
def api_root(request):
    """Root API endpoint returning available endpoints."""
    return Response({
        'message': 'Cancer Detection Histopathology API',
        'version': '0.1.0',
        'endpoints': {
            'health': '/api/health/',
            'predict': '/api/predict/',
            'models': '/api/models/',
            'demo': '/api/demo/',
            'mlops': '/api/mlops/status/',
            'admin': '/admin/',
        }
    })


urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', api_root, name='api-root'),
    path('api/', include('apps.core.urls')),
    path('api/', include('apps.predictions.urls')),
    path('api/mlops/', include('apps.mlops.urls')),
]