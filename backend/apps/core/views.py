"""
Health check API views for Cancer Detection Histopathology project.
"""

import logging
from datetime import datetime, timezone
from django.conf import settings
from django.db import connection
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

logger = logging.getLogger(__name__)


class HealthCheckView(APIView):
    """
    Health check endpoint for monitoring and load balancers.

    GET /api/health/

    Returns:
        - status: "healthy", "degraded", or "unhealthy"
        - version: Application version string
        - timestamp: Current UTC time in ISO format
        - components: Status of individual components (database, etc.)
        - environment: Current environment name (development/production)
    """
    permission_classes = []
    authentication_classes = []

    def get(self, request):
        """Perform health check and return status."""
        health_status = "healthy"
        components = {}

        # Check database connectivity
        db_status = self._check_database()
        components['database'] = db_status
        if db_status != 'connected':
            health_status = 'degraded' if health_status == 'healthy' else 'unhealthy'

        # Determine environment
        environment = 'production' if not settings.DEBUG else 'development'

        response_data = {
            'status': health_status,
            'version': getattr(settings, 'APP_VERSION', '0.1.0'),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'components': components,
            'environment': environment,
        }

        logger.info(
            "Health check performed",
            extra={
                'status': health_status,
                'database': db_status,
            }
        )

        http_status = (
            status.HTTP_200_OK
            if health_status == 'healthy'
            else status.HTTP_503_SERVICE_UNAVAILABLE
        )
        return Response(response_data, status=http_status)

    def _check_database(self) -> str:
        """Check database connectivity and return status string."""
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
            return 'connected'
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return f'error: {str(e)[:100]}'