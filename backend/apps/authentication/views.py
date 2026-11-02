"""
Authentication API Views: Registration, Login, Logout, and User Profile.
"""

import logging
from django.contrib.auth import login as django_login, logout as django_logout
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.authtoken.models import Token

from .serializers import RegisterSerializer, LoginSerializer, UserSerializer

logger = logging.getLogger(__name__)


class RegisterView(APIView):
    """
    POST /api/auth/register/
    Register a new user account and obtain an authentication token.
    """
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        serializer = RegisterSerializer(data=request.data)
        if not serializer.is_valid():
            response_data = {
                "status": "error",
                "error": "Registration validation failed.",
                "details": serializer.errors,
            }
            response_data.update(serializer.errors)
            return Response(response_data, status=status.HTTP_400_BAD_REQUEST)

        user = serializer.save()
        token, _ = Token.objects.get_or_create(user=user)

        # Also establish Django session if available
        try:
            django_login(request, user)
        except Exception:
            pass

        logger.info(f"User '{user.username}' registered successfully.")
        return Response(
            {
                "status": "success",
                "message": "User registered successfully.",
                "token": token.key,
                "user": UserSerializer(user).data,
            },
            status=status.HTTP_201_CREATED,
        )


class LoginView(APIView):
    """
    POST /api/auth/login/
    Authenticate with username/email and password to receive an auth token.
    """
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        serializer = LoginSerializer(data=request.data)
        if not serializer.is_valid():
            response_data = {
                "status": "error",
                "error": "Authentication failed.",
                "details": serializer.errors,
            }
            response_data.update(serializer.errors)
            return Response(response_data, status=status.HTTP_400_BAD_REQUEST)

        user = serializer.validated_data["user"]

        # Token rotation: delete stale token and issue fresh token
        Token.objects.filter(user=user).delete()
        token = Token.objects.create(user=user)

        try:
            django_login(request, user)
        except Exception:
            pass

        logger.info(f"User '{user.username}' logged in successfully.")
        return Response(
            {
                "status": "success",
                "message": "Authentication successful.",
                "token": token.key,
                "user": UserSerializer(user).data,
            },
            status=status.HTTP_200_OK,
        )


class LogoutView(APIView):
    """
    POST /api/auth/logout/
    Invalidate current authentication token and clear session.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        username = request.user.username if request.user else "unknown"

        # Invalidate auth token
        Token.objects.filter(user=request.user).delete()

        try:
            django_logout(request)
        except Exception:
            pass

        logger.info(f"User '{username}' logged out.")
        return Response(
            {
                "status": "success",
                "message": "Successfully logged out.",
            },
            status=status.HTTP_200_OK,
        )


class UserProfileView(APIView):
    """
    GET /api/auth/me/
    Retrieve current authenticated user profile.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        user_data = UserSerializer(request.user).data
        response_data = dict(user_data)
        response_data["status"] = "success"
        response_data["user"] = user_data
        return Response(response_data, status=status.HTTP_200_OK)
