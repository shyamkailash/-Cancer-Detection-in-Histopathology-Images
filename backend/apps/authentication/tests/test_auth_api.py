"""
Automated tests for Authentication API endpoints.
Tests registration, validation, login, token issuance, logout, and user profile.
"""

import pytest
from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.test import APIClient
from rest_framework.authtoken.models import Token


@pytest.fixture
def api_client():
    return APIClient()


@pytest.mark.django_db
def test_register_success(api_client):
    """Test successful user registration returns HTTP 201 with token and user object."""
    payload = {
        "username": "researcher_01",
        "email": "researcher01@lab.org",
        "password": "SecurePassword123!",
    }
    response = api_client.post("/api/auth/register/", data=payload, format="json")
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert "token" in data
    assert "user" in data
    assert data["user"]["username"] == "researcher_01"
    assert data["user"]["email"] == "researcher01@lab.org"
    assert User.objects.filter(username="researcher_01").exists()
    assert Token.objects.filter(key=data["token"]).exists()


@pytest.mark.django_db
def test_register_duplicate_username(api_client):
    """Test registration fails with HTTP 400 when username already exists."""
    User.objects.create_user(username="existing_user", email="user1@lab.org", password="ValidPassword123!")
    payload = {
        "username": "existing_user",
        "email": "user2@lab.org",
        "password": "ValidPassword123!",
    }
    response = api_client.post("/api/auth/register/", data=payload, format="json")
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "username" in response.json()


@pytest.mark.django_db
def test_register_duplicate_email(api_client):
    """Test registration fails with HTTP 400 when email already exists."""
    User.objects.create_user(username="user_one", email="shared@lab.org", password="ValidPassword123!")
    payload = {
        "username": "user_two",
        "email": "shared@lab.org",
        "password": "ValidPassword123!",
    }
    response = api_client.post("/api/auth/register/", data=payload, format="json")
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "email" in response.json()


@pytest.mark.django_db
def test_register_weak_password(api_client):
    """Test registration fails when password is too short or weak."""
    payload = {
        "username": "new_user",
        "email": "new@lab.org",
        "password": "123",
    }
    response = api_client.post("/api/auth/register/", data=payload, format="json")
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "password" in response.json()


@pytest.mark.django_db
def test_login_success_with_username(api_client):
    """Test login with username succeeds and returns token."""
    user = User.objects.create_user(username="valid_user", email="valid@lab.org", password="CorrectPassword123!")
    response = api_client.post(
        "/api/auth/login/",
        data={"username": "valid_user", "password": "CorrectPassword123!"},
        format="json",
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "token" in data
    assert data["user"]["username"] == "valid_user"


@pytest.mark.django_db
def test_login_success_with_email(api_client):
    """Test login with email address succeeds and returns token."""
    User.objects.create_user(username="email_user", email="doctor@hospital.org", password="CorrectPassword123!")
    response = api_client.post(
        "/api/auth/login/",
        data={"username": "doctor@hospital.org", "password": "CorrectPassword123!"},
        format="json",
    )
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["user"]["username"] == "email_user"


@pytest.mark.django_db
def test_login_invalid_credentials(api_client):
    """Test login fails with HTTP 400 for incorrect password."""
    User.objects.create_user(username="test_user", password="CorrectPassword123!")
    response = api_client.post(
        "/api/auth/login/",
        data={"username": "test_user", "password": "WrongPassword!"},
        format="json",
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "non_field_errors" in response.json()


@pytest.mark.django_db
def test_user_profile_authenticated(api_client):
    """Test GET /api/auth/me/ returns current user profile when token is provided."""
    user = User.objects.create_user(username="auth_user", email="auth@lab.org", password="Password123!")
    token, _ = Token.objects.get_or_create(user=user)

    api_client.credentials(HTTP_AUTHORIZATION=f"Token {token.key}")
    response = api_client.get("/api/auth/me/")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["username"] == "auth_user"
    assert data["email"] == "auth@lab.org"


@pytest.mark.django_db
def test_user_profile_unauthenticated(api_client):
    """Test GET /api/auth/me/ returns HTTP 401 when unauthenticated."""
    response = api_client.get("/api/auth/me/")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.django_db
def test_logout(api_client):
    """Test POST /api/auth/logout/ deletes auth token."""
    user = User.objects.create_user(username="logout_user", password="Password123!")
    token, _ = Token.objects.get_or_create(user=user)

    api_client.credentials(HTTP_AUTHORIZATION=f"Token {token.key}")
    response = api_client.post("/api/auth/logout/")
    assert response.status_code == status.HTTP_200_OK
    assert not Token.objects.filter(user=user).exists()
