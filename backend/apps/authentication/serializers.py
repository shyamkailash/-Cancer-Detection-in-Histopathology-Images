"""
Serializers for User Registration, Authentication, and Profile management.
"""

from django.contrib.auth import authenticate, get_user_model
from django.contrib.auth.password_validation import validate_password
from django.core.validators import validate_email
from django.core.exceptions import ValidationError as DjangoValidationError
from rest_framework import serializers

User = get_user_model()


class UserSerializer(serializers.ModelSerializer):
    """Safe read-only serializer for user profile representation."""

    class Meta:
        model = User
        fields = ["id", "username", "email", "is_staff", "date_joined"]
        read_only_fields = ["id", "is_staff", "date_joined"]


class RegisterSerializer(serializers.Serializer):
    """Validates and creates a new user account."""

    username = serializers.CharField(
        max_length=150,
        min_length=3,
        trim_whitespace=True,
    )
    email = serializers.EmailField(
        max_length=254,
        required=True,
        trim_whitespace=True,
    )
    password = serializers.CharField(
        write_only=True,
        min_length=8,
        trim_whitespace=False,
    )

    def validate_username(self, value):
        username = value.strip()
        if not username:
            raise serializers.ValidationError("Username cannot be blank.")
        if User.objects.filter(username__iexact=username).exists():
            raise serializers.ValidationError("A user with that username already exists.")
        return username

    def validate_email(self, value):
        email = value.strip().lower()
        try:
            validate_email(email)
        except DjangoValidationError as exc:
            raise serializers.ValidationError(str(exc))
        if User.objects.filter(email__iexact=email).exists():
            raise serializers.ValidationError("A user with that email already exists.")
        return email

    def validate_password(self, value):
        try:
            validate_password(value)
        except DjangoValidationError as exc:
            raise serializers.ValidationError(list(exc.messages))
        return value

    def create(self, validated_data):
        user = User.objects.create_user(
            username=validated_data["username"],
            email=validated_data["email"],
            password=validated_data["password"],
        )
        return user


class LoginSerializer(serializers.Serializer):
    """Validates user credentials for authentication."""

    username = serializers.CharField(trim_whitespace=True)
    password = serializers.CharField(write_only=True, trim_whitespace=False)

    def validate(self, attrs):
        login_id = attrs.get("username", "").strip()
        password = attrs.get("password", "")

        if not login_id or not password:
            raise serializers.ValidationError("Both username/email and password are required.")

        # Support login via either username or email
        user = None
        if "@" in login_id:
            try:
                user_obj = User.objects.filter(email__iexact=login_id).first()
                if user_obj:
                    user = authenticate(username=user_obj.username, password=password)
            except Exception:
                user = None
        else:
            user = authenticate(username=login_id, password=password)

        if not user:
            raise serializers.ValidationError("Invalid credentials. Please verify your username and password.")

        if not user.is_active:
            raise serializers.ValidationError("This account has been deactivated.")

        attrs["user"] = user
        return attrs
