from django.urls import path
from .views import RegisterView, LoginView, LogoutView, UserProfileView

app_name = 'authentication'

urlpatterns = [
    path('register/', RegisterView.as_view(), name='register'),
    path('login/', LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('me/', UserProfileView.as_view(), name='me'),
    path('user/', UserProfileView.as_view(), name='user-profile'),
]
