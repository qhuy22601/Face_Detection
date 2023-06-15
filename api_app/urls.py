from django.urls import path
from . import views

urlpatterns = [
    path('view_a/', views.view_a, name='view_a'),
    path('view_b/', views.api_view_b, name='api_view_b'),
    path('view_c/', views.view_c, name='view_c'),
    path('view_d/', views.view_d, name='view_d'),
]
