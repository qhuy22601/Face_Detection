from django.urls import path
from . import views

urlpatterns = [
    path('view_a/', views.view_a, name='view_a'),
    path('view_b/', views.api_view_b, name='api_view_b'),
    path('view_c/', views.view_c, name='view_c'),
    path('view_d/', views.view_d, name='view_d'),
    path("check_in/", views.check_in, name='check_in'),
    path("check_out/", views.check_out, name='check_out'),
    path("upload_image/", views.upload_image, name='upload_image'),
    path("upload_images/", views.upload_images, name='upload_images'),
    path("capture/", views.capture_view, name='capture_view'),

]
