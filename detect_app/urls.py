from . import views
from django.urls import path

urlpatterns = [
    path("",views.detection_view),
    ]

