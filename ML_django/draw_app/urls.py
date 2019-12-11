from django.urls import include, path
from rest_framework import routers
from . import views

urlpatterns = [

    path('imagestore/', views.ImageStoreListView.as_view()),
    path('imagestore/create_image/', views.create_image)

]