from django.urls import path
from .views import predict, get_models

urlpatterns = [
    path('predict/', predict, name='predict'),
    path('models/', get_models, name='get-models'),
]
