from django.urls import path
from .views import PriceEndpoint

urlpatterns = [
    path('v1/price/', PriceEndpoint.as_view(), name='price_endpoint'),
]
