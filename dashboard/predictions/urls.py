from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing_page, name='landing'),
    path('forecasting/', views.forecasting_page, name='forecasting'),
    path('daya-beli/', views.daya_beli_page, name='daya_beli'),
    path('api/simulate/', views.simulate_daya_beli, name='api_simulate'),
]
