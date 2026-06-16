from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_page, name='home'),
    path('dashboard/', views.landing_page, name='landing'),
    path('panduan/', views.guide_page, name='guide'),
    path('forecasting/', views.forecasting_page, name='forecasting'),
    path('daya-beli/', views.daya_beli_page, name='daya_beli'),
    path('datasets/', views.datasets_page, name='datasets'),
    path('compare/', views.compare_page, name='compare'),
    path('scenarios/', views.scenarios_page, name='scenarios'),
    path('api/simulate/', views.simulate_daya_beli, name='api_simulate'),
    path('api/dataset-sample/', views.api_dataset_sample, name='api_dataset_sample'),
    path('api/provinces/', views.api_province_list, name='api_province_list'),
    path('api/province-data/', views.api_province_data, name='api_province_data'),
    path('api/metrics-latest/', views.api_all_metrics_latest, name='api_metrics_latest'),
    path('api/scenario-analysis/', views.api_scenario_analysis, name='api_scenario_analysis'),
    path('api/commodity-prices/', views.api_commodity_prices, name='api_commodity_prices'),
    path('api/usd-idr/', views.api_usd_idr_latest, name='api_usd_idr_latest'),
    path('api/arima-forecast/', views.api_arima_forecast, name='api_arima_forecast'),
    path('api/ensemble-forecast/', views.api_ensemble_forecast, name='api_ensemble_forecast'),
    path('api/inflation-forecast/', views.api_inflation_forecast, name='api_inflation_forecast'),
    path('api/inflasi-summary/', views.api_inflasi_summary, name='api_inflasi_summary'),
    path('map/', views.map_page, name='map'),
]
