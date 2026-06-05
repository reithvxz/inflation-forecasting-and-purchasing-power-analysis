# -*- coding: utf-8 -*-
"""
Re-train ARIMA dengan order konsisten (0,0,1) dan save semua artifact.
Sekaligus hitung sMAPE sebagai metric yang lebih robust.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")
np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "datasets", "processed", "clean_inflasi_ts.csv")
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH, parse_dates=["Tanggal"])
df = df.sort_values("Tanggal").reset_index(drop=True)
series = df["Inflasi_MoM"].values
print(f"Data: {len(series)} months ({df.Tanggal.min().date()} -> {df.Tanggal.max().date()})")

# Stationarity test
adf = adfuller(series)
print(f"ADF statistic: {adf[0]:.4f}, p-value: {adf[1]:.4f} (stationary if p<0.05)")
print()

# Metrics
def smape(actual, forecast):
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    denom = np.abs(actual) + np.abs(forecast)
    diff = np.abs(actual - forecast)
    mask = denom > 0
    return float(200.0 * np.mean(diff[mask] / denom[mask])) if mask.sum() else 0.0

def mae(actual, forecast):
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(forecast))))

def rmse(actual, forecast):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(forecast)) ** 2)))

# Train/test split
n_test = 24
train = series[:-n_test]
test = series[-n_test:]
print(f"Train: {len(train)}, Test: {len(test)}")

# Fit final model on train
print("\nFitting ARIMA(0,0,1) on training data...")
model = ARIMA(train, order=(0, 0, 1))
fitted = model.fit()
print(f"AIC: {fitted.aic:.4f}, BIC: {fitted.bic:.4f}")

# Predict on test
test_pred = fitted.forecast(steps=n_test)
test_mae = mae(test, test_pred)
test_rmse = rmse(test, test_pred)
test_smape = smape(test, test_pred)
print(f"Test MAE:   {test_mae:.4f}")
print(f"Test RMSE:  {test_rmse:.4f}")
print(f"Test sMAPE: {test_smape:.2f}%  <- robust metric")

# Re-fit on ALL data for production forecast
print("\nFitting ARIMA(0,0,1) on full data...")
model_full = ARIMA(series, order=(0, 0, 1))
fitted_full = model_full.fit()
fc_3 = np.asarray(fitted_full.forecast(steps=3))
print(f"Forecast next 3 months: {[f'{v:.3f}' for v in fc_3]}")

# Save model
with open(os.path.join(MODELS_DIR, "arima_inflasi.pkl"), "wb") as f:
    pickle.dump(fitted_full, f)
print(f"[+] Saved arima_inflasi.pkl (order=(0,0,1))")

# Save forecast with consistent order
forecast_data = {
    "forecast": {
        "forecast_mean": [float(x) for x in fc_3],
        "forecast_lower": [float(x) for x in fitted_full.get_forecast(steps=3).conf_int()[:, 0]],
        "forecast_upper": [float(x) for x in fitted_full.get_forecast(steps=3).conf_int()[:, 1]]
    },
    "order": (0, 0, 1),
    "last_date": str(df["Tanggal"].iloc[-1].date()),
    "last_value": float(series[-1]),
    "metrics": {
        "aic": float(fitted_full.aic),
        "bic": float(fitted_full.bic),
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_smape": test_smape
    }
}
with open(os.path.join(MODELS_DIR, "arima_forecast.pkl"), "wb") as f:
    pickle.dump(forecast_data, f)
print(f"[+] Saved arima_forecast.pkl (consistent order)")

# Save metrics with sMAPE
metrics_data = {
    "order": (0, 0, 1),
    "aic": float(fitted_full.aic),
    "bic": float(fitted_full.bic),
    "mae": test_mae,
    "rmse": test_rmse,
    "smape": test_smape,
    "mape": None,  # deprecated, use smape instead
    "note": "MAPE 248% sebelumnya tidak reliable karena denominator kecil saat deflasi. Gunakan sMAPE sebagai gantinya."
}
with open(os.path.join(MODELS_DIR, "arima_metrics.pkl"), "wb") as f:
    pickle.dump(metrics_data, f)
print(f"[+] Saved arima_metrics.pkl (with sMAPE)")

print("\nDONE")
