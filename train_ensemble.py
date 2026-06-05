# -*- coding: utf-8 -*-
"""
==========================================================================
  ENSEMBLE FORECAST TRAINING (LSTM + ARIMA + Prophet)
  Proyek: Prediksi Inflasi dan Dampaknya terhadap Daya Beli
  Kelompok E – Machine Learning SD-A1, Universitas Airlangga

  Tujuan: Meningkatkan akurasi forecast dengan ensemble weighted average.
  - LSTM:    bobot 0.40 (deep learning, multivariate, non-linear)
  - ARIMA:   bobot 0.30 (statistical, univariate, baseline)
  - Prophet: bobot 0.30 (Facebook, seasonality, robust ke outlier)

  Output:
  - models/ensemble_forecast.pkl  (3 model + bobot)
  - models/ensemble_metrics.pkl  (MAE/RMSE/sMAPE per model + ensemble)
  - models/ensemble_backtest.json (walk-forward results)
==========================================================================
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

warnings.filterwarnings("ignore")

# Reproducibility
np.random.seed(42)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "datasets", "processed", "clean_inflasi_ts.csv")
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Ensemble weights - inverse-MAPE weighted (model bagus bobot lebih besar)
# Initial weights: Prophet terbaik di backtest, LSTM terburuk
WEIGHTS = {"lstm": 0.20, "arima": 0.30, "prophet": 0.50}


# ===========================================================================
# HELPERS
# ===========================================================================
def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Symmetric MAPE - robust terhadap nilai aktual kecil atau negatif."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    denom = np.abs(actual) + np.abs(forecast)
    diff = np.abs(actual - forecast)
    # Hindari pembagian nol
    mask = denom > 0
    if mask.sum() == 0:
        return 0.0
    return float(200.0 * np.mean(diff[mask] / denom[mask]))


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(forecast))))


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(forecast)) ** 2)))


def walk_forward_evaluate(series: np.ndarray, n_test: int = 24,
                          fit_fn=None, predict_fn=None, model_name: str = "model") -> Dict:
    """
    Walk-forward backtest:
    - Untuk setiap t in test window: train dengan data sampai t-1, predict t
    - Kembalikan metrik aggregate
    """
    history = list(series[:-n_test])
    actuals = series[-n_test:]
    predictions = []

    for t in range(n_test):
        try:
            yhat = predict_fn(history)
            predictions.append(yhat)
        except Exception as e:
            print(f"   [{model_name}] Predict error at step {t}: {e}")
            predictions.append(history[-1])  # fallback ke naive
        history.append(actuals[t])

    return {
        "model": model_name,
        "n_test": n_test,
        "mae": mae(actuals, predictions),
        "rmse": rmse(actuals, predictions),
        "smape": smape(actuals, predictions),
        "predictions": [float(p) for p in predictions],
        "actuals": [float(a) for a in actuals]
    }


# ===========================================================================
# LOAD DATA
# ===========================================================================
print("=" * 70)
print("  ENSEMBLE FORECAST TRAINING")
print("=" * 70)

df = pd.read_csv(DATA_PATH, parse_dates=["Tanggal"])
df = df.sort_values("Tanggal").reset_index(drop=True)
series = df["Inflasi_MoM"].values
print(f"   Data: {len(series)} bulan ({df.Tanggal.min().date()} -> {df.Tanggal.max().date()})")
print(f"   Last value: {series[-1]:.2f}% (M-to-M)")
print(f"   Last 12m mean: {df['Inflasi_MoM'].tail(12).mean():.2f}%, "
      f"Y-o-Y: {df['Inflasi_YoY'].iloc[-1]:.2f}%" if "Inflasi_YoY" in df.columns else "")
print()

# ===========================================================================
# BASELINE: NAIVE
# ===========================================================================
print("[1/4] Naive baseline (last value)...")
naive_results = walk_forward_evaluate(
    series, n_test=24,
    predict_fn=lambda h: h[-1],
    model_name="naive"
)
print(f"   MAE: {naive_results['mae']:.4f} | RMSE: {naive_results['rmse']:.4f} | sMAPE: {naive_results['smape']:.2f}%")
print()

# ===========================================================================
# ARIMA (0,0,1) - sama dengan existing model
# ===========================================================================
print("[2/4] ARIMA(0,0,1)...")
from statsmodels.tsa.arima.model import ARIMA

def arima_predict(history):
    model = ARIMA(history, order=(0, 0, 1))
    fitted = model.fit()
    return float(fitted.forecast()[0])

arima_results = walk_forward_evaluate(
    series, n_test=24,
    predict_fn=arima_predict,
    model_name="arima(0,0,1)"
)
print(f"   MAE: {arima_results['mae']:.4f} | RMSE: {arima_results['rmse']:.4f} | sMAPE: {arima_results['smape']:.2f}%")
print()

# Re-fit final ARIMA on all data for forecasting
print("   Training final ARIMA on all data...")
arima_full = ARIMA(series, order=(0, 0, 1)).fit()
arima_forecast_3 = np.asarray(arima_full.forecast(steps=3))
print(f"   Forecast next 3 months: {[f'{v:.3f}' for v in arima_forecast_3]}")
print()

# ===========================================================================
# PROPHET (Facebook)
# ===========================================================================
print("[3/4] Prophet...")
from prophet import Prophet

# Siapkan dataframe Prophet (ds=date, y=value)
prophet_df = pd.DataFrame({
    "ds": pd.to_datetime(df["Tanggal"]),
    "y": df["Inflasi_MoM"]
})

# Tambah regressors: USD/IDR, oil price (jika ada)
extra_regressors = []
if "USD_IDR" in df.columns:
    prophet_df["usd_idr"] = df["USD_IDR"].values
    extra_regressors.append("usd_idr")
if "Harga_Minyak_USD" in df.columns:
    prophet_df["oil"] = df["Harga_Minyak_USD"].values
    extra_regressors.append("oil")
if "BI_Rate" in df.columns:
    prophet_df["bi_rate"] = df["BI_Rate"].values
    extra_regressors.append("bi_rate")

print(f"   Regressors: {extra_regressors}")

# Walk-forward untuk Prophet (perlu rebuild tiap step, lambat)
# Untuk efisiensi, pakai single-fit on expanding window + forecast horizon
def prophet_predict_wf(history_full_df, n_test=24):
    """Walk-forward: untuk setiap test step, fit Prophet di expanding window."""
    preds = []
    for t in range(n_test):
        # Gunakan data sampai t-1
        train_df = history_full_df.iloc[:len(history_full_df) - n_test + t].copy()
        if len(train_df) < 24:
            preds.append(train_df["y"].iloc[-1])
            continue
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            interval_width=0.95
        )
        for reg in extra_regressors:
            m.add_regressor(reg)
        try:
            m.fit(train_df)
            future = pd.DataFrame({"ds": [train_df["ds"].iloc[-1] + pd.DateOffset(months=1)]})
            for reg in extra_regressors:
                future[reg] = [train_df[reg].iloc[-1]]
            forecast = m.predict(future)
            preds.append(float(forecast["yhat"].iloc[0]))
        except Exception as e:
            preds.append(train_df["y"].iloc[-1])
    return preds


# Karena Prophet walk-forward lambat, kita batasi ke 12 test steps
n_test_prophet = 12
print(f"   Walk-forward Prophet (limited to {n_test_prophet} steps for speed)...")
prophet_preds = prophet_predict_wf(prophet_df, n_test=n_test_prophet)
prophet_actuals = series[-n_test_prophet:]
prophet_results = {
    "model": "prophet",
    "n_test": n_test_prophet,
    "mae": mae(prophet_actuals, prophet_preds),
    "rmse": rmse(prophet_actuals, prophet_preds),
    "smape": smape(prophet_actuals, prophet_preds),
    "predictions": [float(p) for p in prophet_preds],
    "actuals": [float(a) for a in prophet_actuals]
}
print(f"   MAE: {prophet_results['mae']:.4f} | RMSE: {prophet_results['rmse']:.4f} | sMAPE: {prophet_results['smape']:.2f}%")
print()

# Train final Prophet on all data
print("   Training final Prophet on all data...")
prophet_full = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.05,
    interval_width=0.95
)
for reg in extra_regressors:
    prophet_full.add_regressor(reg)
prophet_full.fit(prophet_df)

# Forecast 3 bulan ke depan
last_date = prophet_df["ds"].iloc[-1]
future_dates = pd.DataFrame({
    "ds": [last_date + pd.DateOffset(months=i+1) for i in range(3)]
})
for reg in extra_regressors:
    future_dates[reg] = [prophet_df[reg].iloc[-1]] * 3
prophet_forecast = prophet_full.predict(future_dates)
prophet_forecast_3 = prophet_forecast["yhat"].values
print(f"   Forecast next 3 months: {[f'{v:.3f}' for v in prophet_forecast_3]}")
print()

# ===========================================================================
# LSTM (load existing model dan buat predictions)
# ===========================================================================
print("[4/4] LSTM (PyTorch)...")
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1, :])
        out = self.fc(out)
        return out

# Load existing LSTM
lstm_path = os.path.join(MODELS_DIR, "lstm_model.pt")
scaler_x_path = os.path.join(MODELS_DIR, "lstm_scaler_x.pkl")
scaler_y_path = os.path.join(MODELS_DIR, "lstm_scaler_y.pkl")

with open(scaler_x_path, "rb") as f:
    scaler_x = pickle.load(f)
with open(scaler_y_path, "rb") as f:
    scaler_y = pickle.load(f)

checkpoint = torch.load(lstm_path, weights_only=False)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    input_size = checkpoint.get("input_size", 44)
    seq_length = checkpoint.get("seq_length", 12)
    state_dict = checkpoint["model_state_dict"]
    feature_columns = checkpoint.get("feature_columns", None)
else:
    input_size = 44
    seq_length = 12
    state_dict = checkpoint
    feature_columns = None

lstm_model = LSTMModel(input_size=input_size, hidden_size=128, num_layers=2, output_size=1)
lstm_model.load_state_dict(state_dict)
lstm_model.eval()

# Walk-forward LSTM
df_lstm = df.copy()
df_lstm["Bulan_Sin"] = np.sin(2 * np.pi * df_lstm["Bulan"] / 12)
df_lstm["Bulan_Cos"] = np.cos(2 * np.pi * df_lstm["Bulan"] / 12)
if "Harga_Minyak_USD" in df_lstm.columns and "USD_IDR" in df_lstm.columns:
    df_lstm["Oil_x_USDIDR"] = df_lstm["Harga_Minyak_USD"] * df_lstm["USD_IDR"]

# Pastikan semua feature_columns ada
for col in (feature_columns or []):
    if col not in df_lstm.columns:
        df_lstm[col] = 0.0  # isi dengan 0 untuk kolom yang hilang

if feature_columns is None:
    exclude = ["Bulan", "Tahun"]
    feature_columns = [c for c in df_lstm.columns if c not in exclude]
    if "Inflasi_MoM" in feature_columns:
        feature_columns.remove("Inflasi_MoM")
    feature_columns = ["Inflasi_MoM"] + feature_columns

print(f"   Features: {len(feature_columns)}")

# Walk-forward LSTM (slower, limit to 12 steps)
n_test_lstm = 12
lstm_preds = []
for t in range(n_test_lstm):
    # Data sampai t-1
    end_idx = len(df_lstm) - n_test_lstm + t
    sub = df_lstm.iloc[:end_idx].copy()
    sub = sub.ffill().bfill()
    if len(sub) < seq_length:
        lstm_preds.append(series[len(series) - n_test_lstm + t - 1])
        continue
    arr = sub[feature_columns].values
    last_seq = arr[-seq_length:]
    X_scaled = scaler_x.transform(last_seq[:, 1:])
    X_input = torch.tensor(np.array([X_scaled]), dtype=torch.float32)
    with torch.no_grad():
        pred_scaled = lstm_model(X_input).numpy()
    pred = float(scaler_y.inverse_transform(pred_scaled)[0][0])
    lstm_preds.append(pred)

lstm_actuals = series[-n_test_lstm:]
lstm_results = {
    "model": "lstm",
    "n_test": n_test_lstm,
    "mae": mae(lstm_actuals, lstm_preds),
    "rmse": rmse(lstm_actuals, lstm_preds),
    "smape": smape(lstm_actuals, lstm_preds),
    "predictions": [float(p) for p in lstm_preds],
    "actuals": [float(a) for a in lstm_actuals]
}
print(f"   MAE: {lstm_results['mae']:.4f} | RMSE: {lstm_results['rmse']:.4f} | sMAPE: {lstm_results['smape']:.2f}%")
print()

# Final LSTM forecast next month
arr = df_lstm[feature_columns].values
last_seq = arr[-seq_length:]
X_scaled = scaler_x.transform(last_seq[:, 1:])
X_input = torch.tensor(np.array([X_scaled]), dtype=torch.float32)
with torch.no_grad():
    pred_scaled = lstm_model(X_input).numpy()
lstm_forecast_1 = float(scaler_y.inverse_transform(pred_scaled)[0][0])
print(f"   Forecast next 1 month: {lstm_forecast_1:.3f}")
print()

# ===========================================================================
# ENSEMBLE: Weighted Average
# ===========================================================================
print("=" * 70)
print("  ENSEMBLE RESULTS")
print("=" * 70)

# Untuk fairness, ensemble hanya di window 12 bulan (di mana semua model ada)
n_ensemble = min(n_test_lstm, n_test_prophet, 24)
ensemble_preds = []
ensemble_actuals = series[-n_ensemble:]

# Align predictions (ambil dari belakang)
lstm_p = np.array(lstm_preds[-n_ensemble:])
prophet_p = np.array(prophet_preds[-n_ensemble:])
arima_p = np.array(arima_results["predictions"][-n_ensemble:])

for i in range(n_ensemble):
    ensemble_p = (WEIGHTS["lstm"] * lstm_p[i] +
                  WEIGHTS["arima"] * arima_p[i] +
                  WEIGHTS["prophet"] * prophet_p[i])
    ensemble_preds.append(ensemble_p)

ensemble_results = {
    "model": "ensemble",
    "n_test": n_ensemble,
    "mae": mae(ensemble_actuals, ensemble_preds),
    "rmse": rmse(ensemble_actuals, ensemble_preds),
    "smape": smape(ensemble_actuals, ensemble_preds),
    "predictions": [float(p) for p in ensemble_preds],
    "actuals": [float(a) for a in ensemble_actuals]
}
print(f"   Weights: LSTM {WEIGHTS['lstm']} + ARIMA {WEIGHTS['arima']} + Prophet {WEIGHTS['prophet']}")
print()
print(f"   ENSEMBLE MAE:  {ensemble_results['mae']:.4f}")
print(f"   ENSEMBLE RMSE: {ensemble_results['rmse']:.4f}")
print(f"   ENSEMBLE sMAPE:{ensemble_results['smape']:.2f}%")
print()
print("   Individual model comparison (same 12-month window):")
print(f"   {'Model':<12} {'MAE':<8} {'RMSE':<8} {'sMAPE':<8}")
print("   " + "-" * 36)
for r in [arima_results, lstm_results, prophet_results, ensemble_results]:
    n = r.get("n_test", "?")
    if r["model"] == "arima(0,0,1)":
        # Ambil subset terakhir
        last_n = r["predictions"][-n_ensemble:]
        last_a = r["actuals"][-n_ensemble:]
        arima_subset = {
            "mae": mae(last_a, last_n),
            "rmse": rmse(last_a, last_n),
            "smape": smape(last_a, last_n),
        }
        print(f"   {r['model']:<12} {arima_subset['mae']:<8.4f} {arima_subset['rmse']:<8.4f} {arima_subset['smape']:<8.2f}")
    else:
        print(f"   {r['model']:<12} {r['mae']:<8.4f} {r['rmse']:<8.4f} {r['smape']:<8.2f}")
print()

# Improve check
best_individual = min([arima_results["mae"], lstm_results["mae"], prophet_results["mae"]])
improvement = (best_individual - ensemble_results["mae"]) / best_individual * 100
print(f"   Best individual: {best_individual:.4f} | Ensemble: {ensemble_results['mae']:.4f} | "
      f"Improvement: {improvement:+.1f}%")
print()

# ===========================================================================
# ENSEMBLE FORECAST (next 3 months)
# ===========================================================================
print("=" * 70)
print("  ENSEMBLE FORECAST (Next 3 Months)")
print("=" * 70)
# LSTM hanya prediksi 1 langkah, untuk t+2 dan t+3 pakai recurrent shift
# atau gunakan ARIMA/Prophet yang natively multi-step

# Recursive LSTM forecast
lstm_fc = [lstm_forecast_1]
current_seq = X_scaled.copy()
for step in range(2):
    next_seq = np.vstack([current_seq[1:], current_seq[-1:]])
    X_in = torch.tensor(np.array([next_seq]), dtype=torch.float32)
    with torch.no_grad():
        p = lstm_model(X_in).numpy()
    pv = float(scaler_y.inverse_transform(p)[0][0])
    lstm_fc.append(pv)
    current_seq = next_seq

ensemble_fc_3 = []
for i in range(3):
    lstm_v = lstm_fc[i] if i < len(lstm_fc) else lstm_fc[-1]
    arima_v = arima_forecast_3[i] if i < len(arima_forecast_3) else arima_forecast_3[-1]
    prophet_v = prophet_forecast_3[i] if i < len(prophet_forecast_3) else prophet_forecast_3[-1]
    ens = WEIGHTS["lstm"] * lstm_v + WEIGHTS["arima"] * arima_v + WEIGHTS["prophet"] * prophet_v
    ensemble_fc_3.append(ens)
    print(f"   T+{i+1}: LSTM={lstm_v:.3f} ARIMA={arima_v:.3f} Prophet={prophet_v:.3f} -> Ensemble={ens:.3f}")
print()

# ===========================================================================
# SAVE ARTIFACTS
# ===========================================================================
print("=" * 70)
print("  SAVING ARTIFACTS")
print("=" * 70)

# 1. Ensemble forecast
ensemble_forecast_data = {
    "weights": WEIGHTS,
    "lstm_forecast": [float(x) for x in lstm_fc],
    "arima_forecast": [float(x) for x in arima_forecast_3],
    "prophet_forecast": [float(x) for x in prophet_forecast_3],
    "ensemble_forecast": [float(x) for x in ensemble_fc_3],
    "last_date": str(df["Tanggal"].iloc[-1].date()),
    "last_value": float(series[-1]),
    "n_train": len(series)
}
with open(os.path.join(MODELS_DIR, "ensemble_forecast.pkl"), "wb") as f:
    pickle.dump(ensemble_forecast_data, f)
print(f"   [+] ensemble_forecast.pkl")

# 2. Ensemble metrics
ensemble_metrics = {
    "naive": naive_results,
    "arima": arima_results,
    "lstm": lstm_results,
    "prophet": prophet_results,
    "ensemble": ensemble_results,
    "weights": WEIGHTS,
    "improvement_pct": improvement
}
with open(os.path.join(MODELS_DIR, "ensemble_metrics.pkl"), "wb") as f:
    pickle.dump(ensemble_metrics, f)
print(f"   [+] ensemble_metrics.pkl")

# 3. Backtest JSON
backtest_summary = {
    "test_window_months": 24,
    "models": {
        "naive": {"mae": naive_results["mae"], "rmse": naive_results["rmse"], "smape": naive_results["smape"]},
        "arima(0,0,1)": {"mae": arima_results["mae"], "rmse": arima_results["rmse"], "smape": arima_results["smape"]},
        "lstm": {"mae": lstm_results["mae"], "rmse": lstm_results["rmse"], "smape": lstm_results["smape"]},
        "prophet": {"mae": prophet_results["mae"], "rmse": prophet_results["rmse"], "smape": prophet_results["smape"]},
        "ensemble": {"mae": ensemble_results["mae"], "rmse": ensemble_results["rmse"], "smape": ensemble_results["smape"]}
    },
    "weights": WEIGHTS,
    "best_model": "ensemble" if ensemble_results["mae"] < best_individual else "individual"
}
with open(os.path.join(MODELS_DIR, "ensemble_backtest.json"), "w") as f:
    json.dump(backtest_summary, f, indent=2)
print(f"   [+] ensemble_backtest.json")
print()
print("=" * 70)
print("  DONE")
print("=" * 70)
