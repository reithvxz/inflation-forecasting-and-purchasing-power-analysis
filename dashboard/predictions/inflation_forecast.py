import json
import os
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd


FORECAST_HORIZONS = OrderedDict(
    [
        ("1m", 1),
        ("3m", 3),
        ("6m", 6),
        ("12m", 12),
    ]
)

FORECAST_INTERVAL_LEVEL = 0.90
FORECAST_HISTORY_WINDOW = 24
FORECAST_TEST_WINDOW = 24

CORE_EXOG_COLUMNS = [
    "USD_IDR",
    "Harga_Minyak_USD",
    "Brent_USD",
    "BI_Rate",
    "DXY",
    "FedRate_Pct",
    "Gold_USD",
    "FAO_FPI",
]


def inflation_dataset_path(project_root):
    corrected = os.path.join(
        project_root, "datasets", "processed", "clean_inflasi_ts_corrected.csv"
    )
    legacy = os.path.join(
        project_root, "datasets", "processed", "clean_inflasi_ts.csv"
    )
    return corrected if os.path.exists(corrected) else legacy

SARIMAX_REGRESSOR_SHORTLIST = [
    "USD_IDR",
    "Brent_USD",
    "BI_Rate",
    "DXY",
    "FAO_FPI",
]


def prepare_inflation_dataframe(df):
    frame = df.copy()
    frame["Tanggal"] = pd.to_datetime(frame["Tanggal"])
    frame = frame.sort_values("Tanggal").reset_index(drop=True)
    frame = frame.ffill().bfill()
    frame["Bulan_Sin"] = np.sin(2 * np.pi * frame["Bulan"] / 12.0)
    frame["Bulan_Cos"] = np.cos(2 * np.pi * frame["Bulan"] / 12.0)
    if "Harga_Minyak_USD" in frame.columns and "USD_IDR" in frame.columns:
        frame["Oil_x_USDIDR"] = frame["Harga_Minyak_USD"] * frame["USD_IDR"]
    return frame


def forecast_artifact_path(project_root):
    return os.path.join(project_root, "models", "inflation_multihorizon_forecast.json")


def comparison_artifact_path(project_root):
    return os.path.join(project_root, "models", "inflation_multihorizon_comparison.json")


def sarimax_feature_audit_path(project_root):
    return os.path.join(project_root, "models", "sarimax_feature_audit.json")


def label_for_horizon(months):
    if months == 1:
        return "1 Bulan"
    if months == 12:
        return "12 Bulan / 1 Tahun"
    return f"{months} Bulan"


def risk_note_for_horizon(months):
    if months == 1:
        return "Horizon 1 bulan lebih cocok untuk pembacaan taktis jangka dekat."
    if months == 3:
        return "Horizon 3 bulan cocok untuk membaca arah kuartalan, tetapi tetap sensitif terhadap shock baru."
    if months == 6:
        return "Horizon 6 bulan lebih cocok untuk orientasi kebijakan dibanding angka presisi."
    return "Horizon 12 bulan dipakai untuk membaca arah makro. Rentang estimasinya harus dibaca lebih hati-hati daripada angka titiknya."


def professional_model_name(model_id):
    names = {
        "naive": "Naive Baseline",
        "arima": "ARIMA",
        "sarimax": "SARIMAX",
        "prophet": "Prophet",
        "lstm": "LSTM",
        "bilstm": "Bi-LSTM",
        "ensemble": "Ensemble",
        "garch": "GARCH",
    }
    return names.get(model_id, model_id.upper())


def load_saved_forecast_payload(project_root):
    path = forecast_artifact_path(project_root)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_saved_sarimax_feature_audit(project_root):
    path = sarimax_feature_audit_path(project_root)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_forecast_payload(
    source_df,
    horizon_results,
    comparison_summary,
    assumptions=None,
    generated_at=None,
):
    history_window = source_df.tail(FORECAST_HISTORY_WINDOW).copy()
    history_payload = {
        "labels": history_window["Tanggal"].dt.strftime("%Y-%m").tolist(),
        "actual_mom": [round(float(v), 4) for v in history_window["Inflasi_MoM"].tolist()],
        "actual_yoy": [
            round(float(v), 4) if not pd.isna(v) else None
            for v in history_window.get("Inflasi_YoY", pd.Series(dtype=float)).tolist()
        ],
        "actual_ytd": [
            round(float(v), 4) if not pd.isna(v) else None
            for v in history_window.get("Inflasi_YtD", pd.Series(dtype=float)).tolist()
        ],
        "last_date": history_window["Tanggal"].iloc[-1].strftime("%Y-%m-%d"),
        "last_actual_mom": round(float(history_window["Inflasi_MoM"].iloc[-1]), 4),
    }

    return {
        "generated_at": generated_at or datetime.utcnow().isoformat() + "Z",
        "interval_level": FORECAST_INTERVAL_LEVEL,
        "assumptions": assumptions
        or {
            "future_exog": "Untuk forecast publik, indikator eksogen masa depan diasumsikan bergerak dekat observasi terakhir (flat baseline).",
            "selection_rule": "Dua model yang ditampilkan dipilih berdasarkan MAE walk-forward terendah pada horizon yang sama.",
        },
        "history": history_payload,
        "horizons": horizon_results,
        "comparison_summary": comparison_summary,
    }

def recursive_forecast(model, steps: int, exog_future: pd.DataFrame | None = None):
    """
    Forecast `steps` bulan secara berurutan.
    - model: objek statsmodels/prophet yang sudah ter‑fit.
    - exog_future: DataFrame eksogen yang sudah diprediksi (jika ada).
    """
    preds = []
    past = model.endog.copy()               # nilai terakhir yang diketahui
    for h in range(1, steps + 1):
        # untuk statsmodels gunakan `forecast` satu langkah
        if hasattr(model, "forecast"):
            step_pred = model.forecast(steps=1, exog=exog_future.iloc[[h-1]] if exog_future is not None else None)
        else:                                 # Prophet
            df = pd.DataFrame({"ds": [model.make_future_dataframe(periods=1).ds.iloc[-1]]})
            step_pred = model.predict(df).yhat.values[0]
        preds.append(step_pred.item())
        # update series dengan prediksi tadi (hanya untuk model yang memerlukan history)
        past = np.append(past, step_pred)
        if hasattr(model, "update"):
            model = model.apply(past)         # beberapa model statsmodels punya .apply()
    return np.array(preds)
