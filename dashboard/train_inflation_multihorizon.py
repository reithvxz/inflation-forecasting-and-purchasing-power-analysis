import json
import logging
import os
import random
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from arch import arch_model
except ImportError:
    arch_model = None

# ----------------------------------------------------------------------
#  Imports from the project package (paths are defined in predictions/ )
# ----------------------------------------------------------------------
from predictions.inflation_forecast import (
    CORE_EXOG_COLUMNS,
    FORECAST_HISTORY_WINDOW,
    FORECAST_INTERVAL_LEVEL,
    FORECAST_TEST_WINDOW,
    FORECAST_HORIZONS,
    SARIMAX_REGRESSOR_SHORTLIST,
    comparison_artifact_path,
    forecast_artifact_path,
    inflation_dataset_path,
    sarimax_feature_audit_path,
    label_for_horizon,
    make_forecast_payload,
    prepare_inflation_dataframe,
    professional_model_name,
    risk_note_for_horizon,
)

# ----------------------------------------------------------------------
#  Global configuration & reproducibility
# ----------------------------------------------------------------------
warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

# Seed everything for reproducible results
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = inflation_dataset_path(PROJECT_ROOT)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

ARIMA_ORDER = (3, 0, 3)
SARIMAX_ORDER = (1, 0, 1)
SARIMAX_SEASONAL_ORDER = (1, 0, 0, 12)
SEQ_LENGTH = 12
LSTM_EPOCHS = 80
LSTM_PATIENCE = 8
LSTM_HIDDEN = 48
LSTM_LR = 0.001
INTERVAL_ALPHA = (1.0 - FORECAST_INTERVAL_LEVEL) / 2.0
PROPHET_REGRESSOR_CANDIDATES = [
    c for c in CORE_EXOG_COLUMNS if c in set(SARIMAX_REGRESSOR_SHORTLIST)
]


# ----------------------------------------------------------------------
#  Helper utilities
# ----------------------------------------------------------------------
def smape(y_true, y_pred):
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    denominator = np.abs(true) + np.abs(pred)
    denominator = np.where(denominator == 0, 1e-8, denominator)
    return float(np.mean(2.0 * np.abs(pred - true) / denominator) * 100.0)


def metric_block(y_true, y_pred):
    """Calculate common regression metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "smape": float(smape(y_true, y_pred)),
        "n_test": int(len(y_true)),
    }


def empirical_interval(point_forecast, residuals):
    residuals = np.asarray(residuals, dtype=float).reshape(-1)
    if residuals.size < 8:
        return float(point_forecast), float(point_forecast), "confidence band terbatas"
    lower_offset = np.quantile(residuals, INTERVAL_ALPHA)
    upper_offset = np.quantile(residuals, 1.0 - INTERVAL_ALPHA)
    return (
        float(point_forecast + lower_offset),
        float(point_forecast + upper_offset),
        None,
    )


def get_feature_columns(df):
    excluded = {"Tanggal", "Inflasi_MoM"}
    return [c for c in df.columns if c not in excluded]


def get_prophet_regressors(df):
    return [column for column in PROPHET_REGRESSOR_CANDIDATES if column in df.columns]


def metric_source_priority(metric_source):
    priority = {
        "walk_forward": 0,
        "chronological_holdout": 1,
        "not_evaluated": 2,
    }
    return priority.get(metric_source, 3)


def get_torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def recursive_forecast(model, horizon, exog_future=None):
    """
    Forecast `horizon` bulan secara iteratif (recursive).
    - `model`   : objek yang sudah ter‑fit (ARIMA, SARIMAX, Prophet, …)
    - `horizon` : berapa langkah ke depan yang di‑minta
    - `exog_future` : DataFrame eksogen yang sudah diprediksi untuk horizon
    """
    preds = []
    # untuk Prophet, model tidak memiliki .forecast; gunakan .predict pada DataFrame ds
    for step in range(1, horizon + 1):
        # ---- ARIMA / SARIMAX ----
        if hasattr(model, "forecast"):
            # ambil exog untuk step ini bila ada
            exog_step = None
            if exog_future is not None:
                exog_step = exog_future.iloc[[step - 1]]
            # forecast satu langkah
            forecast = model.forecast(steps=1, exog=exog_step) \
                if exog_step is not None else model.forecast(steps=1)
            pred = float(np.asarray(forecast).reshape(-1)[0])
            preds.append(pred)

        # ---- Prophet ----
        else:  # model is Prophet
            # Prophet memerlukan DataFrame dengan kolom ds (tanggal)
            future_df = pd.DataFrame({
                "ds": [model.make_future_dataframe(periods=step).ds.iloc[-1]]
            })
            # gabungkan eksogen jika disediakan
            if exog_future is not None:
                for col in exog_future.columns:
                    future_df[col] = exog_future[col].iloc[step - 1]
            forecast = model.predict(future_df)
            pred = float(forecast["yhat"].iloc[-1])
            preds.append(pred)

    return np.array(preds)

# ----------------------------------------------------------------------
#  Baseline & classical models
# ----------------------------------------------------------------------
def evaluate_naive(df, horizon):
    usable = df.iloc[:-horizon].copy()
    test = usable.tail(FORECAST_TEST_WINDOW).copy()
    y_true = df["Inflasi_MoM"].shift(-horizon).dropna().tail(FORECAST_TEST_WINDOW).values
    y_pred = test["Inflasi_MoM"].values
    point_forecast = float(df["Inflasi_MoM"].iloc[-1])
    return {
        "id": "naive",
        "name": professional_model_name("naive"),
        "metrics": metric_block(y_true, y_pred),
        "residuals": (y_true - y_pred).tolist(),
        "point_forecast": point_forecast,
        "metric_source": "walk_forward",
        "status": "ok",
    }


def build_future_exog(df, horizon, columns):
    future_dates = pd.date_range(
        df["Tanggal"].iloc[-1] + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )
    last_row = df.iloc[-1]
    rows = []
    for future_date in future_dates:
        row = {column: float(last_row[column]) for column in columns}
        if "Bulan_Sin" in columns or "Bulan_Cos" in columns:
            month = future_date.month
            row["Bulan_Sin"] = float(np.sin(2 * np.pi * month / 12.0))
            row["Bulan_Cos"] = float(np.cos(2 * np.pi * month / 12.0))
        if "Oil_x_USDIDR" in columns and "Harga_Minyak_USD" in row and "USD_IDR" in row:
            row["Oil_x_USDIDR"] = float(row["Harga_Minyak_USD"] * row["USD_IDR"])
        rows.append(row)
    return pd.DataFrame(rows), future_dates


def walkforward_arima(df, horizon):
    y = df["Inflasi_MoM"].reset_index(drop=True)
    start = len(df) - horizon - FORECAST_TEST_WINDOW
    predictions = []
    actuals = []

    for origin in range(start, len(df) - horizon):
        train_y = y.iloc[: origin + 1]
        actual = float(y.iloc[origin + horizon])
        try:
            model = ARIMA(train_y, order=ARIMA_ORDER)
            fitted = model.fit()
            forecast = fitted.forecast(steps=horizon)
            pred = float(np.asarray(forecast).reshape(-1)[-1])
        except Exception:
            pred = float(train_y.iloc[-1])
        predictions.append(pred)
        actuals.append(actual)

    full_model = ARIMA(y, order=ARIMA_ORDER).fit()
    future_point = float(np.asarray(full_model.forecast(steps=horizon)).reshape(-1)[-1])
    return {
        "id": "arima",
        "name": professional_model_name("arima"),
        "metrics": metric_block(actuals, predictions),
        "residuals": (np.asarray(actuals) - np.asarray(predictions)).tolist(),
        "point_forecast": future_point,
        "metric_source": "walk_forward",
        "status": "ok",
        "backtest_predictions": [float(v) for v in predictions],
        "backtest_actuals": [float(v) for v in actuals],
        "backtest_dates": [
            df["Tanggal"].iloc[origin + horizon].strftime("%Y-%m-%d")
            for origin in range(start, len(df) - horizon)
        ],
    }


def walkforward_sarimax(df, horizon, regressors=None, result_id="sarimax", result_name=None):
    y = df["Inflasi_MoM"].reset_index(drop=True)
    regressors = [
        column
        for column in (
            regressors if regressors is not None else get_prophet_regressors(df)
        )
        if column in df.columns
    ]
    exog = df[regressors].reset_index(drop=True) if regressors else None
    start = len(df) - horizon - FORECAST_TEST_WINDOW
    predictions = []
    actuals = []

    for origin in range(start, len(df) - horizon):
        train_y = y.iloc[: origin + 1]
        train_exog = exog.iloc[: origin + 1] if exog is not None else None
        future_exog = (
            exog.iloc[origin + 1 : origin + horizon + 1] if exog is not None else None
        )
        actual = float(y.iloc[origin + horizon])
        try:
            model = SARIMAX(
                train_y,
                exog=train_exog,
                order=SARIMAX_ORDER,
                seasonal_order=SARIMAX_SEASONAL_ORDER,
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False)
            forecast = fitted.forecast(steps=horizon, exog=future_exog)
            pred = float(np.asarray(forecast).reshape(-1)[-1])
        except Exception:
            pred = float(train_y.iloc[-1])
        predictions.append(pred)
        actuals.append(actual)

    future_exog, _ = (
        build_future_exog(df, horizon, regressors) if regressors else (None, None)
    )
    full_model = SARIMAX(
        y,
        exog=exog,
        order=SARIMAX_ORDER,
        seasonal_order=SARIMAX_SEASONAL_ORDER,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    future_point = float(
        np.asarray(full_model.forecast(steps=horizon, exog=future_exog)).reshape(-1)[-1]
    )
    return {
        "id": result_id,
        "name": result_name or professional_model_name("sarimax"),
        "metrics": metric_block(actuals, predictions),
        "residuals": (np.asarray(actuals) - np.asarray(predictions)).tolist(),
        "point_forecast": future_point,
        "metric_source": "walk_forward",
        "status": "ok",
        "regressors": regressors,
        "backtest_predictions": [float(v) for v in predictions],
        "backtest_actuals": [float(v) for v in actuals],
        "backtest_dates": [
            df["Tanggal"].iloc[origin + horizon].strftime("%Y-%m-%d")
            for origin in range(start, len(df) - horizon)
        ],
    }


def build_sarimax_feature_audit(df, horizon, base_result):
    regressors = list(base_result.get("regressors") or get_prophet_regressors(df))
    base_mae = float(base_result["metrics"]["mae"])
    audit_rows = []

    for feature in regressors:
        remaining = [column for column in regressors if column != feature]
        try:
            reduced_result = walkforward_sarimax(
                df,
                horizon,
                regressors=remaining,
                result_id="sarimax_ablation",
                result_name="SARIMAX drop-one",
            )
            reduced_mae = float(reduced_result["metrics"]["mae"])
            delta_mae = reduced_mae - base_mae
            audit_rows.append(
                {
                    "feature": feature,
                    "remaining_regressors": remaining,
                    "status": "ok",
                    "dropped_model_metrics": {
                        "mae": round(reduced_mae, 4),
                        "rmse": round(float(reduced_result["metrics"]["rmse"]), 4),
                        "smape": round(float(reduced_result["metrics"]["smape"]), 2),
                        "n_test": int(reduced_result["metrics"]["n_test"]),
                    },
                    "delta_mae": round(delta_mae, 4),
                    "delta_rmse": round(
                        float(
                            reduced_result["metrics"]["rmse"]
                            - base_result["metrics"]["rmse"]
                        ),
                        4,
                    ),
                    "interpretation": (
                        "penghapusan fitur memperburuk error; fitur memberi kontribusi positif"
                        if delta_mae > 0.01
                        else "penghapusan fitur hampir tidak mengubah error; kontribusi fitur cenderung marjinal"
                        if delta_mae >= -0.01
                        else "penghapusan fitur justru menurunkan error; shortlist layak ditinjau ulang"
                    ),
                }
            )
        except Exception as exc:
            audit_rows.append(
                {
                    "feature": feature,
                    "remaining_regressors": remaining,
                    "status": "skipped",
                    "reason": str(exc),
                }
            )

    audit_rows.sort(
        key=lambda item: (
            item.get("status") != "ok",
            -item.get("delta_mae", -9999),
            item.get("feature", ""),
        )
    )
    return {
        "base_regressors": regressors,
        "base_metrics": {
            "mae": round(base_mae, 4),
            "rmse": round(float(base_result["metrics"]["rmse"]), 4),
            "smape": round(float(base_result["metrics"]["smape"]), 2),
            "n_test": int(base_result["metrics"]["n_test"]),
        },
        "method": "drop-one walk-forward ablation",
        "drop_one_tests": audit_rows,
    }


def _fit_prophet(train_frame):
    regressors = get_prophet_regressors(train_frame)
    prophet_df = train_frame[["Tanggal", "Inflasi_MoM"] + regressors].rename(
        columns={"Tanggal": "ds", "Inflasi_MoM": "y"}
    )
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.15,
        seasonality_prior_scale=10.0,
    )
    for regressor in regressors:
        model.add_regressor(regressor)
    model.fit(prophet_df)
    return model, regressors


def walkforward_prophet(df, horizon):
    start = len(df) - horizon - FORECAST_TEST_WINDOW
    predictions = []
    actuals = []

    for origin in range(start, len(df) - horizon):
        train_frame = df.iloc[: origin + 1].copy()
        future_frame = df.iloc[origin + 1 : origin + horizon + 1].copy()
        actual = float(df["Inflasi_MoM"].iloc[origin + horizon])
        try:
            model, regressors = _fit_prophet(train_frame)
            predict_df = future_frame[["Tanggal"] + regressors].rename(columns={"Tanggal": "ds"})
            forecast = model.predict(predict_df)
            pred = float(forecast["yhat"].iloc[-1])
        except Exception:
            pred = float(train_frame["Inflasi_MoM"].iloc[-1])
        predictions.append(pred)
        actuals.append(actual)

    model, regressors = _fit_prophet(df.copy())
    future_exog, future_dates = build_future_exog(df, horizon, regressors)
    future_frame = future_exog.copy()
    future_frame.insert(0, "ds", future_dates)
    future_point = float(model.predict(future_frame)["yhat"].iloc[-1])
    return {
        "id": "prophet",
        "name": professional_model_name("prophet"),
        "metrics": metric_block(actuals, predictions),
        "residuals": (np.asarray(actuals) - np.asarray(predictions)).tolist(),
        "point_forecast": future_point,
        "metric_source": "walk_forward",
        "status": "ok",
        "backtest_predictions": [float(v) for v in predictions],
        "backtest_actuals": [float(v) for v in actuals],
        "backtest_dates": [
            df["Tanggal"].iloc[origin + horizon].strftime("%Y-%m-%d")
            for origin in range(start, len(df) - horizon)
        ],
    }


# ----------------------------------------------------------------------
#  Deep‑learning model (LSTM / Bi‑LSTM)
# ----------------------------------------------------------------------
class SequenceForecastModel(nn.Module):
    def __init__(self, input_size, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=LSTM_HIDDEN,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        direction_factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(0.15)
        self.fc = nn.Linear(LSTM_HIDDEN * direction_factor, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        tail = output[:, -1, :]
        return self.fc(self.dropout(tail))


def _train_sequence_model(X_train, y_train, X_val, y_val, bidirectional=False):
    device = get_torch_device()
    model = SequenceForecastModel(X_train.shape[2], bidirectional=bidirectional).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    best_state = None
    best_loss = float("inf")
    patience_left = LSTM_PATIENCE

    for _ in range(LSTM_EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        if val_loss + 1e-6 < best_loss:
            best_loss = val_loss
            patience_left = LSTM_PATIENCE
            best_state = deepcopy(model.state_dict())
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, device


def evaluate_sequence_model(df, horizon, model_id):
    """Train / evaluate LSTM or Bi‑LSTM on the inflation series.

    This implementation avoids data leakage by fitting the scalers **only on the
    training partition** before transforming the whole dataset.
    """
    bidirectional = model_id == "bilstm"
    feature_columns = get_feature_columns(df)

    # ------------------------------------------------------------------
    #   1️⃣  Create target (shifted) and drop rows without a target
    # ------------------------------------------------------------------
    usable = df.copy()
    usable["target"] = usable["Inflasi_MoM"].shift(-horizon)
    usable = usable.dropna(subset=["target"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    #   2️⃣  Partition indices (chronological hold‑out)
    # ------------------------------------------------------------------
    test_size = FORECAST_TEST_WINDOW
    val_size = 12
    total_len = len(usable)
    train_end_idx = total_len - test_size - val_size
    val_end_idx = total_len - test_size

    # ------------------------------------------------------------------
    #   3️⃣  Fit scalers **only on training data** (prevents leakage)
    # ------------------------------------------------------------------
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    train_features = usable.loc[: train_end_idx - 1, feature_columns].values
    train_targets = usable.loc[: train_end_idx - 1, "target"].values.reshape(-1, 1)

    scaler_x.fit(train_features)
    scaler_y.fit(train_targets)

    # Transform the complete dataframe using the training‑fit scalers
    x_scaled_all = scaler_x.transform(usable[feature_columns].values)
    y_scaled_all = scaler_y.transform(usable["target"].values.reshape(-1, 1))

    # ------------------------------------------------------------------
    #   4️⃣  Build sliding windows (sequences) for the LSTM
    # ------------------------------------------------------------------
    sequences = []
    targets = []
    sequence_dates = []
    dates = usable["Tanggal"].tolist()

    for idx in range(SEQ_LENGTH - 1, len(usable)):
        start = idx - SEQ_LENGTH + 1
        sequences.append(x_scaled_all[start : idx + 1])
        targets.append(y_scaled_all[idx])
        sequence_dates.append(dates[idx])

    X_seq = np.asarray(sequences, dtype=np.float32)
    y_seq = np.asarray(targets, dtype=np.float32)

    # ------------------------------------------------------------------
    #   5️⃣  Slice into train / val / test after sequence creation
    # ------------------------------------------------------------------
    seq_train_end = train_end_idx - (SEQ_LENGTH - 1)
    seq_val_end = val_end_idx - (SEQ_LENGTH - 1)

    X_train = X_seq[:seq_train_end]
    y_train = y_seq[:seq_train_end]
    X_val = X_seq[seq_train_end:seq_val_end]
    y_val = y_seq[seq_train_end:seq_val_end]
    X_test = X_seq[seq_val_end:]
    y_test = y_seq[seq_val_end:]

    if (
        X_train.size == 0
        or X_val.size == 0
        or X_test.size == 0
        or y_train.size == 0
        or y_val.size == 0
        or y_test.size == 0
    ):
        raise RuntimeError(
            f"Sequence split kosong untuk {professional_model_name(model_id)} pada horizon {horizon} bulan."
        )

    # ------------------------------------------------------------------
    #   6️⃣  Train the model
    # ------------------------------------------------------------------
    model, device = _train_sequence_model(
        X_train, y_train, X_val, y_val, bidirectional=bidirectional
    )
    model.eval()

    # ------------------------------------------------------------------
    #   7️⃣  Predict on the test set
    # ------------------------------------------------------------------
    with torch.no_grad():
        test_pred_scaled = model(
            torch.tensor(X_test, dtype=torch.float32, device=device)
        ).cpu().numpy()
    test_pred = scaler_y.inverse_transform(test_pred_scaled).reshape(-1)
    y_true = scaler_y.inverse_transform(y_test).reshape(-1)

    # ------------------------------------------------------------------
    #   8️⃣  Forecast one step ahead using the most recent sequence
    # ------------------------------------------------------------------
    last_sequence = x_scaled_all[-SEQ_LENGTH:]
    with torch.no_grad():
        future_scaled = model(
            torch.tensor(np.array([last_sequence]), dtype=torch.float32, device=device)
        ).cpu().numpy()
    point_forecast = float(
        scaler_y.inverse_transform(future_scaled).reshape(-1)[0]
    )

    # ------------------------------------------------------------------
    #   9️⃣  Return the result dictionary (same schema as other models)
    # ------------------------------------------------------------------
    return {
        "id": model_id,
        "name": professional_model_name(model_id),
        "metrics": metric_block(y_true, test_pred),
        "residuals": (y_true - test_pred).tolist(),
        "point_forecast": point_forecast,
        "metric_source": "chronological_holdout",
        "status": "ok",
        "backtest_dates": [
            dt.strftime("%Y-%m-%d")
            if isinstance(dt, pd.Timestamp)
            else str(dt)
            for dt in sequence_dates[seq_val_end:]
        ],
    }


def walkforward_garch(df, horizon):
    if arch_model is None:
        raise RuntimeError(
            "Package arch belum terpasang, sehingga kandidat GARCH belum dapat dievaluasi."
        )

    y = df["Inflasi_MoM"].reset_index(drop=True).astype(float)
    start = len(df) - horizon - FORECAST_TEST_WINDOW
    predictions = []
    actuals = []

    for origin in range(start, len(df) - horizon):
        train_y = y.iloc[: origin + 1]
        actual = float(y.iloc[origin + horizon])
        try:
            fitted = arch_model(
                train_y,
                mean="ARX",
                lags=1,
                vol="GARCH",
                p=1,
                q=1,
                dist="normal",
                rescale=False,
            ).fit(disp="off")
            forecast = fitted.forecast(horizon=horizon, reindex=False)
            pred = float(forecast.mean.iloc[-1, horizon - 1])
        except Exception:
            pred = float(train_y.iloc[-1])
        predictions.append(pred)
        actuals.append(actual)

    full_model = arch_model(
        y,
        mean="ARX",
        lags=1,
        vol="GARCH",
        p=1,
        q=1,
        dist="normal",
        rescale=False,
    ).fit(disp="off")
    future_point = float(
        full_model.forecast(horizon=horizon, reindex=False).mean.iloc[-1, horizon - 1]
    )
    return {
        "id": "garch",
        "name": professional_model_name("garch"),
        "metrics": metric_block(actuals, predictions),
        "residuals": (np.asarray(actuals) - np.asarray(predictions)).tolist(),
        "point_forecast": future_point,
        "metric_source": "walk_forward",
        "status": "ok",
        "backtest_predictions": [float(v) for v in predictions],
        "backtest_actuals": [float(v) for v in actuals],
        "backtest_dates": [
            df["Tanggal"].iloc[origin + horizon].strftime("%Y-%m-%d")
            for origin in range(start, len(df) - horizon)
        ],
    }


def maybe_garch_candidate(df, horizon):
    if arch_model is None:
        return {
            "id": "garch",
            "name": professional_model_name("garch"),
            "status": "skipped",
            "reason": "Package arch belum terpasang, sehingga kandidat GARCH belum bisa dievaluasi pada artefak ini.",
            "metric_source": "not_evaluated",
        }
    try:
        return walkforward_garch(df, horizon)
    except Exception as exc:
        return {
            "id": "garch",
            "name": professional_model_name("garch"),
            "status": "skipped",
            "reason": f"GARCH gagal dievaluasi: {exc}",
            "metric_source": "walk_forward",
        }


def build_ensemble_result(base_results):
    usable = [result for result in base_results if result.get("status") == "ok"]
    if len(usable) < 2:
        return None

    weights_raw = {
        result["id"]: 1.0 / max(result["metrics"]["mae"], 1e-6) for result in usable
    }
    weight_total = sum(weights_raw.values())
    weights = {key: value / weight_total for key, value in weights_raw.items()}

    prediction_matrix = np.vstack(
        [np.asarray(result["backtest_predictions"], dtype=float) for result in usable]
    )
    actuals = np.asarray(usable[0]["backtest_actuals"], dtype=float)
    ensemble_pred = np.zeros_like(actuals)
    for idx, result in enumerate(usable):
        ensemble_pred += weights[result["id"]] * prediction_matrix[idx]

    future_point = 0.0
    for result in usable:
        future_point += weights[result["id"]] * float(result["point_forecast"])

    return {
        "id": "ensemble",
        "name": professional_model_name("ensemble"),
        "metrics": metric_block(actuals, ensemble_pred),
        "residuals": (actuals - ensemble_pred).tolist(),
        "point_forecast": float(future_point),
        "metric_source": "walk_forward",
        "status": "ok",
        "weights": {key: round(value, 4) for key, value in weights.items()},
        "backtest_predictions": ensemble_pred.tolist(),
        "backtest_actuals": actuals.tolist(),
        "backtest_dates": usable[0]["backtest_dates"],
    }


def summarize_candidate(result):
    summary = {
        "id": result["id"],
        "name": result["name"],
        "status": result.get("status", "ok"),
        "metric_source": result.get("metric_source", "walk_forward"),
    }
    if result.get("status") == "ok":
        summary["metrics"] = {
            "mae": round(float(result["metrics"]["mae"]), 4),
            "rmse": round(float(result["metrics"]["rmse"]), 4),
            "smape": round(float(result["metrics"]["smape"]), 2),
            "n_test": int(result["metrics"]["n_test"]),
        }
        summary["point_forecast"] = round(float(result["point_forecast"]), 4)
    if "reason" in result:
        summary["reason"] = result["reason"]
    if "weights" in result:
        summary["weights"] = result["weights"]
    return summary


def forecast_for_horizon(df, horizon):
    print(f"\n=== Horizon {horizon} bulan ===")
    last_date = df["Tanggal"].iloc[-1]
    future_date = (last_date + pd.DateOffset(months=horizon)).strftime("%Y-%m")

    naive_result = evaluate_naive(df, horizon)
    arima_result = walkforward_arima(df, horizon)
    sarimax_result = walkforward_sarimax(df, horizon)
    prophet_result = walkforward_prophet(df, horizon)

    deep_candidates = []
    for model_id in ("lstm", "bilstm"):
        try:
            deep_candidates.append(evaluate_sequence_model(df, horizon, model_id))
        except Exception as exc:
            deep_candidates.append(
                {
                    "id": model_id,
                    "name": professional_model_name(model_id),
                    "status": "skipped",
                    "reason": str(exc),
                    "metric_source": "chronological_holdout",
                }
            )

    garch_candidate = maybe_garch_candidate(df, horizon)

    base_results = [naive_result, arima_result, sarimax_result, prophet_result]
    ensemble_result = build_ensemble_result(
        [arima_result, sarimax_result, prophet_result]
    )
    if ensemble_result is not None:
        base_results.append(ensemble_result)

    public_candidates = base_results + ([garch_candidate] if garch_candidate.get("status") == "ok" else [])
    ranked_public = sorted(
        [
            result
            for result in public_candidates
            if result.get("status") == "ok"
            and result.get("metric_source") == "walk_forward"
        ],
        key=lambda item: item["metrics"]["mae"],
    )[:2]

    top_models = []
    for rank, result in enumerate(ranked_public, start=1):
        ci_lower, ci_upper, interval_note = empirical_interval(
            result["point_forecast"], result["residuals"]
        )
        top_models.append(
            {
                "id": result["id"],
                "name": result["name"],
                "rank": rank,
                "point_forecast": round(float(result["point_forecast"]), 4),
                "ci_lower": round(float(ci_lower), 4),
                "ci_upper": round(float(ci_upper), 4),
                "metrics": {
                    "mae": round(float(result["metrics"]["mae"]), 4),
                    "rmse": round(float(result["metrics"]["rmse"]), 4),
                    "smape": round(float(result["metrics"]["smape"]), 2),
                    "n_test": int(result["metrics"]["n_test"]),
                    "interval_level": int(FORECAST_INTERVAL_LEVEL * 100),
                },
                "series": {
                    "forecast_label": future_date,
                    "anchor_label": df["Tanggal"].iloc[-1].strftime("%Y-%m"),
                    "anchor_actual": round(float(df["Inflasi_MoM"].iloc[-1]), 4),
                },
                "interval_note": interval_note,
            }
        )

    comparison = [summarize_candidate(result) for result in base_results]
    comparison.extend(summarize_candidate(result) for result in deep_candidates)
    comparison.append(summarize_candidate(garch_candidate))
    comparison = sorted(
        comparison,
        key=lambda item: (
            item["status"] != "ok",
            metric_source_priority(item.get("metric_source")),
            item.get("metrics", {}).get("mae", 9999.0),
            item["name"],
        ),
    )

    return {
        "label": label_for_horizon(horizon),
        "forecast_months": horizon,
        "forecast_date": future_date,
        "headline_model": top_models[0]["id"] if top_models else None,
        "headline_forecast": top_models[0]["point_forecast"] if top_models else None,
        "headline_interval": {
            "lower": top_models[0]["ci_lower"] if top_models else None,
            "upper": top_models[0]["ci_upper"] if top_models else None,
            "level": int(FORECAST_INTERVAL_LEVEL * 100),
        },
        "future_labels": [future_date],
        "top_models": top_models,
        "series": {
            "history_labels": df["Tanggal"]
            .tail(FORECAST_HISTORY_WINDOW)
            .dt.strftime("%Y-%m")
            .tolist(),
            "history_actual": [
                round(float(v), 4)
                for v in df["Inflasi_MoM"]
                .tail(FORECAST_HISTORY_WINDOW)
                .tolist()
            ],
        },
        "risk_note": risk_note_for_horizon(horizon),
        "comparison": comparison,
        "sarimax_feature_audit": build_sarimax_feature_audit(
            df, horizon, sarimax_result
        ),
    }


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(DATA_PATH)

    os.makedirs(MODELS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df = prepare_inflation_dataframe(df)

    horizon_results = {}
    comparison_summary = {}
    sarimax_feature_audit = {
        "generated_at": None,
        "methodology": {
            "selection_basis": "Shortlist awal ditentukan dari teori ekonomi, lalu diuji ulang dengan ablation drop-one out-of-sample pada SARIMAX.",
            "metric_primary": "MAE",
            "note": "Delta MAE positif berarti menghapus fitur memperburuk performa, sehingga fitur tersebut membantu model.",
        },
        "horizons": {},
    }
    for horizon_key, horizon_months in FORECAST_HORIZONS.items():
        result = forecast_for_horizon(df, horizon_months)
        horizon_results[horizon_key] = result
        comparison_summary[horizon_key] = result["comparison"]
        sarimax_feature_audit["horizons"][horizon_key] = {
            "label": result["label"],
            "forecast_months": horizon_months,
            **(result.get("sarimax_feature_audit") or {}),
        }

    payload = make_forecast_payload(df, horizon_results, comparison_summary)
    sarimax_feature_audit["generated_at"] = payload.get("generated_at")

    forecast_path = forecast_artifact_path(PROJECT_ROOT)
    with open(forecast_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    comparison_path = comparison_artifact_path(PROJECT_ROOT)
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, ensure_ascii=False, indent=2)

    feature_audit_path = sarimax_feature_audit_path(PROJECT_ROOT)
    with open(feature_audit_path, "w", encoding="utf-8") as f:
        json.dump(sarimax_feature_audit, f, ensure_ascii=False, indent=2)

    # Keep the historical filename in sync so tracked artifacts still tell the latest story.
    with open(os.path.join(MODELS_DIR, "forecast_results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved: {forecast_path}")
    print(f"Saved: {comparison_path}")
    print(f"Saved: {feature_audit_path}")
    print(f"Saved: {os.path.join(MODELS_DIR, 'forecast_results.json')}")
    return payload


if __name__ == "__main__":
    main()
