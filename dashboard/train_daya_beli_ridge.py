import os
import pickle
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from predictions.daya_beli_model import (
    MODEL_CATEGORICAL_FEATURES,
    RAW_COMPONENT_COLUMNS,
    TARGET_COLUMN,
    TEST_START_YEAR,
    TRAIN_END_YEAR,
    YEAR_COLUMN,
    build_model_frame,
)


@dataclass
class MetricSummary:
    r2: float
    mae: float
    rmse: float


def metric_summary(y_true, y_pred) -> MetricSummary:
    return MetricSummary(
        r2=float(r2_score(y_true, y_pred)),
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
    )


def build_pipeline(numeric_features, categorical_features):
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", Ridge()),
        ]
    )


def run_walk_forward(df, numeric_features, categorical_features, alpha):
    unique_years = sorted(df[YEAR_COLUMN].unique().tolist())
    fold_results = []

    for idx in range(1, len(unique_years) - 1):
        train_end_year = unique_years[idx]
        test_year = unique_years[idx + 1]

        train_df = df[df[YEAR_COLUMN] <= train_end_year].copy()
        test_df = df[df[YEAR_COLUMN] == test_year].copy()
        if train_df.empty or test_df.empty:
            continue

        feature_columns = numeric_features + categorical_features
        X_train = train_df[feature_columns]
        y_train = train_df[TARGET_COLUMN]
        X_test = test_df[feature_columns]
        y_test = test_df[TARGET_COLUMN]

        model = build_pipeline(numeric_features, categorical_features)
        model.set_params(regressor__alpha=alpha)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = metric_summary(y_test, predictions)
        fold_results.append(
            {
                "train_end_year": int(train_end_year),
                "test_year": int(test_year),
                **asdict(metrics),
            }
        )

    if not fold_results:
        return {"folds": [], "mean": None}

    mae_values = [fold["mae"] for fold in fold_results]
    rmse_values = [fold["rmse"] for fold in fold_results]
    r2_values = [fold["r2"] for fold in fold_results]
    return {
        "folds": fold_results,
        "mean": {
            "mae": float(np.mean(mae_values)),
            "rmse": float(np.mean(rmse_values)),
            "r2": float(np.mean(r2_values)),
        },
    }


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_root, "datasets", "processed", "clean_daya_beli.csv")
    output_path = os.path.join(project_root, "models", "best_daya_beli_ridge.pkl")

    df = pd.read_csv(dataset_path)
    model_df, numeric_features, categorical_features = build_model_frame(df)

    train_df = model_df[model_df[YEAR_COLUMN] <= TRAIN_END_YEAR].copy()
    test_df = model_df[model_df[YEAR_COLUMN] >= TEST_START_YEAR].copy()
    if train_df.empty or test_df.empty:
        raise RuntimeError("Split train/test kosong. Cek rentang tahun dataset.")

    feature_columns = numeric_features + categorical_features
    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]

    pipeline = build_pipeline(numeric_features, categorical_features)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid={"regressor__alpha": [0.01, 0.1, 1.0, 5.0, 10.0, 25.0, 50.0]},
        cv=3,
        scoring="r2",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    train_metrics = metric_summary(y_train, best_model.predict(X_train))
    test_metrics = metric_summary(y_test, best_model.predict(X_test))
    walk_forward = run_walk_forward(
        model_df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        alpha=search.best_params_["regressor__alpha"],
    )

    bundle = {
        "pipeline": best_model,
        "num_features": numeric_features,
        "cat_features": categorical_features,
        "target_column": TARGET_COLUMN,
        "excluded_raw_component_columns": RAW_COMPONENT_COLUMNS,
        "data_scope": {
            "year_min": int(model_df[YEAR_COLUMN].min()),
            "year_max": int(model_df[YEAR_COLUMN].max()),
            "province_count": int(model_df["Provinsi"].nunique()),
            "row_count": int(len(model_df)),
        },
        "split_strategy": {
            "type": "chronological_by_year",
            "train_end_year": TRAIN_END_YEAR,
            "test_start_year": TEST_START_YEAR,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_years": sorted(train_df[YEAR_COLUMN].unique().tolist()),
            "test_years": sorted(test_df[YEAR_COLUMN].unique().tolist()),
        },
        "best_alpha": float(search.best_params_["regressor__alpha"]),
        "cv_best_score": float(search.best_score_),
        "train_r2": train_metrics.r2,
        "train_mae": train_metrics.mae,
        "train_rmse": train_metrics.rmse,
        "test_r2": test_metrics.r2,
        "test_mae": test_metrics.mae,
        "test_rmse": test_metrics.rmse,
        "walk_forward": walk_forward,
        "model_note": "Leakage-free ridge deployment model trained on 2021-2025 clean_daya_beli.csv.",
    }

    with open(output_path, "wb") as file_obj:
        pickle.dump(bundle, file_obj)

    print("Saved:", output_path)
    print("Rows:", bundle["data_scope"]["row_count"], "| Provinces:", bundle["data_scope"]["province_count"])
    print("Years:", bundle["data_scope"]["year_min"], "-", bundle["data_scope"]["year_max"])
    print("Features:", len(numeric_features), "numeric +", len(categorical_features), "categorical")
    print("Excluded target components:", ", ".join(RAW_COMPONENT_COLUMNS))
    print("Best alpha:", bundle["best_alpha"])
    print("Train R2:", round(bundle["train_r2"], 4), "| Test R2:", round(bundle["test_r2"], 4))
    print("Test MAE:", round(bundle["test_mae"], 2), "| Test RMSE:", round(bundle["test_rmse"], 2))
    if bundle["walk_forward"]["mean"] is not None:
        print(
            "Walk-forward mean:",
            {
                "r2": round(bundle["walk_forward"]["mean"]["r2"], 4),
                "mae": round(bundle["walk_forward"]["mean"]["mae"], 2),
                "rmse": round(bundle["walk_forward"]["mean"]["rmse"], 2),
            },
        )


if __name__ == "__main__":
    main()
