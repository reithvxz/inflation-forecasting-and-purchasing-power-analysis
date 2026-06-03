"""
============================================================================
  SAVE BEST REGRESSION MODEL — v2
  Proyek: Prediksi Inflasi dan Dampaknya terhadap Daya Beli
============================================================================
Script ini melatih model Ridge Regression terbaik menggunakan pipeline v2
dan menyimpannya ke file .pkl untuk deployment di dashboard.

Fitur numerik: Real_UMP, TPT, PDRB_HargaKonstan, Inflasi_Rata_Tahunan
Fitur kategorikal: Provinsi (one-hot encoding)
============================================================================
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_pipeline import get_regression_pipeline_data, get_regression_preprocessor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load data dari pipeline v2
    print("Memuat data dari pipeline v2...")
    X_train, X_test, y_train, y_test, df = \
        get_regression_pipeline_data(target_col="Total_Pengeluaran")

    # 2. Definisi fitur
    num_features = ["Real_UMP", "TPT", "PDRB_HargaKonstan", "Inflasi_Rata_Tahunan"]
    cat_features = ["Provinsi"]

    # 3. Buat preprocessor
    preprocessor = get_regression_preprocessor(num_features, cat_features)

    # 4. Bangun pipeline Ridge Regression
    best_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=1.0))
    ])

    # 5. Fit model
    print("\nMelatih Ridge Regression (alpha=1.0)...")
    best_pipeline.fit(X_train, y_train)

    # 6. Evaluasi
    y_pred_train = best_pipeline.predict(X_train)
    y_pred_test = best_pipeline.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print("-" * 50)
    print(f"  Train R² : {r2_train:.4f}")
    print(f"  Test R²  : {r2_test:.4f}")
    print(f"  Test MAE : {mae_test:.4f}")
    print(f"  Test RMSE: {rmse_test:.4f}")

    # 7. Simpan pipeline lengkap (preprocessor + model)
    model_path = os.path.join(MODELS_DIR, "best_daya_beli_ridge.pkl")

    model_bundle = {
        "pipeline": best_pipeline,
        "num_features": num_features,
        "cat_features": cat_features,
        "train_r2": r2_train,
        "test_r2": r2_test,
        "test_mae": mae_test,
        "test_rmse": rmse_test,
    }

    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)

    print(f"\n  ✓ Model disimpan → {model_path}")

    # 8. Test loading
    with open(model_path, "rb") as f:
        loaded = pickle.load(f)

    loaded_pipeline = loaded["pipeline"]
    test_pred = loaded_pipeline.predict(X_test.iloc[[0]])
    print(f"  ✓ Test load model: SUCCESS!")
    print(f"  ✓ Contoh prediksi pertama: Rp {np.expm1(test_pred[0]):,.2f}")


if __name__ == "__main__":
    main()
