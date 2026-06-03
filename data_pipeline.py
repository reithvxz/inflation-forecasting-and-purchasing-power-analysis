"""
============================================================================
  DATA PIPELINE (ANTI-LEAKAGE) — v2
  Proyek: Prediksi Inflasi dan Dampaknya terhadap Daya Beli
============================================================================
Script ini bertugas melakukan split (Train/Val/Test) TERLEBIH DAHULU pada
data yang sudah bersih, baru kemudian melakukan Scaling, Imputasi,
dan pembuatan Fitur Lag. Hal ini secara mutlak mencegah Data Leakage.

Perubahan v2:
  - Model 1 (LSTM): Drop IHK (missing 2019–2026), tambah fitur inflasi
    komponen (Inti, HargaDiatur, Bergejolak) dan Harga Minyak.
  - Model 2 (Regresi): Drop Tahun, ganti ke Real_UMP, PDRB_HargaKonstan,
    TPT, Inflasi_Rata_Tahunan + Provinsi (one-hot). Chronological split.
============================================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "datasets", "processed")


def get_lstm_pipeline_data(seq_length=12):
    """
    Pipeline untuk Model 1 (Forecasting LSTM).
    Membaca data time-series murni, membagi chronologically murni (Train -> Val -> Test),
    lalu fit Scaler HANYA pada set Train, serta membuat sequence X, y secara aman.

    Fitur yang digunakan (8):
      Inflasi_MoM (target sekaligus fitur), BI_Rate, USD_IDR,
      Inflasi_Umum_MoM, Inflasi_Inti_MoM, Inflasi_HargaDiatur_MoM,
      Inflasi_Bergejolak_MoM, Harga_Minyak_USD

    IHK DIBUANG karena >50% data hilang (post-2019) dan imputasi 7 tahun
    terlalu banyak untuk valid.
    """
    print("\n" + "=" * 50)
    print("  LSTM Data Pipeline (Chronological Split) v2")
    print("=" * 50)

    path = os.path.join(OUT_DIR, "clean_inflasi_ts.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} tidak ditemukan. Jalankan preprocessing.py dulu.")

    df = pd.read_csv(path)
    df["Tanggal"] = pd.to_datetime(df["Tanggal"])
    df = df.sort_values("Tanggal").reset_index(drop=True)

    # 1. Forward-fill missing values untuk fitur yang valid
    #    BI_Rate & USD_IDR: ffill + bfill (data kontinu)
    df["BI_Rate"] = df["BI_Rate"].ffill().bfill()
    df["USD_IDR"] = df["USD_IDR"].ffill().bfill()

    #    Harga_Minyak_USD: ffill + bfill
    df["Harga_Minyak_USD"] = df["Harga_Minyak_USD"].ffill().bfill()

    #    Inflasi Komponen (mulai 2009, NaN sebelumnya): ffill lalu bfill
    komponen_cols = [
        "Inflasi_Umum_MoM", "Inflasi_Inti_MoM",
        "Inflasi_HargaDiatur_MoM", "Inflasi_Bergejolak_MoM"
    ]
    for col in komponen_cols:
        df[col] = df[col].ffill().bfill()

    # 2. Pilih fitur (TANPA IHK)
    feature_cols = [
        "Inflasi_MoM", "BI_Rate", "USD_IDR",
        "Inflasi_Umum_MoM", "Inflasi_Inti_MoM",
        "Inflasi_HargaDiatur_MoM", "Inflasi_Bergejolak_MoM",
        "Harga_Minyak_USD"
    ]

    # Drop baris yang masih ada NaN setelah ffill+bfill
    df_clean = df[["Tanggal"] + feature_cols].dropna().reset_index(drop=True)

    raw_data = df_clean[feature_cols].values

    # 3. CHRONOLOGICAL SPLIT (70% Train, 15% Val, 15% Test)
    n = len(raw_data)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_data = raw_data[:train_end]
    val_data = raw_data[train_end:val_end]
    test_data = raw_data[val_end:]

    # 4. FIT SCALER HANYA PADA TRAIN
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)

    # 5. TRANSFORM KETIGANYA MENGGUNAKAN SCALER TRAIN
    train_scaled = scaler.transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)

    # 6. FUNGSI UNTUK MEMBUAT SEQUENCE/WINDOWING (X dan y)
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:(i + seq_length)]
            y = data[i + seq_length, 0]  # Index 0 adalah Inflasi_MoM
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_val, y_val = create_sequences(val_scaled, seq_length)
    X_test, y_test = create_sequences(test_scaled, seq_length)

    print(f"   ✓ Fitur ({len(feature_cols)}): {feature_cols}")
    print(f"   ✓ Rentang Train: {df_clean['Tanggal'].iloc[0].date()} s.d {df_clean['Tanggal'].iloc[train_end-1].date()}")
    print(f"   ✓ Rentang Val  : {df_clean['Tanggal'].iloc[train_end].date()} s.d {df_clean['Tanggal'].iloc[val_end-1].date()}")
    print(f"   ✓ Rentang Test : {df_clean['Tanggal'].iloc[val_end].date()} s.d {df_clean['Tanggal'].iloc[-1].date()}")
    print("-" * 50)
    print(f"   ✓ X_train : {X_train.shape}, y_train: {y_train.shape}")
    print(f"   ✓ X_val   : {X_val.shape}, y_val  : {y_val.shape}")
    print(f"   ✓ X_test  : {X_test.shape}, y_test : {y_test.shape}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, df_clean


def get_regression_pipeline_data(target_col="Total_Pengeluaran"):
    """
    Pipeline untuk Model 2 (Regresi Dampak Daya Beli) — v2.

    Perubahan dari v1:
      - Drop 'Tahun' dari fitur (hanya 5 data point unik, riskan overfitting).
      - Drop 'PDRB_HargaBerlaku' (multikolinearitas VIF > 50 dengan PDRB_HargaKonstan).
      - Tambah 'PDRB_HargaKonstan' sebagai fitur.
      - Tambah 'Real_UMP' = UMP / (1 + Inflasi_Rata_Tahunan).
      - Tambah 'Provinsi' sebagai fitur kategorikal (one-hot encoding).
      - Ganti random split → chronological split (train ≤ 2023, test ≥ 2024).

    Fitur numerik: Real_UMP, TPT, PDRB_HargaKonstan, Inflasi_Rata_Tahunan
    Fitur kategorikal: Provinsi (one-hot)

    Argumen:
        target_col (str): Pilihan target ("Pengeluaran_Makanan",
                          "Pengeluaran_Bukan_Makanan", "Total_Pengeluaran")
    """
    print("\n" + "=" * 50)
    print(f"  Regression Pipeline v2 (Target: {target_col})")
    print("=" * 50)

    path = os.path.join(OUT_DIR, "clean_daya_beli.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} tidak ditemukan. Jalankan preprocessing.py dulu.")

    df = pd.read_csv(path)

    # 1. FEATURE ENGINEERING
    #    Real_UMP = UMP yang disesuaikan inflasi
    df["Real_UMP"] = df["UMP"] / (1 + df["Inflasi_Rata_Tahunan"])

    # 2. DEFINISI FITUR
    num_features = ["Real_UMP", "TPT", "PDRB_HargaKonstan", "Inflasi_Rata_Tahunan"]
    cat_features = ["Provinsi"]

    # 3. CHRONOLOGICAL SPLIT (train ≤ 2023, test ≥ 2024)
    train_mask = df["Tahun"] <= 2023
    test_mask = df["Tahun"] >= 2024

    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    X_train = df_train[num_features + cat_features]
    X_test = df_test[num_features + cat_features]
    y_train = df_train[target_col]
    y_test = df_test[target_col]

    # 4. IMPUTASI TPT (mean dari TRAIN saja untuk hindari leakage)
    mean_tpt = X_train["TPT"].mean()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["TPT"] = X_train["TPT"].fillna(mean_tpt)
    X_test["TPT"] = X_test["TPT"].fillna(mean_tpt)

    # 5. LOG TRANSFORM target dan UMP (mengurangi skewness)
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    X_train["Real_UMP"] = np.log1p(X_train["Real_UMP"])
    X_test["Real_UMP"] = np.log1p(X_test["Real_UMP"])

    print(f"   ✓ Total observasi panel: {len(df)}")
    print(f"   ✓ Fitur numerik: {num_features}")
    print(f"   ✓ Fitur kategorikal: {cat_features} (one-hot)")
    print(f"   ✓ Train (≤2023): {len(X_train)}, Test (≥2024): {len(X_test)}")
    print("-" * 50)

    return X_train, X_test, y_train_log, y_test_log, df


def get_regression_preprocessor(num_features, cat_features):
    """
    Membuat ColumnTransformer untuk regresi:
    - StandardScaler untuk fitur numerik
    - OneHotEncoder untuk Provinsi
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ]
    )


if __name__ == "__main__":
    print("\nMenyelesaikan uji coba (Dry Run) Pipeline v2...")

    try:
        lstm_data = get_lstm_pipeline_data(seq_length=12)
        print("✓ LSTM Pipeline OK.")
    except Exception as e:
        print(f"✗ Gagal LSTM Pipeline: {e}")

    try:
        reg_data = get_regression_pipeline_data(target_col="Total_Pengeluaran")
        print("✓ Regression Pipeline OK.\n")
    except Exception as e:
        print(f"✗ Gagal Regression Pipeline: {e}")
