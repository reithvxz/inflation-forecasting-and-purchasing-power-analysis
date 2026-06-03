# Prediksi Inflasi dan Dampaknya terhadap Daya Beli

> **Kelompok E – Machine Learning SD-A1, Universitas Airlangga**

---

## Deskripsi Proyek

Proyek ini membangun sistem prediksi inflasi dan analisis dampaknya terhadap daya beli masyarakat Indonesia. Terdapat dua model utama:

1. **Forecasting (LSTM)** — Memprediksi nilai inflasi bulanan ke depan berdasarkan data historis 8 fitur.
2. **Regresi (Ridge)** — Mengukur pengaruh inflasi terhadap daya beli masyarakat (pengeluaran per kapita) per provinsi.

Output disajikan melalui **Dashboard Web (Django)** yang menampilkan grafik prediksi dan fitur simulasi interaktif.

---

## Struktur Proyek

```
Project-Machine-Learning/
├── datasets/
│   ├── ... (13 dataset mentah)
│   └── processed/
│       ├── clean_inflasi_ts.csv   ← time-series bulanan untuk LSTM
│       └── clean_daya_beli.csv    ← panel provinsi untuk Regresi
├── dashboard/                     ← Django project
│   ├── dashboard/
│   │   ├── settings.py
│   │   └── urls.py
│   └── predictions/               ← Django app
│       ├── urls.py
│       ├── views.py
│       └── templates/predictions/
│           ├── base.html
│           ├── home.html
│           ├── forecast.html
│           └── daya_beli.html
├── models/
│   ├── best_lstm_inflasi.pt       ← Model LSTM (PyTorch)
│   ├── lstm_scaler.pkl            ← Scaler untuk LSTM
│   └── best_daya_beli_ridge.pkl   ← Model Ridge Regression
├── explore_datasets.py
├── preprocessing.py               ← Pipeline join dataset
├── data_pipeline.py               ← ANTI-LEAKAGE PIPELINE v2
├── save_best_lstm.py              ← Training & simpan model LSTM
├── save_best_model.py             ← Training & simpan model Ridge
├── notebooks/
│   ├── forecasting_inflasi_models.ipynb
│   └── analisis_daya_beli_regresi.ipynb
├── requirements.txt
└── README.md
```

---

## Dataset

| # | Dataset | Sumber | Periode | Model | Peran |
|---|---------|--------|---------|-------|-------|
| 1 | Indeks Harga Konsumen (IHK) | BPS | 2005–2019 | — | Dihilangkan (missing >50%) |
| 2 | Inflasi Bulanan (M-to-M) | BPS | 2005–2026 | Model 1 | **Target Y** + Fitur |
| 3 | Inflasi Tahun Kalender (Y-to-D) | BPS | Historis | — | Referensi |
| 4 | BI Rate / Data Inflasi | Bank Indonesia | 2005–2026 | Model 1 | Fitur X |
| 5 | Upah Minimum Provinsi (UMP) | BPS Jateng | 2021–2025 | Model 2 | Fitur X |
| 6 | Rata-rata Pengeluaran per Kapita | BPS | 2017–2025 | Model 2 | **Target Y** |
| 7 | Kurs USD/IDR Historis | Investing.com | 2005–2025 | Model 1 | Fitur X |
| 8 | Tingkat Pengangguran Terbuka (Semester) | Open Data Jabar | 2020–2025 | Model 2 | Fitur X |
| 9 | TPT & TPAK Menurut Provinsi | BPS | 2017–2025 | Model 2 | Fitur X |
| 10 | PDRB Per Kapita (Ribu Rupiah) | BPS | 2010–2025 | Model 2 | Fitur X |
| 11 | Persentase Penduduk Miskin per Provinsi | Open Data Jabar | 2010–2024 | Model 2 | Fitur X |
| 12 | Inflasi Umum, Inti, Harga Diatur, Bergejolak | BPS | 2009–2026 | Model 1 | Fitur X |
| 13 | Harga Bulanan Minyak Mentah (USD/Barel) | IndexMundi | 2001–2026 | Model 1 | Fitur X |

---

## Model Machine Learning

### Model 1 – Forecasting Inflasi (LSTM)

| Aspek | Detail |
|-------|-------|
| **Arsitektur** | LSTM, 2 layer, 64 hidden units, dropout 0.2 |
| **Window** | 12 bulan (sequence length) |
| **Fitur (8)** | Inflasi_MoM, BI_Rate, USD_IDR, Inflasi_Umum_MoM, Inflasi_Inti_MoM, Inflasi_HargaDiatur_MoM, Inflasi_Bergejolak_MoM, Harga_Minyak_USD |
| **Split** | Chronological: 70% Train, 15% Val, 15% Test |
| **Scaler** | MinMaxScaler, fit hanya pada Train |
| **Output** | Prediksi inflasi bulan berikutnya (iteratif untuk N bulan) |
| **Metrik** | MAE, RMSE |

**Perubahan v2:**
- IHK di-drop karena missing 2019–2026 (terlalu banyak untuk imputasi valid).
- Ditambahkan 4 fitur baru: Inflasi Komponen (Inti, HargaDiatur, Bergejolak) dan Harga Minyak.

### Model 2 – Dampak Inflasi terhadap Daya Beli (Ridge Regression)

| Aspek | Detail |
|-------|--------|
| **Model** | Ridge Regression (alpha=1.0, L2 regularization) |
| **Fitur numerik** | Real_UMP, TPT, PDRB_HargaKonstan, Inflasi_Rata_Tahunan |
| **Fitur kategorikal** | Provinsi (one-hot encoding, 38 provinsi) |
| **Split** | Chronological: Train (2021–2023), Test (2024–2025) |
| **Preprocessing** | StandardScaler (numerik), OneHotEncoder (Provinsi) |
| **Target** | log(1 + Total_Pengeluaran) |
| **Metrik** | R², MAE, RMSE |

**Perubahan v2:**
- `Tahun` di-drop dari fitur (hanya 5 data point unik, riskan overfitting).
- `PDRB_HargaBerlaku` di-drop (multikolinearitas VIF > 50 dengan PDRB_HargaKonstan).
- Ditambahkan `Real_UMP` = UMP / (1 + Inflasi) sebagai fitur engineered.
- Ditambahkan `Provinsi` sebagai fitur kategorikal (one-hot encoding).
- Split diubah dari random ke chronological (lebih valid untuk data panel time-series).

---

## Preprocessing & Data Pipeline (Anti-Leakage)

Proses pengolahan data dibagi menjadi dua tahapan ketat untuk **mencegah Data Leakage**:

1. **`preprocessing.py`**: Hanya melakukan pembersihan teks dan penggabungan secara waktu (join).
2. **`data_pipeline.py`**: Melakukan Train/Val/Test Split TERLEBIH DAHULU, kemudian Scaling, Imputasi, dan pembuatan Fitur Lag.

---

## Dashboard Web

Dashboard dibangun dengan **Django + Bootstrap 5 + Chart.js** dan menyediakan 3 halaman:

| Halaman | Fungsi | Input User |
|---------|--------|------------|
| **Beranda** | Grafik historis inflasi & BI Rate | — |
| **Forecast Inflasi** | Prediksi inflasi N bulan ke depan | Jumlah bulan (1–24) |
| **Simulasi Daya Beli** | Estimasi pengeluaran per kapita | Provinsi, UMP, TPT, PDRB, Inflasi |

---

## Cara Menjalankan

```bash
# 1. Install dependensi
pip install -r requirements.txt

# 2. Eksplorasi dataset (opsional)
python explore_datasets.py

# 3. Jalankan preprocessing (Menghasilkan clean_*.csv)
python preprocessing.py

# 4. Tes Pipeline
python data_pipeline.py

# 5. Training & simpan model LSTM
python save_best_lstm.py

# 6. Training & simpan model Ridge
python save_best_model.py

# 7. Jalankan web dashboard
cd dashboard
python manage.py runserver
```

> **Catatan Windows**: Jalankan preprocessing dengan `$env:PYTHONIOENCODING='utf-8'; python preprocessing.py` jika ada error encoding.

---

## Anggota Kelompok E

| Nama | NIM |
|------|-----|
| Muhammad Rajif Al Farikhi | 162112133008 |
| Sahrul Adicandra Effendy | 164231013 |
| Semaya David Petroes Putra | 164231048 |
| Adrina Firda Marwah | 164231087 |
| Okan Athallah Maredith | 164231088 |

---

## Referensi Data
- Badan Pusat Statistik (BPS): https://www.bps.go.id
- Bank Indonesia: https://www.bi.go.id
- Open Data Jabar: https://opendata.jabarprov.go.id
- Investing.com: https://id.investing.com
- IndexMundi: https://www.indexmundi.com
