# Prediksi Inflasi dan Dampaknya terhadap Daya Beli

> **Kelompok E – Machine Learning SD-A1, Universitas Airlangga**

Proyek ini membangun dua model machine learning:
1. **Forecasting inflasi bulanan** menggunakan Ensemble LSTM + ARIMA + Prophet (44 fitur time-series)
2. **Regresi daya beli per kapita** per provinsi menggunakan Ridge Regression (15 fitur panel data)

---

## Dataset

Total **30 dataset** dari 6 sumber: BPS, Bank Indonesia, World Bank, Yahoo Finance, FRED, FAO.

### A. Dataset Domestik (BPS & Bank Indonesia)

| # | Dataset | Tahun | Jumlah File | Format | Sumber |
|---|---------|-------|-------------|--------|--------|
| 1 | Inflasi Bulanan (M-to-M) | 2005–2026 | 22 CSV | Per kota, 12 kolom bulanan | BPS |
| 2 | Indeks Harga Konsumen (IHK) Umum | 2005–2023 | 19 CSV | Per kota, 12 kolom bulanan | BPS |
| 3 | Inflasi Komponen (Umum, Inti, Harga Diatur, Bergejolak) | 2009–2026 | 1 CSV | M-to-M dan Y-to-D, 4 kolom komponen | BPS |
| 4 | BI Rate / Data Inflasi | 2005–2026 | 1 Excel | Suku bunga acuan bulanan | Bank Indonesia |
| 5 | Kurs USD/IDR Historis | 2005–2026 | 1 CSV (5.533 baris) | Data harian (Open, High, Low, Close) | Investing.com |
| 6 | Harga Minyak Mentah (WTI) | 2001–2026 | 1 CSV (301 baris) | Harga USD/barel bulanan | IndexMundi + Yahoo Finance |
| 7 | Upah Minimum Provinsi (UMP) | 2021–2025 | 5 CSV | Per provinsi, UMP per bulan | BPS |
| 8 | Rata-rata Pengeluaran per Kapita Sebulan | 2017–2025 | 9 CSV | Per provinsi, Makanan + Bukan Makanan + Total | BPS |
| 9 | Tingkat Pengangguran Terbuka (TPT) Per Provinsi | 2017–2025 | 8 CSV | Per provinsi, per semester | BPS |
| 10 | PDRB Per Kapita (Ribu Rupiah) | 2010–2025 | 16 CSV | Per provinsi, Harga Berlaku & Harga Konstan | BPS |
| 11 | Persentase Penduduk Miskin per Provinsi | 2010–2024 | 1 CSV (514 baris) | Per provinsi per tahun | Open Data Jabar |
| 12 | Gini Ratio per Provinsi | 2010–2025 | 16 XLSX | 38 provinsi, rata-rata Semester 1+2 | BPS |
| 13 | Indeks Pembangunan Manusia (IPM) per Provinsi | 2010–2024 | 15 CSV | Per provinsi, 1 kolom IPM | BPS |
| 14 | Garis Kemiskinan (Rp/Kapita/Bulan) | 2013–2025 | 13 XLSX | Per provinsi, garis kemiskinan | BPS |
| 15 | Jumlah Penduduk per Provinsi (Ribu Jiwa) | 2018–2024 | 7 CSV | Per provinsi, jumlah penduduk | BPS |
| 16 | Tingkat Urbanisasi per Provinsi | 2010–2026 | 12 CSV | Per provinsi, % penduduk perkotaan | BPS |
| 17 | Akses Air Minum Layak per Provinsi | 2009–2025 | 17 XLSX | Per provinsi, % rumah tangga akses air layak | BPS |
| 18 | Rata-rata Konsumsi Protein Per Kapita | 1990–2025 | 1 XLS | Nasional, gram/hari/kapita | BPS |

### B. Dataset Internasional

| # | Dataset | Tahun | Jumlah File | Sumber |
|---|---------|-------|-------------|--------|
| 19 | Crude Oil Brent (USD/Barel) | 2007–2026 | 1 CSV | Yahoo Finance (BZ=F) |
| 20 | Indeks Dollar AS (DXY) | 2003–2026 | 1 CSV | Yahoo Finance (DX-Y.NYB) |
| 21 | The Fed Funds Rate (%) | 2003–2026 | 1 CSV | FRED |
| 22 | Gold Price (USD/oz) | 2003–2026 | 1 CSV | Yahoo Finance (GC=F) |
| 23 | CPO Price (Crude Palm Oil, USD/mt) | 2010–2026 | 1 CSV | Yahoo Finance (CPO=F) |
| 24 | Geopolitical Risk Index (GPR) | 1985–2026 | 1 CSV | Caldara & Iacoviello |
| 25 | FAO Food Price Index | 1990–2026 | 1 Excel | FAO |
| 26 | World Bank Commodity Markets (26 Komoditas) | 1960–2026 | 1 Excel (796 baris) | World Bank CMO |

**26 komoditas World Bank CMO:** Palm Oil, Coal (AU & SA), Coffee (Robusta & Arabica), Wheat (SRW & HRW), Soybeans, Soybean Oil, Sugar, Rubber (TSR20 & RSS3), Cotton, Rice Thailand, Coconut Oil, Groundnuts, Fish Meal, Maize, Tin, Nickel, Copper, Aluminum, Iron Ore, Natural Gas (US & Europe), LNG Japan.

### C. Dataset World Bank API (Nasional, Auto-download)

| # | Indikator | Kode | Tahun | Sumber |
|---|-----------|------|-------|--------|
| 27 | Inflasi (annual %) | `FP.CPI.TOTL.ZG` | 2010–2024 | World Bank |
| 28 | GDP per Capita PPP (constant 2017 $) | `NY.GDP.PCAP.PP.KD` | 2010–2024 | World Bank |
| 29 | Unemployment (% total) | `SL.UEM.TOTL.ZS` | 2010–2024 | World Bank |
| 30 | Poverty Headcount (%) | `SI.POV.NAHC` | 2010–2024 | World Bank |

---

## Mapping Dataset ke Model

### Model 1 — Forecasting Inflasi (Ensemble LSTM + ARIMA + Prophet)

**Output:** `clean_inflasi_ts.csv` (257 baris × 46 kolom, Jan 2005 – Mei 2026)

| # | Dataset | Kolom yang Digunakan | Peran |
|---|---------|---------------------|-------|
| 1 | Inflasi Bulanan M-to-M | `Inflasi_MoM` | Target |
| 2 | IHK Umum | `IHK` | Fitur (level harga) |
| 3 | Inflasi Komponen | `Inflasi_Umum_MoM`, `Inflasi_Inti_MoM`, `Inflasi_HargaDiatur_MoM`, `Inflasi_Bergejolak_MoM` | Fitur (komponen inflasi) |
| 4 | BI Rate | `BI_Rate` | Fitur (kebijakan moneter) |
| 5 | Kurs USD/IDR | `USD_IDR` | Fitur (nilai tukar) |
| 6 | Harga Minyak WTI | `Harga_Minyak_USD` | Fitur (energi global) |
| 19 | Crude Oil Brent | `Brent_USD` | Fitur (energi global) |
| 20 | DXY | `DXY` | Fitur (kekuatan USD) |
| 21 | Fed Funds Rate | `FedRate_Pct` | Fitur (kebijakan moneter US) |
| 22 | Gold Price | `Gold_USD` | Fitur (safe haven) |
| 23 | CPO Price | `CPO_USD` | Fitur (komoditas Indonesia) |
| 24 | GPR Index | `GPR_Index` | Fitur (risiko geopolitik) |
| 25 | FAO Food Price Index | `FAO_FPI` | Fitur (harga pangan global) |
| 26 | World Bank CMO (26 komoditas) | `CMO_*` (26 kolom) | Fitur (komoditas global) |
| - | Engineering | `Bulan_Sin`, `Bulan_Cos`, `Oil_x_USDIDR` | Fitur musiman & interaksi |

### Model 2 — Regresi Daya Beli (Ridge Regression)

**Output:** `clean_daya_beli.csv` (177 baris × 23 kolom, 38 provinsi × 2021–2025)

| # | Dataset | Kolom yang Digunakan | Peran |
|---|---------|---------------------|-------|
| 7 | UMP | `UMP`, `Real_UMP` (UMP / (1 + Inflasi_YoY)) | Fitur (upah minimum riil) |
| 8 | Pengeluaran per Kapita | `Total_Pengeluaran`, `Pengeluaran_Makanan`, `Pengeluaran_Bukan_Makanan` | **Target** + komponen |
| 9 | TPT Per Provinsi | `TPT` | Fitur (pengangguran) |
| 10 | PDRB Per Kapita | `PDRB_HargaBerlaku`, `PDRB_HargaKonstan` | Fitur (produktivitas) |
| 11 | Penduduk Miskin | `Pct_Penduduk_Miskin` | Fitur (kemiskinan) |
| 12 | Gini Ratio | `Gini_Rasio` | Fitur (ketimpangan) |
| 13 | IPM | `IPM` | Fitur (kesejahteraan) |
| 14 | Garis Kemiskinan | `Garis_Kemiskinan` | Fitur (ambang kemiskinan) |
| 15 | Jumlah Penduduk | `Jumlah_Penduduk` | Fitur (skala demografi) |
| 16 | Urbanisasi | `Pct_Populasi` | Fitur (% perkotaan) |
| 17 | Akses Air Bersih | `Pct_Akses_Air_Bersih` | Fitur (infrastruktur) |
| 18 | Konsumsi Protein | `Protein_gram_per_hari` | Fitur (nutrisi) |
| 1 | Inflasi M-to-M | `Inflasi_YoY` (agregat tahunan) | Fitur (inflasi) |
| 27 | World Bank Inflasi | `Inflasi_WB_Annual` | Fitur (inflasi referensi) |
| 28 | World Bank GDP PPP | `GDP_PerCapita_PPP` | Fitur (kemakmuran) |
| 29 | World Bank Unemployment | `Pct_Unemployment_WB` | Fitur (pengangguran nasional) |
| 30 | World Bank Poverty | `Poverty_Headcount_Pct` | Fitur (kemiskinan nasional) |
| - | Categorical | `Provinsi` (one-hot encoding) | Fitur (identitas provinsi) |

---

## Model Machine Learning

### Model 1 — Forecasting Inflasi (Ensemble)

| Aspek | Spesifikasi |
|-------|-------------|
| Arsitektur | LSTM 2-layer, 128 hidden units, LayerNorm, Dropout 0.3 |
| Window | 12 bulan |
| Fitur | 44 (1 target + 43 eksogenous) |
| Split | Chronological: 80% Train, 20% Validation |
| Scaler | MinMaxScaler (terpisah fitur & target) |
| Optimizer | AdamW + ReduceLROnPlateau + Early Stopping |
| ARIMA | ARIMA(0,0,1) — univariate, ADF test p < 0.001 |
| Prophet | Yearly seasonality + 3 regressors (USD/IDR, Oil, BI Rate) |
| Ensemble | Weighted Average: Prophet 0.50 + ARIMA 0.30 + LSTM 0.20 (LSTM berkontribusi dari 44 fitur multivariate) |

### Model 2 — Regresi Daya Beli (Ridge)

| Aspek | Spesifikasi |
|-------|-------------|
| Model | Ridge Regression (alpha tuning via GridSearchCV) |
| Fitur | 15 numerik + Provinsi (one-hot encoding) |
| Split | Chronological: Train (≤2023), Test (≥2024) |
| Best Alpha | 0.1 |
| Pipeline | StandardScaler + OneHotEncoder + Ridge |

**Fitur numerik (15):** TPT, PDRB_HargaKonstan, Inflasi_YoY, Gini_Rasio, IPM, Garis_Kemiskinan, Jumlah_Penduduk, Pct_Populasi, Pct_Akses_Air_Bersih, Protein_gram_per_hari, Inflasi_WB_Annual, GDP_PerCapita_PPP, Pct_Unemployment_WB, Poverty_Headcount_Pct, Real_UMP.

### Performa Model

#### Ridge Regression

| Metrik | Train | Test |
|--------|-------|------|
| R² | 0.9916 | 0.9080 |
| MAE | - | Rp 93.249 |
| RMSE | - | Rp 108.754 |

#### Walk-Forward Backtest (Inflasi)

| Model | MAE | RMSE | sMAPE | Catatan |
|-------|-----|------|-------|---------|
| **ARIMA(0,0,1)** | 0.3876 | 0.5211 | 108.97% | Statistical baseline, univariate |
| **Prophet** | 0.1962 | 0.2865 | 85.21% | Terbaik individual, menangkap seasonality |
| **Ensemble** | **0.2590** | **0.3118** | **79.98%** | Weighted avg: Prophet 50% + ARIMA 30% + LSTM 20% |

**Catatan:**
- sMAPE lebih robust dari MAPE untuk data yang mengandung nilai deflasi (dekat nol)
- Prophet menjadi model terbaik individual karena menangkap pola musiman (yearly seasonality)
- Ensemble lebih robust karena merata-ratakan error antar model
- LSTM (44 fitur) tetap digunakan dalam ensemble untuk kontribusi multivariate, meskipun single-model performance-nya lebih rendah

#### Perbedaan "0.28%" vs "2-3%" di Berita

Dataset kita menggunakan **M-to-M (Month-to-Month)**: perubahan dari bulan lalu → selalu kecil (0-1%).
Berita/TradingView/BPS pakai **Y-o-Y (Year-on-Year)**: dibanding 12 bulan lalu → biasanya 2-3%.

Contoh data Mei 2026:
- M-to-M: **+0.28%** (perubahan dari April 2026)
- Y-o-Y: **+3.10%** (perubahan dari Mei 2025)
- Y-to-D: **+1.51%** (akumulasi sejak Januari 2026)

Dashboard menampilkan **ketiganya** agar user tidak bingung.

---

## Struktur Proyek

```
├── datasets/
│   ├── Inflasi Bulanan/                    (22 CSV, 2005-2026)    — #1
│   ├── Indeks Harga Konsumen (Umum)/       (19 CSV, 2005-2023)    — #2
│   ├── Inflasi Umum, Inti, Harga Diatur*/  (1 CSV, 2009-2026)     — #3
│   ├── BI Rate (Data Inflasi)/             (1 Excel, 2005-2026)   — #4
│   ├── Data Historis USD_IDR/              (1 CSV, 2005-2026)     — #5
│   ├── Harga Bulanan Minyak Mentah*/       (1 CSV, 2001-2026)     — #6
│   ├── Upah Minimum Provinsi/              (5 CSV, 2021-2025)     — #7
│   ├── Rata-rata Pengeluaran*/             (9 CSV, 2017-2025)     — #8
│   ├── Tingkat Pengangguran*/              (8 CSV, 2017-2025)     — #9
│   ├── Produk Domestik*/                   (16 CSV, 2010-2025)    — #10
│   ├── Persentase Penduduk Miskin*/        (1 CSV, 2010-2024)     — #11
│   ├── domestic_baru/
│   │   ├── Gini_Rasio/                     (16 XLSX, 2010-2025)  — #12
│   │   ├── IPM/                            (15 CSV, 2010-2024)   — #13
│   │   ├── Garis_Kemiskinan/               (13 XLSX, 2013-2025)  — #14
│   │   ├── Jumlah_Penduduk/                (7 CSV, 2018-2024)    — #15
│   │   ├── Tingkat_Urbanisasi/             (12 CSV, 2010-2026)   — #16
│   │   ├── Akses_Air_Bersih/               (17 XLSX, 2009-2025)  — #17
│   │   ├── Konsumsi_Protein/               (1 XLS, 1990-2025)    — #18
│   │   └── WorldBank_Nasional/             (4 CSV, auto-download) — #27-30
│   └── international/
│       ├── crude_oil_brent.csv             (2007-2026)  — #19
│       ├── dxy_dollar_index.csv            (2003-2026)  — #20
│       ├── fed_funds_rate.csv              (2003-2026)  — #21
│       ├── gold_price.csv                  (2003-2026)  — #22
│       ├── cpo_price.csv                   (2010-2026)  — #23
│       ├── data_gpr_export.csv             (1985-2026)  — #24
│       ├── ffpi-data-2026-05.xlsx          (1990-2026)  — #25
│       └── CMO-Historical-Data-Monthly.xlsx (1960-2026) — #26
├── models/
│   ├── lstm_model.pt                       (PyTorch LSTM)
│   ├── lstm_scaler_x.pkl, lstm_scaler_y.pkl
│   ├── arima_inflasi.pkl, arima_forecast.pkl, arima_metrics.pkl
│   ├── ensemble_forecast.pkl, ensemble_metrics.pkl
│   ├── best_daya_beli_ridge.pkl
│   └── best_daya_beli_xgboost.pkl
├── notebooks/
│   ├── forecasting_inflasi_models.ipynb    (LSTM + ARIMA + Ensemble)
│   └── analisis_daya_beli_regresi.ipynb    (Ridge + XGBoost + Panel FE)
├── dashboard/                              (Django web app)
│   └── predictions/
│       ├── views.py
│       ├── templates/predictions/
│       └── static/predictions/
├── preprocessing.py                        (ETL pipeline, 33 loaders)
├── train_ensemble.py                       (Ensemble LSTM+ARIMA+Prophet)
├── retrain_arima.py                        (Re-train ARIMA konsisten)
├── save_arima_model.py                     (Training ARIMA + grid search)
├── save_lstm_model.py                      (Training LSTM)
├── save_ridge_model.py                     (Training Ridge)
├── download_domestic.py                    (Auto-download World Bank API)
└── requirements.txt
```

---

## Dashboard

| Halaman | URL | Fungsi |
|---------|-----|--------|
| Home | `/` | Landing page, overview proyek, tim, fitur |
| Dashboard | `/dashboard/` | KPI inflasi (M-to-M, Y-o-Y, Y-to-D), USD/IDR live, Rupiah Purchasing Power |
| Forecasting | `/forecasting/` | Histori inflasi multi-tahun + prediksi 1 bulan (4 model), year range selector, model comparison |
| Daya Beli | `/daya-beli/` | Simulasi dampak inflasi terhadap daya beli (slider interaktif) |
| Datasets | `/datasets/` | Explorer 26 dataset dengan filter & preview |
| Compare | `/compare/` | Perbandingan antar provinsi (line chart + radar) |
| Map | `/map/` | Peta heatmap Indonesia (Leaflet + leaflet.heat) |
| Scenarios | `/scenarios/` | What-if scenarios (6 skenario) |

---

## Cara Menjalankan

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data
python download_domestic.py        # World Bank API

# 3. Preprocessing
python preprocessing.py

# 4. Train models
python save_lstm_model.py          # LSTM
python save_ridge_model.py         # Ridge
python save_arima_model.py         # ARIMA
python train_ensemble.py           # Ensemble (LSTM+ARIMA+Prophet)

# 5. Run dashboard
cd dashboard && python manage.py runserver
```

---

## Anggota Kelompok E

| Nama | NIM | Role |
|------|-----|------|
| Muhammad Rajif Al Farikhi | 162112133008 | Backend |
| Sahrul Adicandra Effendy | 164231013 | Backend + Data Scrapper |
| Semaya David Petroes Putra | 164231048 | Modelling |
| Adrina Firda Marwah | 164231087 | Modelling |
| Okan Athallah Maredith | 164231088 | Frontend |
