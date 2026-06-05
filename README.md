# Prediksi Inflasi dan Dampaknya terhadap Daya Beli

> **Kelompok E – Machine Learning SD-A1, Universitas Airlangga**

Proyek ini membangun dua model machine learning: (1) **Forecasting inflasi bulanan** menggunakan **Ensemble LSTM + ARIMA + Prophet** dengan 44 fitur time-series, dan (2) **Regresi daya beli per kapita** per provinsi menggunakan Ridge Regression dengan 15 fitur panel data.

---

## Tim Pengembang

| Nama | NIM | Role |
|------|-----|------|
| Muhammad Rajif Al Farikhi | 162112133008 | Backend |
| Sahrul Adicandra Effendy | 164231013 | Backend + Data Scrapper |
| Semaya David Petroes Putra | 164231048 | Modelling |
| Adrina Firda Marwah | 164231087 | Modelling |
| Okan Athallah Maredith | 164231088 | Frontend |

---

## Dataset

Total **23 dataset** dari BPS, Bank Indonesia, World Bank, Yahoo Finance, FRED, FAO.

### Model 1 — Forecasting Inflasi (LSTM, 44 fitur)

#### Domestik

| # | Dataset | Sumber | Link |
|---|---------|--------|------|
| 1 | Indeks Harga Konsumen (Umum) | BPS | https://www.bps.go.id/id/statistics-table/2/MiMy/indeks-harga-konsumen--umum-.html |
| 2 | Inflasi Bulanan (M to M) | BPS | https://www.bps.go.id/id/statistics-table/2/MSMy/inflasi--umum-.html |
| 3 | Tingkat Inflasi Harga Konsumen Nasional Tahun Kalender (Y-to-D) | BPS | https://www.bps.go.id/id/statistics-table/1/OTE0IzE=/tingkat-inflasi-harga-konsumen-nasional-tahun-kalender--y-to-d---sup-1--sup---2022-100-.html |
| 4 | BI Rate Data Inflasi | Bank Indonesia | https://www.bi.go.id/id/statistik/indikator/data-inflasi.aspx |
| 7 | Data Historis USD/IDR | Investing.com | https://id.investing.com/currencies/usd-idr-historical-data |
| 12 | Inflasi Umum, Inti, Harga Diatur, Bergejolak (M-to-M dan Y-to-D) | BPS | https://www.bps.go.id/id/statistics-table/1/OTA4IzE=/inflasi-umum--inti--harga-yang-diatur-pemerintah--dan-barang-bergejolak-inflasi-indonesia--2009-2025.html |
| 13 | Harga Bulanan Minyak Mentah (WTI) | IndexMundi | https://www.indexmundi.com/commodities/?commodity=crude-oil&months=300 |

#### Internasional

| # | Dataset | Sumber | Link |
|---|---------|--------|------|
| 14 | Geopolitical Risk Index | policyuncertainty.com | https://www.policyuncertainty.com/gpr.html |
| 15 | FAO Food Price Index | FAO | https://www.fao.org/worldfoodsituation/foodpricesindex/en/ |
| 16 | Commodity Markets (26 komoditas) | World Bank | https://www.worldbank.org/en/research/commodity-markets |

Komoditas World Bank CMO yang digunakan: Palm Oil, Coal (AU & SA), Coffee (Robusta & Arabica), Wheat (SRW & HRW), Soybeans, Soybean Oil, Sugar, Rubber (TSR20 & RSS3), Cotton, Rice Thailand, Coconut Oil, Groundnuts, Fish Meal, Maize, Tin, Nickel, Copper, Aluminum, Iron Ore, Natural Gas (US & Europe), LNG Japan.

Data internasional lainnya (Crude Oil Brent, DXY, Fed Funds Rate, Gold Price, CPO Price) diunduh otomatis dari Yahoo Finance dan FRED melalui `download_international.py`.

---

### Model 2 — Regresi Daya Beli (Ridge, 15 fitur, panel provinsi × tahun)

#### Data Provinsi (manual scraping dari BPS)

| # | Dataset | Fitur | Sumber | Link |
|---|---------|-------|--------|------|
| 5 | Upah Minimum Provinsi (UMP) | `UMP` | BPS Jateng | https://jateng.bps.go.id/id/statistics-table/2/MjgyNCMy/upah-minimum-provinsi-ump-per-bulan-menurut-provinsi-di-indonesia.html |
| 6 | Rata-rata Pengeluaran per Kapita Sebulan | `Total_Pengeluaran` (target) | BPS | https://www.bps.go.id/id/statistics-table/3/V1ZKMWVrSTNOek5ZZUZOcVZEZGFValJvV0hWalFUMDkjMyMwMDAw/rata-rata-pengeluaran-per-kapita-sebulan-makanan-dan-bukan-makanan-di-daerah-perkotaan-dan-perdesaan-menurut-provinsi--rupiah-.html?year=2025 |
| 8 | Tingkat Pengangguran Terbuka (Semester & Provinsi) | `TPT` | Open Data Jabar | https://opendata.jabarprov.go.id/id/dataset/tingkat-pengangguran-terbuka-berdasarkan-semester-dan-provinsi-di-indonesia |
| 9 | TPT & TPAK Menurut Provinsi | `TPT` | BPS | https://www.bps.go.id/id/statistics-table/3/V2pOVWJWcHJURGg0U2pONFJYaExhVXB0TUhacVFUMDkjMw%3D%3D/tingkat-pengangguran-terbuka--tpt--dan-tingkat-partisipasi-angkatan-kerja--tpak--menurut-provinsi--2019.html |
| 10 | PDRB Per Kapita (Ribu Rupiah) | `PDRB_HargaKonstan` | BPS | https://www.bps.go.id/id/statistics-table/2/Mjg4IzI=/-seri-2010--produk-domestik-regional-bruto-per-kapita--ribu-rupiah-.html |
| 11 | Persentase Penduduk Miskin | `Pct_Penduduk_Miskin` | Open Data Jabar | https://opendata.jabarprov.go.id/id/dataset/persentase-penduduk-miskin-berdasarkan-provinsi-di-indonesia |
| 17 | Gini Ratio Menurut Provinsi | `Gini_Rasio` | BPS | https://www.bps.go.id/id/statistics-table/2/OTgjMg==/gini-rasio--maret-2023.html |
| 18 | Indeks Pembangunan Manusia (IPM) | `IPM` | BPS | https://www.bps.go.id/id/statistics-table/2/NDk0IzI=/-metode-baru--indeks-pembangunan-manusia-menurut-provinsi.html |
| 19 | Garis Kemiskinan (Rp/Kapita/Bulan) | `Garis_Kemiskinan` | BPS | https://www.bps.go.id/id/statistics-table/2/MTk1IzI=/poverty-line--rupiah-kapita-month--by-province-and-area.html |
| 20 | Jumlah Penduduk Menurut Provinsi (Ribu Jiwa) | `Jumlah_Penduduk` | BPS Sulut | https://sulut.bps.go.id/id/statistics-table/2/OTU4IzI=/jumlah-penduduk-menurut-provinsi-di-indonesia.html |
| - | Distribusi & Demografi Penduduk per Provinsi | `Pct_Populasi` | BPS | https://www.bps.go.id |
| 21 | Akses Air Minum Layak per Provinsi (%) | `Pct_Akses_Air_Bersih` | BPS | https://www.bps.go.id/id/statistics-table/2/ODU0IzI=/persentase-rumah-tangga-yang-memiliki-akses-terhadap-sumber-air-minum-layak-menurut-provinsi-dan-klasifikasi-desa--persen-.html |
| 22 | Rata-rata Konsumsi Protein Per Kapita (gram/hari) | `Protein_gram_per_hari` | BPS | https://www.bps.go.id/id/statistics-table/1/MTk4NiMx/rata-rata-harian-konsumsi-protein-per-kapita-dan-konsumsi-kalori-per-kapita-tahun-1990-2023.html |
| 23 | Persentase Rumah Tangga menurut Provinsi | - | BPS | https://www.bps.go.id/id/statistics-table/1/MTYwMyMx/persentase-rumah-tangga-menurut-provinsi--jenis-kelamin-kepala-rumah-tangga--dan-banyaknya-anggota-rumah-tangga--2009-2024.html |

**Catatan:**
- Dataset #23 (Rumah Tangga) berisi persentase distribusi jumlah anggota rumah tangga, bukan jumlah absolut. Belum digunakan dalam model.
- Dataset #2 (Inflasi Bulanan M-to-M) sudah mencakup semua kota BPS, sehingga Inflasi per Kota tidak diperlukan terpisah.
- Kredit Konsumsi per Provinsi dihapus karena tidak tersedia dari sumber publik.

#### Data Nasional (World Bank API, auto-download)

| # | Indikator | Kode | Sumber | Link |
|---|-----------|------|--------|------|
| WB1 | Inflasi (annual %) | `FP.CPI.TOTL.ZG` | World Bank | https://data.worldbank.org |
| WB2 | GDP per Capita PPP (constant 2017 $) | `NY.GDP.PCAP.PP.KD` | World Bank | https://data.worldbank.org |
| WB3 | Unemployment (% total) | `SL.UEM.TOTL.ZS` | World Bank | https://data.worldbank.org |
| WB4 | Poverty Headcount (%) | `SI.POV.NAHC` | World Bank | https://data.worldbank.org |

Diunduh otomatis oleh `download_domestic.py` melalui World Bank Indicators API.

---

## Model Machine Learning

### Model 1 — Forecasting Inflasi (LSTM)

| Aspek | Spesifikasi |
|-------|-------------|
| Arsitektur | LSTM 2-layer, 128 hidden units, LayerNorm, Dropout 0.3 |
| Window | 12 bulan |
| Fitur | 44 (1 target + 43 eksogenous) |
| Split | Chronological: 80% Train, 20% Validation |
| Scaler | MinMaxScaler (terpisah fitur & target) |
| Optimizer | AdamW + ReduceLROnPlateau + Early Stopping |
| Forecast | 1-2 bulan ke depan (recursive) |

### Model 2 — Regresi Daya Beli (Ridge)

| Aspek | Spesifikasi |
|-------|-------------|
| Model | Ridge Regression (alpha tuning via GridSearchCV) |
| Fitur | 15 numerik + Provinsi (one-hot) |
| Split | Chronological: Train (≤2023), Test (≥2024) |
| Best Alpha | 0.1 |
| Metrik | R², MAE, RMSE |

**Fitur numerik (15):** TPT, PDRB_HargaKonstan, Inflasi_Rata_Tahunan, Gini_Rasio, IPM, Garis_Kemiskinan, Jumlah_Penduduk, Pct_Populasi, Pct_Akses_Air_Bersih, Protein_gram_per_hari, Inflasi_WB_Annual, GDP_PerCapita_PPP, Pct_Unemployment_WB, Poverty_Headcount_Pct, Real_UMP.

**Feature Engineering:**
- `Real_UMP` = UMP / (1 + Inflasi_Rata_Tahunan) → daya beli upah riil
- Carry-forward imputation per provinsi untuk fitur yang tidak tersedia di semua tahun

### Hasil

| Metrik | Train | Test |
|--------|-------|------|
| R² | 0.9916 | 0.9080 |
| MAE | - | Rp 93,249 |
| RMSE | - | Rp 108,754 |

### Analisis Lanjutan (Notebook)

Notebook `analisis_daya_beli_regresi.ipynb` berisi analisis tambahan:

| Model | Test R² | Catatan |
|-------|---------|---------|
| XGBoost | 0.834 | Model prediktif terbaik (non-linear) |
| Random Forest | 0.693 | Tree ensemble |
| Lasso | 0.679 | Feature selection otomatis |
| Panel FE Macro | 0.427 | Fixed Effects + 4 variabel makro (interpretatif) |
| OLS Baseline | 0.183 | 3 fitur dasar |

Temuan ekonomi dari Panel FE:
- **TPT** (pengangguran): efek negatif paling kuat (t=-6.97)
- **PDRB**: pertumbuhan ekonomi → daya beli naik (t=4.25)
- **Real_UMP**: upah riil → daya beli naik (t=4.18)
- **UMP +10%** → daya beli naik 4.7% (simulasi counterfactual)

---

## Struktur Proyek

```
├── datasets/
│   ├── BI Rate (Data Inflasi)/
│   ├── Data Historis USD_IDR/
│   ├── Harga Bulanan Minyak Mentah/
│   ├── Indeks Harga Konsumen (Umum)/
│   ├── Inflasi Bulanan/
│   ├── Inflasi Umum, Inti, Harga Diatur, Bergejolak/
│   ├── Persentase Penduduk Miskin/
│   ├── Produk Domestik Regional Bruto Per Kapita/
│   ├── Rata-rata Pengeluaran per Kapita/
│   ├── Tingkat Pengangguran Terbuka/
│   ├── Upah Minimum Provinsi/
│   ├── international/
│   │   ├── CMO-Historical-Data-Monthly.xlsx
│   │   ├── cpo_price.csv, crude_oil_brent.csv, data_gpr_export.csv
│   │   ├── dxy_dollar_index.csv, fed_funds_rate.csv, gold_price.csv
│   │   └── ...
│   ├── domestic_baru/
│   │   ├── IPM/                      (15 CSV, 2010-2024)
│   │   ├── Jumlah_Penduduk/          (7 CSV, 2018-2024)
│   │   ├── Tingkat_Urbanisasi/       (17 CSV, 2010-2026)
│   │   ├── Gini_Rasio/               (16 XLSX, 2010-2025)
│   │   ├── Garis_Kemiskinan/         (13 XLSX, 2013-2025)
│   │   ├── Akses_Air_Bersih/         (17 XLSX, 2009-2025)
│   │   ├── Konsumsi_Protein/         (1 XLS, 1990-2025)
│   │   ├── Jumlah_Rumah_Tangga/      (1 XLS)
│   │   └── WorldBank_Nasional/       (4 CSV, auto-download)
│   └── processed/
│       ├── clean_inflasi_ts.csv       (257 × 45)
│       └── clean_daya_beli.csv        (177 × 23)
├── notebooks/
│   ├── forecasting_inflasi_models.ipynb
│   └── analisis_daya_beli_regresi.ipynb
├── dashboard/                          (Django web app)
│   └── predictions/
│       └── views.py
├── models/
│   ├── lstm_model.pt, lstm_scaler_x.pkl, lstm_scaler_y.pkl
│   ├── best_daya_beli_ridge.pkl
│   └── best_daya_beli_xgboost.pkl
├── preprocessing.py                    (ETL pipeline)
├── download_international.py           (auto-download Yahoo, FRED, FAO)
├── download_domestic.py                (auto-download World Bank API)
├── save_lstm_model.py                  (training LSTM)
├── save_ridge_model.py                 (training Ridge)
└── requirements.txt
```

---

## Dashboard

| Halaman | URL | Fungsi |
|---------|-----|--------|
| Landing | `/` | Overview inflasi & daya beli |
| Forecasting | `/forecasting/` | Prediksi inflasi 1-2 bulan ke depan |
| Daya Beli | `/daya-beli/` | Simulasi dampak inflasi terhadap daya beli (slider interaktif) |

---

## Performa Model (Walk-Forward Backtest)

| Model | MAE | RMSE | sMAPE | Test Window |
|-------|-----|------|-------|-------------|
| **Naive baseline** (last value) | 0.4538 | 0.6797 | 122.01% | 24 bulan |
| **ARIMA(0,0,1)** | 0.3876 | 0.5211 | 108.97% | 24 bulan |
| **LSTM (44 fitur)** | 0.7237 | 0.7685 | 119.84% | 12 bulan |
| **Prophet** | 0.1962 | 0.2865 | 85.21% | 12 bulan |
| **Ensemble** (LSTM 0.2 + ARIMA 0.3 + Prophet 0.5) | **0.2590** | **0.3118** | **79.98%** | 12 bulan |

**Catatan:**
- Data time series: 257 bulan (Jan 2005 – Mei 2026), `Inflasi_MoM` BPS
- Backtest menggunakan **walk-forward** (prediksi bulan demi bulan, update training setiap step)
- **sMAPE** lebih robust dari MAPE untuk data inflasi yang mengandung nilai deflasi
- Prophet menjadi model terbaik individual karena menangkap pola musiman (yearly seasonality)
- Ensemble lebih robust karena merata-ratakan error antar model

### Perbedaan "0.28%" vs "2-3%" di Berita

Dataset kita menggunakan **M-to-M (Month-to-Month)**: perubahan dari bulan lalu → selalu kecil (0-1%).
Berita/TradingView/BPS pakai **Y-o-Y (Year-on-Year)**: dibanding 12 bulan lalu → biasanya 2-3%.

Contoh data Mei 2026:
- M-to-M: **+0.28%** (perubahan dari April 2026)
- Y-o-Y: **+3.10%** (perubahan dari Mei 2025)
- Y-to-D: **+1.51%** (akumulasi sejak Januari 2026)

Dashboard menampilkan **ketiganya** agar user tidak bingung.

## Cara Menjalankan

```bash
pip install -r requirements.txt

# Unduh data internasional
python download_international.py

# Unduh data domestik (World Bank API)
python download_domestic.py

# Preprocessing
python preprocessing.py

# Train model
python save_lstm_model.py
python save_ridge_model.py

# Jalankan dashboard
cd dashboard && python manage.py runserver
```

---

## Anggota Kelompok E

| Nama | NIM | Role |
|------|-----|-----|
| Muhammad Rajif Al Farikhi | 162112133008 | Backend |
| Sahrul Adicandra Effendy | 164231013 | Backend + Data Scrapper | 
| Semaya David Petroes Putra | 164231048 | Modelling |
| Adrina Firda Marwah | 164231087 | Modelling |
| Okan Athallah Maredith | 164231088 | Frontend |
