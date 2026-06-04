# Prediksi Inflasi dan Dampaknya terhadap Daya Beli

> **Kelompok E – Machine Learning SD-A1, Universitas Airlangga**

---

## Deskripsi Proyek

Proyek ini membangun sistem prediksi inflasi dan analisis dampaknya terhadap daya beli masyarakat Indonesia. Terdapat dua model utama:

1. **Forecasting (LSTM)** — Memprediksi nilai inflasi bulanan ke depan berdasarkan data historis 44 fitur, mencakup indikator domestik, internasional, geopolitik, dan 24 komoditas global untuk mengantisipasi dampak peristiwa global terhadap inflasi Indonesia.
2. **Regresi (Ridge)** — Mengukur pengaruh inflasi terhadap daya beli masyarakat (pengeluaran per kapita) per provinsi.

---

## Struktur Proyek

```
Project-Machine-Learning/
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
│   │   ├── cpo_price.csv
│   │   ├── crude_oil_brent.csv
│   │   ├── data_gpr_export.csv
│   │   ├── dxy_dollar_index.csv
│   │   ├── fed_funds_rate.csv
│   │   ├── ffpi-data-2026-05.xlsx
│   │   ├── gold_price.csv
│   │   ├── usd_idr_2026.csv
│   │   └── wti_apr_may_2026.csv
│   └── processed/
│       ├── clean_inflasi_ts.csv
│       └── clean_daya_beli.csv
├── notebooks/
│   ├── forecasting_inflasi_models.ipynb
│   └── analisis_daya_beli_regresi.ipynb
├── explore_datasets.py
├── preprocessing.py
├── download_international.py
├── requirements.txt
└── README.md
```

---

## Dataset

### Model 1 — Forecasting Inflasi

#### Data Domestik

| # | Dataset | Sumber | Periode | Peran |
|---|---------|--------|---------|-------|
| 1 | Indeks Harga Konsumen (IHK) | [BPS](https://www.bps.go.id/id/statistics-table/2/MiMy/indeks-harga-konsumen--umum-.html) | 2005–2023 | Fitur X |
| 2 | Inflasi Bulanan (M-to-M) | [BPS](https://www.bps.go.id/id/statistics-table/2/MSMy/inflasi--umum-.html) | 2005–2026 | Target Y |
| 3 | Tingkat Inflasi Tahun Kalender (Y-to-D) | [BPS](https://www.bps.go.id/id/statistics-table/1/OTE0IzE=/tingkat-inflasi-harga-konsumen-nasional-tahun-kalender--y-to-d---sup-1--sup---2022-100-.html) | Historis | Referensi |
| 4 | BI Rate / Data Inflasi | [Bank Indonesia](https://www.bi.go.id/id/statistik/indikator/data-inflasi.aspx) | 2005–2025 | Fitur X |
| 5 | Data Historis USD/IDR | [Investing.com](https://id.investing.com/currencies/usd-idr-historical-data) | 2005–2026 | Fitur X |
| 6 | Inflasi Umum, Inti, Harga Diatur, Bergejolak | [BPS](https://www.bps.go.id/id/statistics-table/1/OTA4IzE=/inflasi-umum--inti--harga-yang-diatur-pemerintah--dan-barang-bergejolak-inflasi-indonesia--2009-2025.html) | 2009–2026 | Fitur X |
| 7 | Harga Bulanan Minyak Mentah (WTI) | [IndexMundi](https://www.indexmundi.com/commodities/?commodity=crude-oil&months=300) | 2001–2026 | Fitur X |

#### Data Internasional

| # | Dataset | Sumber | Periode | Peran |
|---|---------|--------|---------|-------|
| 8 | Crude Oil Brent | [Yahoo Finance](https://finance.yahoo.com) (BZ=F) | 2007–2026 | Fitur X |
| 9 | Indeks Dollar AS (DXY) | [Yahoo Finance](https://finance.yahoo.com) (DX-Y.NYB) | 2003–2026 | Fitur X |
| 10 | The Fed Funds Rate | [FRED](https://fred.stlouisfed.org) (FEDFUNDS) | 2003–2026 | Fitur X |
| 11 | Gold Price | [Yahoo Finance](https://finance.yahoo.com) (GC=F) | 2003–2026 | Fitur X |
| 12 | CPO Price (Crude Palm Oil) | [Yahoo Finance](https://finance.yahoo.com) (CPO=F) | 2010–2026 | Fitur X |
| 13 | Geopolitical Risk Index (GPR) | [policyuncertainty.com](https://www.policyuncertainty.com/gpr.html) | 1985–2026 | Fitur X |
| 14 | FAO Food Price Index | [FAO](https://www.fao.org/worldfoodsituation/foodpricesindex/en/) | 1990–2026 | Fitur X |

#### Komoditas World Bank Commodity Markets

| # | Dataset | Sumber | Periode | Relevansi untuk Indonesia |
|---|---------|--------|---------|---------------------------|
| 15 | Palm Oil | [World Bank CMO](https://www.worldbank.org/en/research/commodity-markets) | 1960–2026 | Indonesia produsen terbesar |
| 16 | Coal (Australia) | [World Bank CMO](https://www.worldbank.org/en/research/commodity-markets) | 1960–2026 | Indonesia eksportir utama |
| 17 | Coal (South Africa) | [World Bank CMO](https://www.worldbank.org/en/research/commodity-markets) | 1960–2026 | Benchmark harga batu bara global |
| 18 | Coffee Robusta | [World Bank CMO](https://www.worldbank.org/en/research/commodity-markets) | 1960–2026 | Indonesia produsen utama |
| 19 | Coffee Arabica | [World Bank CMO](https://www.worldbank.org/en/research/commodity-markets) | 1960–2026 | Benchmark harga kopi global |
| 20 | Wheat (US SRW) | [World Bank CMO](https://www.worldbank.org/en/research/commodity-markets) | 1960–2026 | Indonesia importir besar |
| 21 | Wheat (US HRW) | [World Bank CMO](https://www.worldbank.org/en/research/commodity-markets) | 1960–2026 | Indonesia importir besar |
| 22 | Soybeans | [World Bank CMO](https://www.worldbank.org/en/research/commodity-markets) | 1960–2026 | Indonesia importir besar |
| 23 | Soybean Oil | [World Bank CMO](https://www.worldbank.org/en/research/commodity-markets) | 1960–2026 | Bahan baku minyak goreng |
| 24 | Sugar (world) | [World Bank CMO](https://www.worldbank.org/en/research/commodity-markets) | 1960–2026 | Indonesia importir besar |
| 25 | Rubber (TSR20) | [World Bank CMO](https://www.worldbank.org/en/research/commodity-markets) | 1960–2026 | Indonesia produsen utama |
| 26 | Rubber (RSS3) | [World Bank CMO](https://www.worldbank.org/en/research/commodity-markets) | 1960–2026 | Indonesia produsen utama |
| 27 | Cotton (A Index) | [World Bank CMO](https://www.worldbank.org/en/research/commodity-markets) | 1960–2026 | Bahan baku industri tekstil |
| 28 | Rice Thai 5% | [World Bank CMO](https://www.worldbank.org/en/research/commodity-markets) | 1960–2026 | Benchmark harga beras Asia |
| 29 | Coconut Oil | [World Bank CMO](https://www.worldbank.org/en/research/commodity-markets) | 1960–2026 | Indonesia produsen utama |
| 30 | Groundnuts | [World Bank CMO](https://www.worldbank.org/en/research/commodity-markets) | 1960–2026 | Kebutuhan pangan |
| 31 | Fish Meal | [World Bank CMO](https://www.worldbank.org/en/research/commodity-markets) | 1960–2026 | Pakan ikan, industri perikanan |
| 32 | Maize | [World Bank CMO](https://www.worldbank.org/en/research/commodity-markets) | 1960–2026 | Pakan ternak, pangan |
| 33 | Tin | [World Bank CMO](https://www.worldbank.org/en/research/commodity-markets) | 1960–2026 | Indonesia produsen utama |
| 34 | Nickel | [World Bank CMO](https://www.worldbank.org/en/research/commodity-markets) | 1960–2026 | Indonesia produsen utama |
| 35 | Copper | [World Bank CMO](https://www.worldbank.org/en/research/commodity-markets) | 1960–2026 | Benchmark logam industri |
| 36 | Aluminum | [World Bank CMO](https://www.worldbank.org/en/research/commodity-markets) | 1960–2026 | Benchmark logam industri |
| 37 | Iron Ore | [World Bank CMO](https://www.worldbank.org/en/research/commodity-markets) | 1960–2026 | Benchmark logam industri |
| 38 | Natural Gas (US) | [World Bank CMO](https://www.worldbank.org/en/research/commodity-markets) | 1960–2026 | Benchmark energi global |
| 39 | Natural Gas (Europe) | [World Bank CMO](https://www.worldbank.org/en/research/commodity-markets) | 1960–2026 | Benchmark energi global |
| 40 | LNG (Japan) | [World Bank CMO](https://www.worldbank.org/en/research/commodity-markets) | 1960–2026 | Benchmark gas alam cair Asia |

### Model 2 — Regresi Daya Beli

| # | Dataset | Sumber | Periode | Peran |
|---|---------|--------|---------|-------|
| 41 | Upah Minimum Provinsi (UMP) | [BPS Jateng](https://jateng.bps.go.id/id/statistics-table/2/MjgyNCMy/upah-minimum-provinsi-ump-per-bulan-menurut-provinsi-di-indonesia.html) | 2021–2025 | Fitur X |
| 42 | Rata-rata Pengeluaran per Kapita | [BPS](https://www.bps.go.id/id/statistics-table/3/V1ZKMWVrSTNOek5ZZUZOcVZEZGFValJvV0hWalFUMDkjMyMwMDAw/rata-rata-pengeluaran-per-kapita-sebulan-makanan-dan-bukan-makanan-di-daerah-perkotaan-dan-perdesaan-menurut-provinsi--rupiah-.html) | 2017–2025 | Target Y |
| 43 | Tingkat Pengangguran Terbuka (Semester) | [Open Data Jabar](https://opendata.jabarprov.go.id/id/dataset/tingkat-pengangguran-terbuka-berdasarkan-semester-dan-provinsi-di-indonesia) | 2020–2025 | Fitur X |
| 44 | TPT & TPAK Menurut Provinsi | [BPS](https://www.bps.go.id/id/statistics-table/3/V2pOVWJWcHJURGg0U2pONFJYaExhVXB0TUhacVFUMDkjMw%3D%3D/tingkat-pengangguran-terbuka--tpt--dan-tingkat-partisipasi-angkatan-kerja--tpak--menurut-provinsi--2019.html) | 2017–2025 | Fitur X |
| 45 | PDRB Per Kapita (Ribu Rupiah) | [BPS](https://www.bps.go.id/id/statistics-table/2/Mjg4IzI=/-seri-2010--produk-domestik-regional-bruto-per-kapita--ribu-rupiah-.html) | 2010–2025 | Fitur X |
| 46 | Persentase Penduduk Miskin per Provinsi | [Open Data Jabar](https://opendata.jabarprov.go.id/id/dataset/persentase-penduduk-miskin-berdasarkan-provinsi-di-indonesia) | 2010–2024 | Fitur X |

---

## Model Machine Learning

### Model 1 — Forecasting Inflasi (LSTM)

| Aspek | Detail |
|-------|-------|
| **Arsitektur** | LSTM, 2 layer, 64 hidden units, dropout 0.2 |
| **Window** | 12 bulan (sequence length) |
| **Jumlah Fitur** | 44 fitur |
| **Split** | Chronological: 70% Train, 15% Val, 15% Test |
| **Scaler** | MinMaxScaler, fit hanya pada Train |
| **Metrik** | MAE, RMSE |

### Model 2 — Dampak Inflasi terhadap Daya Beli (Ridge Regression)

| Aspek | Detail |
|-------|--------|
| **Model** | Ridge Regression (alpha=1.0) |
| **Fitur** | Real_UMP, TPT, PDRB_HargaKonstan, Inflasi_Rata_Tahunan, Provinsi (one-hot) |
| **Split** | Chronological: Train (≤2023), Test (≥2024) |
| **Metrik** | R², MAE, RMSE |

---

## Preprocessing & Data Pipeline

1. **`preprocessing.py`** — Membersihkan dan menggabungkan 46 dataset menjadi `clean_inflasi_ts.csv` (45 kolom) dan `clean_daya_beli.csv` (12 kolom).
2. **`download_international.py`** — Mengunduh 5 dataset internasional secara otomatis dari Yahoo Finance dan FRED.

---

## Cara Menjalankan

```bash
# 1. Install dependensi
pip install -r requirements.txt

# 2. Unduh dataset internasional (Yahoo Finance, FRED)
python download_international.py

# 3. Jalankan preprocessing
python preprocessing.py

# 4. Buka notebook untuk eksperimen model
jupyter notebook notebooks/forecasting_inflasi_models.ipynb
jupyter notebook notebooks/analisis_daya_beli_regresi.ipynb
```

> **Catatan**: Gunakan `$env:PYTHONIOENCODING='utf-8'; python preprocessing.py` pada Windows jika terjadi error encoding.

---

## Data Historis 2026

Data telah diperbarui hingga **April 2026** (Mei 2026 menunggu rilis resmi BPS):

| Bulan | Inflasi MoM | Sumber |
|-------|-------------|--------|
| Januari 2026 | -0.15% | BPS |
| Februari 2026 | 0.68% | BPS |
| Maret 2026 | 0.41% | Inflasi_Umum_MoM |
| April 2026 | 0.13% | Inflasi_Umum_MoM |

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

- [Badan Pusat Statistik (BPS)](https://www.bps.go.id)
- [Bank Indonesia](https://www.bi.go.id)
- [Open Data Jabar](https://opendata.jabarprov.go.id)
- [Investing.com](https://id.investing.com)
- [IndexMundi](https://www.indexmundi.com)
- [Yahoo Finance](https://finance.yahoo.com)
- [FRED](https://fred.stlouisfed.org)
- [Geopolitical Risk Index](https://www.policyuncertainty.com/gpr.html)
- [FAO Food Price Index](https://www.fao.org/worldfoodsituation/foodpricesindex/en/)
- [World Bank Commodity Markets](https://www.worldbank.org/en/research/commodity-markets)
