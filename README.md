# 📊 Prediksi Inflasi dan Dampaknya terhadap Daya Beli

> **Kelompok E – Machine Learning SD-A1, Universitas Airlangga**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Django](https://img.shields.io/badge/Framework-Django-green?logo=django)](https://djangoproject.com)
[![License](https://img.shields.io/badge/License-Academic-lightgrey)](LICENSE)

---

## 🎯 Deskripsi Proyek

Proyek ini membangun sistem prediksi inflasi dan analisis dampaknya terhadap daya beli masyarakat Indonesia. Terdapat dua model utama:

1. **Forecasting (LSTM)** — Memprediksi nilai inflasi bulanan ke depan berdasarkan data historis.
2. **Regresi (Random Forest / Linear Regression)** — Mengukur pengaruh inflasi terhadap daya beli masyarakat (pengeluaran per kapita).

Output disajikan melalui **Dashboard Web (Django)** yang menampilkan grafik prediksi dan fitur simulasi daya beli.

---

## 🗂️ Struktur Proyek

```
Project-Machine-Learning/
├── datasets/
│   ├── BI Rate (Data Inflasi)/
│   ├── Data Historis USD_IDR/
│   ├── Harga Bulanan Minyak Mentah (minyak bumi) - Dolar AS per Barel/
│   ├── Indeks Harga Konsumen (Umum)/
│   ├── Inflasi Bulanan/
│   ├── Inflasi Umum, Inti, Harga Diatur Pemerintah, dan Bergejolak Nasional (M-to-M dan Y-to-D)/
│   ├── Persentase Penduduk Miskin Berdasarkan Provinsi di Indonesia/
│   ├── Produk Domestik Regional Bruto Per Kapita (Ribu Rupiah)/
│   ├── Rata-rata Pengeluaran per Kapita Sebulan Makanan dan Bukan Makanan/
│   ├── Tingkat Pengangguran Terbuka (TPT) dan Tingkat Partisipasi Angkatan Kerja (TPAK) Menurut Provinsi/
│   ├── Tingkat Pengangguran Terbuka Berdasarkan Semester dan Provinsi di Indonesia/
│   ├── Upah Minimum Provinsi/
│   └── processed/
│       ├── clean_inflasi_ts.csv   ← time-series bulanan untuk LSTM
│       └── clean_daya_beli.csv    ← panel provinsi untuk Regresi
├── dashboard/                     ← Django project
│   └── predictions/               ← Django app
├── explore_datasets.py            ← eksplorasi & visualisasi awal
├── preprocessing.py               ← pipeline join dataset (menghasilkan clean_*.csv)
├── data_pipeline.py               ← ANTI-LEAKAGE PIPELINE (split, scale, log, lag)
├── requirements.txt
└── README.md
```

---

## 📦 Dataset

| # | Dataset | Sumber | Periode | Model | Peran |
|---|---------|--------|---------|-------|-------|
| 1 | **Indeks Harga Konsumen (IHK)** | [BPS](https://www.bps.go.id/id/statistics-table/2/MiMy/indeks-harga-konsumen--umum-.html) | 2005–2023 | Model 1 (LSTM) | Fitur X |
| 2 | **Inflasi Bulanan (M-to-M)** | [BPS](https://www.bps.go.id/id/statistics-table/2/MSMy/inflasi--umum-.html) | 2005–2026 | Model 1 (LSTM) | **Target Y** |
| 3 | **Inflasi Tahun Kalender (Y-to-D)** | [BPS](https://www.bps.go.id/id/statistics-table/1/OTE0IzE=/tingkat-inflasi-harga-konsumen-nasional-tahun-kalender--y-to-d---sup-1--sup---2022-100-.html) | Historis | — | Referensi |
| 4 | **BI Rate / Data Inflasi** | [Bank Indonesia](https://www.bi.go.id/id/statistik/indikator/data-inflasi.aspx) | 2005–2026 | Model 1 (LSTM) | Fitur X |
| 5 | **Upah Minimum Provinsi (UMP)** | [BPS Jateng](https://jateng.bps.go.id/id/statistics-table/2/MjgyNCMy/upah-minimum-provinsi-ump-per-bulan-menurut-provinsi-di-indonesia.html) | 2021–2025 | Model 2 (Regresi) | Fitur X |
| 6 | **Rata-rata Pengeluaran per Kapita** | [BPS](https://www.bps.go.id/id/statistics-table/3/V1ZKMWVrSTNOek5ZZUZOcVZEZGFValJvV0hWalFUMDkjMyMwMDAw) | 2017–2025 | Model 2 (Regresi) | **Target Y** |
| 7 | **Kurs USD/IDR Historis** | [Investing.com](https://id.investing.com/currencies/usd-idr-historical-data) | 2005–2025 | Model 1 (LSTM) | Fitur X |
| 8 | **Tingkat Pengangguran Terbuka (Semester)** | [Open Data Jabar](https://opendata.jabarprov.go.id/id/dataset/tingkat-pengangguran-terbuka-berdasarkan-semester-dan-provinsi-di-indonesia) | 2020–2025 | Model 2 (Regresi) | Fitur X |
| 9 | **TPT & TPAK Menurut Provinsi** | [BPS](https://www.bps.go.id/id/statistics-table/3/V2pOVWJWcHJURGg0U2pONFJYaExhVXB0TUhacVFUMDkjMw%3D%3D/tingkat-pengangguran-terbuka--tpt--dan-tingkat-partisipasi-angkatan-kerja--tpak--menurut-provinsi--2019.html) | 2017–2025 | Model 2 (Regresi) | Fitur X |
| 10 | **PDRB Per Kapita (Ribu Rupiah)** | [BPS](https://www.bps.go.id/id/statistics-table/2/Mjg4IzI=/-seri-2010--produk-domestik-regional-bruto-per-kapita--ribu-rupiah-.html) | 2010–2025 | Model 2 (Regresi) | Fitur X |
| 11 | **Persentase Penduduk Miskin per Provinsi** | [Open Data Jabar](https://opendata.jabarprov.go.id/id/dataset/persentase-penduduk-miskin-berdasarkan-provinsi-di-indonesia) | 2010–2024 | Model 2 (Regresi) | Fitur X |
| 12 | **Inflasi Umum, Inti, Harga Diatur, Bergejolak** | [BPS](https://www.bps.go.id/id/statistics-table/1/OTA4IzE=/inflasi-umum--inti--harga-yang-diatur-pemerintah--dan-barang-bergejolak-inflasi-indonesia--2009-2025.html) | 2009–2026 | Model 1 (LSTM) | Fitur X |
| 13 | **Harga Bulanan Minyak Mentah (USD/Barel)** | [IndexMundi](https://www.indexmundi.com/commodities/?commodity=crude-oil&months=300) | 2001–2026 | Model 1 (LSTM) | Fitur X |

---

## 📈 Visualisasi Dataset

![Dashboard Analisis Dataset](datasets/visualisasi_dataset.png)

---

## 🔧 Preprocessing & Data Pipeline (Anti-Leakage)

Proses pengolahan data dibagi menjadi dua tahapan ketat untuk **mencegah Data Leakage** dari *testing set* ke *training set*:

1. **`preprocessing.py`**: Hanya melakukan pembersihan teks dan penggabungan secara waktu (join).
2. **`data_pipeline.py`**: Melakukan Train/Val/Test Split *TERLEBIH DAHULU*, kemudian melakukan *Scaling*, Interpolasi, Log Transform, dan pembuatan fitur *Lag/Windows*.

---

### Output 1 — `datasets/processed/clean_inflasi_ts.csv` (Raw untuk LSTM)

**Alur `preprocessing.py`:**
```text
Inflasi Bulanan (22 file CSV, 2005–2026)
  -> Parse tanggal bahasa Indonesia
  -> Gabungkan jadi 1 kolom: [Tanggal, Inflasi_MoM]
  -> Join IHK (NaN untuk data setelah 2019)
  -> Join BI Rate (bulanan)
  -> Join USD/IDR (bulanan)
  -> Join Inflasi Komponen: Inti, Harga Diatur, Bergejolak (2009–2026)
  -> Join Harga Minyak Mentah USD/Barel (2001–2026)
  -> Tambah kolom Bulan dan Tahun
```
*(Catatan: Fitur lag 1-12 dan scaling akan digenerate otomatis di memori oleh `data_pipeline.py` spesifik pada data Train untuk mencegah leakage)*

| Kolom | Keterangan |
|-------|-----------|
| `Tanggal` | Periode bulanan (2005–2026) |
| `Inflasi_MoM` | Target prediksi (%) |
| `IHK` | Indeks harga konsumen (NaN setelah 2019) |
| `BI_Rate` | Suku bunga acuan BI (%) |
| `USD_IDR` | Kurs dolar–rupiah rata-rata bulanan (Rp) |
| `Inflasi_Umum_MoM` | Inflasi umum MoM — komponen BPS (2009–) |
| `Inflasi_Inti_MoM` | Inflasi inti MoM (2009–) |
| `Inflasi_HargaDiatur_MoM` | Inflasi harga diatur pemerintah MoM (2009–) |
| `Inflasi_Bergejolak_MoM` | Inflasi bergejolak MoM (2009–) |
| `Harga_Minyak_USD` | Harga minyak mentah (USD/Barel) |
| `Bulan`, `Tahun` | Fitur siklus waktu |

---

### Output 2 — `datasets/processed/clean_daya_beli.csv` (Raw untuk Regresi)

**Alur `preprocessing.py`:**
```text
Pengeluaran per Kapita (per provinsi, 2017–2025)
  -> Join UMP per provinsi (2021–2025)
  -> Join TPT per provinsi (BPS 2017–2025, fallback Open Data Jabar 2020–2025)
  -> Join TPAK per provinsi (BPS 2017–2025)
  -> Join PDRB per kapita per provinsi (2010–2025)
  -> Join Persentase Penduduk Miskin per provinsi (2010–2024)
  -> Join Inflasi rata-rata tahunan (dari inflasi bulanan)
  -> Filter tahun overlap: 2021–2025
```

| Kolom | Keterangan |
|-------|-----------|
| `Provinsi` | 38 provinsi Indonesia |
| `Tahun` | 2021–2025 |
| `Pengeluaran_Makanan` | Pengeluaran per kapita makanan (Rp/bulan) |
| `Pengeluaran_Bukan_Makanan` | Pengeluaran per kapita bukan makanan (Rp/bulan) |
| `Total_Pengeluaran` | Total pengeluaran per kapita (Rp/bulan) — **Target Y** |
| `UMP` | Upah minimum (Rp/bulan) |
| `TPT` | Tingkat Pengangguran Terbuka (%) |
| `TPAK` | Tingkat Partisipasi Angkatan Kerja (%) |
| `PDRB_HargaBerlaku` | PDRB per kapita harga berlaku (Ribu Rp) |
| `PDRB_HargaKonstan` | PDRB per kapita harga konstan 2010 (Ribu Rp) |
| `Pct_Penduduk_Miskin` | Persentase penduduk miskin (%) |
| `Inflasi_Rata_Tahunan` | Rata-rata inflasi MoM per tahun (%) |

---

## 🤖 Model Machine Learning

### Model 1 – Forecasting Inflasi (LSTM)
- **Input**: *Windowing sequences* 12 bulanan (`clean_inflasi_ts.csv` diproses oleh `data_pipeline.py`)
- **Output**: Prediksi inflasi bulan berikutnya
- **Metrik**: MAE, RMSE

### Model 2 – Dampak Inflasi terhadap Daya Beli (Regresi)
- **Input**: Panel data provinsi (`clean_daya_beli.csv` diproses oleh `data_pipeline.py`)
- **Output**: Estimasi pengeluaran per kapita berdasarkan inflasi & variabel ekonomi lainnya
- **Metrik**: R², MSE, koefisien regresi

---

## 🚀 Cara Menjalankan

```bash
# 1. Install dependensi
pip install -r requirements.txt

# 2. Eksplorasi dataset (opsional)
python explore_datasets.py

# 3. Jalankan preprocessing (Menghasilkan clean_*.csv)
python preprocessing.py

# 4. Tes Split Pipeline AI
python data_pipeline.py

# 5. Jalankan web dashboard
cd dashboard
python manage.py runserver
```

> **Catatan Windows**: Jalankan preprocessing dengan `$env:PYTHONIOENCODING='utf-8'; python preprocessing.py` jika ada error encoding.

---

## 👥 Anggota Kelompok E

| Nama | NIM |
|------|-----|
| Muhammad Rajif Al Farikhi | 162112133008 |
| Sahrul Adicandra Effendy | 164231013 |
| Semaya David Petroes Putra | 164231048 |
| Adrina Firda Marwah | 164231087 |
| Okan Athallah Maredith | 164231088 |

---

## 📚 Referensi Data
- Badan Pusat Statistik (BPS): https://www.bps.go.id
- Bank Indonesia: https://www.bi.go.id
- Open Data Jabar: https://opendata.jabarprov.go.id
- Investing.com: https://id.investing.com
- IndexMundi: https://www.indexmundi.com
