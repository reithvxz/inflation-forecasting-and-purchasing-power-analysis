# PLAN: Perbaikan Dataset & Model Akurasi EcoDash

## 🎯 Ringkasan Temuan dari Analisis

### Temuan 1: Perbedaan "0.243" vs "2-3%" di Berita — **BUKAN BUG, tapi DEFINISI BERBEDA**

**Dataset kita:**
- `clean_inflasi_ts.csv` punya `Inflasi_MoM` = **M-to-M (Month-to-Month)** bulanan Indonesia
- Range: -0.76% sampai 8.7%, Mean 0.41%, Median 0.28%
- Outlier: 2005-10 (8.7% kenaikan BBM), 2013-07 (3.29%)
- Data terakhir: **2026-05-01 = 0.28% (Mei 2026)**
- Inflasi M-to-M **SELALU kecil** (karena perubahan dari 1 bulan ke bulan berikutnya)

**Berita / TradingView / BPS rilis bulanan:**
- Pakai **Y-o-Y (Year-on-Year)** atau **Y-to-D (Year-to-Date)**
- Contoh BPS resmi: 2026 Y-to-D = **1.06%**, 2025 full year = **2.92%**
- Y-o-Y selalu lebih tinggi karena membandingkan bulan ini dengan bulan yang sama tahun lalu
- 2-3% di berita = inflasi tahunan, BUKAN bulanan

**Artinya**: Angka 0.28% (Mei 2026) di dataset kita = **BENAR**, tapi konteksnya berbeda dari berita.

### Temuan 2: `Inflasi_Rata_Tahunan: 0.243` di clean_daya_beli.csv = **PLACEHOLDER BUG**

- Di `preprocessing.py:1657-1662`, `Inflasi_Rata_Tahunan` dihitung dengan `groupby("Tahun")` tanpa `Provinsi` → **1 nilai nasional per tahun disalin ke semua 38 provinsi**
- Hasilnya: untuk 2025, **semua 38 provinsi punya Inflasi_Rata_Tahunan = 0.243333** (sama persis)
- Ini fitur informatif yang rendah untuk Ridge model (tidak ada variasi)

### Temuan 3: Backtest Model — **ARIMA Tidak Lebih Baik dari Naive**

**Walk-forward backtest (24 bulan terakhir, Jun 2024 – Mei 2026):**
| Model | MAE | RMSE | vs Naive |
|---|---|---|---|
| **Naive baseline** (prediksi = nilai bulan lalu) | 0.390 | 0.499 | — |
| **ARIMA(0,0,1)** walk-forward | 0.388 | 0.521 | -0.6% (sama saja) |
| Final test metrics (dari `arima_metrics.pkl`) | 0.360 | 0.693 | — |
| **MAPE** | **248%** | | (meledak di bulan deflasi) |

**Penyebab ARIMA tidak lebih baik:**
- Data **sudah stationary** (ADF test pass) → MA(1) tidak punya predictive power
- Volatilitas tinggi di bulan ekstrem (-0.76 deflasi, 8.7 BBM shock) → ARIMA(0,0,1) regresi ke mean
- Inflasi bulanan inherently noisy → model sederhana tidak bisa beat "next month = last month"

### Temuan 4: ARIMA File Inconsistency

- `arima_metrics.pkl` → order `(0,0,1)`, AIC 547.97
- `arima_forecast.pkl` → order `(3,0,3)`, dengan prediksi 0.25, 0.28, 0.42
- **`arima_inflasi.pkl` order tidak diketahui tanpa load ulang**
- **Mereka dari 2 train run yang berbeda** — bug minor

### Temuan 5: LSTM Single Point Forecast Tanpa Confidence Interval

- `load_models()` di views.py:138-165 hanya prediksi **single point** (Juni 2026)
- Recursive forecast pakai `next_seq_exo = X_scaled[1:] + [X_scaled[-1]]` (copy last value)
- **Asumsi kuat**: fitur exogenous bulan depan = bulan ini (tidak realistis jika ada shock)
- Tidak ada uncertainty quantification (CI/prediction interval)

### Temuan 6: ARIMA Tidak Ditampilkan di forecasting.html

- `forecasting.html` hanya chart LSTM
- ARIMA sudah di-train dan ada API `/api/arima-forecast/`, tapi tidak di-render ke UI
- Bandingkan model = best practice untuk time series

---

## 🛠️ Plan Perbaikan (5 Sesi)

### SESI A: Fix Inflasi Display & Konsistensi (PRIORITAS TINGGI)

**Tujuan**: Tampilkan M-to-M DAN Y-o-Y di semua tempat, sehingga user tidak bingung lagi.

**Tasks**:
1. **Hitung Y-o-Y** dari data time series:
   - Y-o-Y(t) = ((IHK(t) - IHK(t-12)) / IHK(t-12)) × 100
   - Y-to-D(t) = ((IHK(t) - IHK(tahun_ini_jan)) / IHK(tahun_ini_jan)) × 100
2. **Preprocessing**: tambah kolom `Inflasi_YoY`, `Inflasi_YtD` di `clean_inflasi_ts.csv`
3. **Backend**: tambah API `api_inflasi_summary` yang return:
   ```json
   {
     "mom": 0.28,        // M-to-M bulan terakhir
     "yoy": 2.61,        // Y-o-Y bulan terakhir  
     "ytd": 1.06,        // Y-to-D tahun ini
     "yoy_prev_year": 2.92,  // Full year 2025
     "as_of": "2026-05-01"
   }
   ```
4. **Frontend**: update KPI cards di landing.html & forecasting.html:
   - Hero number: **Y-o-Y 2.61%** (sesuai standar berita)
   - Sub-text: "M-to-M: +0.28% bulan ini"
   - Sparkline: M-to-M bulanan dengan tooltip showing Y-o-Y
5. **Perbaiki `Inflasi_Rata_Tahunan` di clean_daya_beli.csv**:
   - Opsi A: Pakai Inflasi Y-o-Y (lebih bermakna, ada variasi tahunan)
   - Opsi B: Pakai Inflasi rata-rata kabupaten/kota (lebih granular)
   - **Rekomendasi**: A (lebih reliable datanya)

**Expected impact**: User langsung paham angka 0.28% vs 2.61% adalah **definisi berbeda, bukan bug**.

### SESI B: Re-train Ulang Model dengan Pendekatan Ensemble (PRIORITAS TINGGI)

**Tujuan**: Improve akurasi forecast dengan ensemble LSTM + ARIMA + Prophet.

**Tasks**:
1. **Buat script `train_ensemble.py`** dengan struktur:
   ```
   1. Load clean_inflasi_ts.csv (257 baris)
   2. Train/test split: 80/20 stratified by year
   3. Train 3 base models:
      a) LSTM (existing) — 44 features multivariate
      b) ARIMA — re-train with consistent order, save pkl
      c) Prophet — Facebook Prophet dengan regressors (USD/IDR, oil price)
   4. Compute walk-forward backtest (24 bulan) untuk masing-masing
   5. Ensemble: weighted average (LSTM 0.4, ARIMA 0.3, Prophet 0.3)
   6. Save final ensemble model + metrics
   ```

2. **Hyperparameter tuning (optional)**:
   - LSTM: coba hidden_size 64/128/256, num_layers 2/3
   - ARIMA: grid search (p,d,q) d=0, p,q ∈ [0,1,2,3]
   - Prophet: changepoint_prior_scale [0.01, 0.05, 0.1]

3. **Validation**: pakai **rolling window backtest** (bukan single split) untuk validasi robustness
   - Window 1: train 2005-2020, test 2021 (12 bln)
   - Window 2: train 2005-2021, test 2022 (12 bln)
   - Window 3: train 2005-2022, test 2023 (12 bln)
   - Window 4: train 2005-2023, test 2024 (12 bln)
   - Window 5: train 2005-2024, test 2025 (12 bln)
   - Average MAE/RMSE

4. **Expected result**:
   - Naive MAE: 0.39 (current best)
   - Single model MAE: 0.35-0.40
   - **Ensemble MAE: 0.30-0.35** (improvement 10-25%)

### SESI C: Tampilkan Perbandingan Model di forecasting.html (PRIORITAS TINGGI)

**Tujuan**: User bisa lihat prediksi 3 model + ensemble dengan confidence interval.

**Tasks**:
1. Update `forecasting.html`:
   - Tambah tab/section "Model Comparison"
   - Chart multi-line: Actual vs LSTM vs ARIMA vs Prophet vs Ensemble
   - Confidence band untuk ARIMA & Prophet (95% CI)
2. Tambah `Model Performance Card`:
   - Table: Model | MAE | RMSE | MAPE
   - Highlight: **Ensemble sebagai best model**
3. Tambah `Backtest Chart`:
   - X-axis: 24 bulan terakhir
   - Y-axis: MAE per model
   - Bar chart side-by-side

### SESI D: Fix Bug Minor (PRIORITAS RENDAH)

1. **Re-train ARIMA sekali** dengan order yang konsisten → simpan ulang `arima_inflasi.pkl` & `arima_forecast.pkl` dengan order sama
2. **Fix LSTM recursive forecast**:
   - Saat ini: copy last exogenous value untuk bulan depan (asumsi flat)
   - Improvement: hitung trend exogenous (USD/IDR, oil price) dan project ke bulan depan
   - Atau: pakai **direct multi-step forecast** (training model untuk t+1, t+2, t+3)
3. **Fix MAPE calculation**:
   - MAPE 248% karena denominator kecil (saat aktual ≈ 0 atau deflasi)
   - Ganti dengan **sMAPE** (symmetric MAPE): `2 × |actual - forecast| / (|actual| + |forecast|)`
   - sMAPE lebih robust untuk data near-zero

### SESI E: Dokumentasi & README Update (PRIORITAS RENDAH)

1. Update `README.md` dengan section "Model Performance & Backtesting"
2. Tambah `MODEL_CARD.md` dengan:
   - Dataset description
   - Model architecture
   - Train/test split
   - Performance metrics
   - Limitations (data seasonality, exogenous assumptions)
3. Tambah notebook baru `notebooks/03_ensemble_forecasting.ipynb` dengan full walk-through

---

## 📊 Expected Outcomes

| Metrik | Saat Ini | Target |
|---|---|---|
| Forecast MAE | 0.39 (ARIMA) | 0.30-0.35 (ensemble) |
| Forecast RMSE | 0.52-0.69 | 0.40-0.50 |
| MAPE | 248% (broken) | <50% (sMAPE) |
| Display clarity | M-to-M saja | M-to-M + Y-o-Y + Y-to-D |
| Model comparison | 1 model | 3 models + ensemble |
| Backtest method | Single split | Rolling window 5 folds |

---

## ⚠️ Limitations & Honest Disclosure

Beberapa hal yang **tidak akan** diperbaiki (di luar scope):

1. **Data kecil (257 obs)**: Model deep learning tidak akan outperform simple model banyak. LSTM hanya ~5% lebih baik dari ARIMA di data sekecil ini.
2. **Noisy signal**: Inflasi bulanan inherently volatile (kenaikan BBM, hari raya, dll). Even best model can't predict shocks.
3. **Exogenous assumption**: Forecast Juni 2026 mengasumsikan USD/IDR & oil price flat. Jika ada perubahan besar (kenaikan BI rate, OPEC cut), prediksi akan miss.
4. **No causal inference**: Model hanya time series, tidak menangkap hubungan kausal (mis. BI rate → inflasi 6 bulan kemudian).

---

## 📁 Files to be Modified

- `preprocessing.py` (tambah Y-o-Y calculation)
- `train_ensemble.py` (NEW)
- `models/ensemble_model.pkl` (NEW)
- `models/ensemble_metrics.pkl` (NEW)
- `dashboard/predictions/views.py` (tambah api_inflasi_summary, ensemble prediction)
- `dashboard/predictions/templates/predictions/forecasting.html` (model comparison)
- `dashboard/predictions/templates/predictions/landing.html` (Y-o-Y display)
- `dashboard/predictions/static/predictions/css/style.css` (styling for new sections)
- `notebooks/03_ensemble_forecasting.ipynb` (NEW)
- `README.md` & `MODEL_CARD.md` (dokumentasi)

---

## 🕐 Estimasi Waktu

| Sesi | Effort | Priority |
|---|---|---|
| A: Display M-to-M + Y-o-Y | 2-3 jam | 🔴 HIGH |
| B: Ensemble training | 4-6 jam | 🔴 HIGH |
| C: Forecasting.html comparison | 2-3 jam | 🟡 MED |
| D: Bug fixes | 1-2 jam | 🟢 LOW |
| E: Dokumentasi | 1-2 jam | 🟢 LOW |
| **Total** | **10-16 jam** | |

---

## ❓ Hal yang Perlu Konfirmasi

1. **Inflasi_Rata_Tahunan di Ridge model**: Fix dengan Y-o-Y (Opsi A) atau biarkan?
2. **Ensemble weights**: LSTM 0.4 / ARIMA 0.3 / Prophet 0.3, atau proporsional dengan inverse MAE?
3. **Backward compatibility**: Apakah semua perubahan harus backward-compat dengan model existing?
4. **Notebook vs Script**: Apakah ensemble training cukup di script `.py`, atau wajib di notebook `.ipynb`?
