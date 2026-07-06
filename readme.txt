================================================================================
                        PANDUAN PENGGUNAAN DAN INSTALASI
                                PROYEK ECODASH
        (Sistem Intelijen Ekonomi & Dashboard Analitik Multi-Horizon)
================================================================================

Mata Kuliah : Machine Learning (SD-A1)
Kelompok    : Kelompok E
Program     : S1 Teknologi Sains Data, Universitas Airlangga

Daftar Anggota Kelompok:
1. Muhammad Rajif Al Farikhi      (162112133008) - Backend Developer
2. Sahrul Adicandra Effendy       (164231013)    - Backend & Data Pipeline
3. Semaya David Petroes Putra     (164231048)    - Machine Learning Modelling
4. Adrina Firda Marwah            (164231087)    - Machine Learning Modelling
5. Okan Athallah Maredith         (164231088)    - Frontend & UI/UX Designer

================================================================================
A. PERSYARATAN SISTEM (SYSTEM REQUIREMENTS)
================================================================================
Sebelum menjalankan program, pastikan komputer/laptop telah memenuhi kriteria:
1. Versi Python      : Python 3.10, 3.11, atau 3.12 (Sangat disarankan Python 3.12 64-bit).
                       *Harap hindari Python 3.13 jika ada library yang belum mendukung.
2. Sistem Operasi    : Windows 10/11, macOS, atau Linux.
3. Koneksi Internet  : Diperlukan saat instalasi library (pip) dan saat pertama kali
                       sistem menarik data kurs USD/IDR harian secara live.

================================================================================
B. DAFTAR PUSTAKA (REQUIRED LIBRARIES)
================================================================================
Seluruh pustaka yang dibutuhkan proyek telah dirangkum dalam file 'requirements.txt'.
Secara garis besar, teknologi dan pustaka yang digunakan meliputi:

1. Web Framework & API Delivery:
   - django               : Server aplikasi web dan pengelola API endpoint.

2. Pemodelan Machine Learning, Deep Learning, & Statistik:
   - scikit-learn         : Regresi Ridge/Lasso, standardisasi skala, dan evaluasi.
   - statsmodels          : Pemodelan deret waktu klasik (ARIMA & SARIMAX).
   - prophet              : Pemodelan tren musiman aditif (Facebook Prophet).
   - torch                : PyTorch untuk pemodelan Deep Learning (LSTM & Bi-LSTM).
   - arch                 : Pemodelan volatilitas bersyarat (GARCH).
   - xgboost              : Kandidat algoritma Extreme Gradient Boosting.
   - linearmodels         : Estimator regresi data panel ekonometrika.

3. Manipulasi Data & Integrasi Eksternal:
   - pandas & numpy       : Pengolahan struktur data tabular dan komputasi numerik.
   - openpyxl             : Pengolahan berkas spreadsheet Excel (.xlsx).
   - joblib               : Serialisasi dan penyimpanan artefak model (.pkl).
   - yfinance             : Penarikan data pasar valas harian secara real-time.
   - fredapi & wbdata     : Integrasi indikator makro dari Federal Reserve & World Bank.

4. Eksplorasi & Visualisasi Data:
   - matplotlib & seaborn : Pembuatan grafik analisis diagnostik dan evaluasi model.

================================================================================
C. PANDUAN INSTALASI & PERSIAPAN LINGKUNGAN KERJA (SETUP)
================================================================================
Berikut adalah tata cara instalasi dari file ZIP yang telah diekstrak:

1. Ekstraksi Folder Proyek:
   Ekstrak file ZIP proyek ini ke folder lokal komputer Anda.

2. Membuka Terminal / Command Prompt di Folder Proyek:
   - Cara Mudah (Windows File Explorer):
     Buka folder hasil ekstraksi proyek ini. Klik kanan pada area kosong di dalam
     folder tersebut, lalu pilih "Open in Terminal" atau "Open PowerShell window here".
   - Cara Manual (CMD/Terminal/PowerShell):
     Buka terminal, lalu navigasikan (cd) menuju lokasi folder proyek yang diekstrak.
     Contoh:
     cd "C:\Lokasi\Folder\Ekstraksi\Project-Machine-Learning"
     (Sesuaikan jalur/path dengan lokasi folder di komputer Anda).

3. Buat Virtual Environment (Sangat Disarankan):
   Untuk mencegah bentrok versi library dengan sistem komputer, buat lingkungan
   virtual Python baru dengan perintah:
   
   python -m venv venv

4. Aktifkan Virtual Environment:
   - Pada Windows (PowerShell) : .\venv\Scripts\Activate.ps1
   - Pada Windows (CMD)        : .\venv\Scripts\activate.bat
   - Pada macOS / Linux        : source venv/bin/activate
   
   *(Jika berhasil, akan muncul tulisan "(venv)" di bagian kiri terminal).*

5. Instal Seluruh Dependencies:
   Jalankan perintah berikut untuk mengunduh dan memasang semua library:
   
   pip install -r requirements.txt

================================================================================
D. CARA MENJALANKAN APLIKASI WEB DASHBOARD (QUICK START)
================================================================================
Catatan Penting: Seluruh model terbaik (ARIMA, SARIMAX, Prophet, LSTM, Bi-LSTM,
Ensemble, dan Ridge) SUDAH DILATIH dan disimpan di dalam folder '/models'.
Anda TIDAK PERLU menunggu proses training model untuk melihat hasil dashboard.

1. Pastikan Anda berada di root folder proyek dan virtual environment sudah aktif.
2. Jalankan server lokal Django dengan perintah:

   python dashboard/manage.py runserver

   *(Jika menggunakan macOS/Linux, gunakan perintah: python3 dashboard/manage.py runserver)*

3. Buka browser internet (Google Chrome, Microsoft Edge, atau Mozilla Firefox).
4. Akses alamat web berikut di kolom URL browser:

   http://127.0.0.1:8000/
   
   atau
   
   http://localhost:8000/

5. Menu Eksplorasi Utama di Dashboard:
   - Home          : Ringkasan narasi eksekutif dan indikator makro utama.
   - Dashboard     : Panel pemantauan inflasi bulanan dan kurs valas live.
   - Forecasting   : Evaluasi turnamen model multi-horizon (1m, 3m, 6m, 12m) beserta
                     band interval kepercayaan (confidence intervals).
   - Daya Beli     : Simulasi interaktif pengaruh UMP, PDRB, IPM, dan Inflasi
                     terhadap pengeluaran riil per kapita menggunakan Ridge Regression.
   - Map           : Visualisasi peta geospasial indikator ekonomi provinsi di Indonesia.
   - Panduan       : Dokumentasi metodologi, interpretasi metrik, dan glosarium.

================================================================================
E. TATA CARA PELATIHAN ULANG MODEL / RETRAIN PIPELINE (OPSIONAL)
================================================================================
Bagian ini OPSIONAL dan hanya perlu dijalankan apabila Anda ingin menguji ulang
pipeline pemrosesan data mentah dan melatih ulang algoritma machine learning dari awal.

1. Jalankan Pembersihan & Harmonisasi Data Mentah (Preprocessing):
   python preprocessing.py
   
2. Latih Ulang & Evaluasi Model Proksi Daya Beli (Model 2 - Ridge Regression):
   python dashboard/train_daya_beli_ridge.py
   
3. Latih Ulang Turnamen Model Prediksi Inflasi Multi-Horizon (Model 1):
   python dashboard/train_inflation_multihorizon.py

*(Proses no. 3 akan memakan waktu sekitar 1-2 menit karena melatih dan mengevaluasi
model ARIMA, SARIMAX, GARCH, Prophet, LSTM, Bi-LSTM, dan Ensemble secara walk-forward).*

================================================================================
F. PENGUJIAN OTOMATIS (UNIT TESTING)
================================================================================
Untuk memastikan seluruh fungsi kalkulasi, prediktor model, dan endpoint API backend
berfungsi tanpa kendala, jalankan perintah unit testing Django:

   python dashboard/manage.py test predictions.tests

================================================================================
G. STRUKTUR DIREKTORI PROYEK
================================================================================
├── dashboard/                     # Kode sumber aplikasi web Django
│   ├── manage.py                  # Skrip utilitas Django
│   ├── predictions/               # Modul utama (views, API endpoints, tests)
│   │   ├── templates/predictions/ # Antarmuka HTML/CSS/JS bergaya profesional
│   │   └── static/predictions/    # Aset gambar dan styling statis
│   ├── train_inflation_multihorizon.py # Skrip latih & evaluasi Model 1
│   └── train_daya_beli_ridge.py        # Skrip latih & evaluasi Model 2
├── datasets/                      # Direktori penyimpanan data
│   ├── raw/                       # Data mentah resmi dari BPS & BI
│   └── processed/                 # Data terharmonisasi siap pakai (CSV)
├── models/                        # Artefak model terlatih (.pkl & .json)
├── preprocessing.py               # Pipeline pembersihan data mentah
├── requirements.txt               # Daftar dependensi library Python
└── readme.txt                     # Dokumentasi panduan penggunaan (berkas ini)

================================================================================
               Terima kasih atas perhatian dan evaluasi yang diberikan.
================================================================================
