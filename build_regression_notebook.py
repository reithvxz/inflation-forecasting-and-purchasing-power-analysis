import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Markdown: Judul
nb.cells.append(nbf.v4.new_markdown_cell("""# Eksperimen Model Regresi Daya Beli (Model 2)
Notebook ini membandingkan berbagai model Machine Learning (dari Baseline hingga Model Ensemble) untuk memprediksi **Total Pengeluaran per Kapita** (sebagai proksi Daya Beli) per provinsi di Indonesia.

Sesuai instruksi:
- Fitur **TPAK** dan **Pct_Penduduk_Miskin** **dieliminasi** dari pemodelan karena memiliki terlalu banyak *missing values* (masing-masing 38 dan 37 baris kosong).
- Validasi dilakukan secara **Chronological Split** (Train: 2021-2023, Val: 2024, Test: 2025) untuk menghindari data leakage temporal.
- Kita mengimplementasikan beberapa teknik feature engineering tingkat lanjut dan menggunakan model ensambel (XGBoost, LightGBM, CatBoost) untuk memperoleh hasil prediksi yang tangguh.
"""))

# Import
nb.cells.append(nbf.v4.new_code_cell("""import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
"""))

# Load Data
nb.cells.append(nbf.v4.new_markdown_cell("""## 1. Load Data & Analisis Missing Value
Kita load data `clean_daya_beli.csv` yang tersimpan di folder processed."""))

nb.cells.append(nbf.v4.new_code_cell("""# Load dataset
data_path = os.path.join('..', 'datasets', 'processed', 'clean_daya_beli.csv')
df = pd.read_csv(data_path)

print("Informasi Dataset Awal:")
print(df.info())
print("\\nJumlah Missing Value per Kolom:")
print(df.isnull().sum())
"""))

# Drop TPAK & Pct_Penduduk_Miskin
nb.cells.append(nbf.v4.new_markdown_cell("""## 2. Seleksi Fitur & Eliminasi Missing Value
Kita menghapus fitur `TPAK` dan `Pct_Penduduk_Miskin` karena persentase kekosongan data yang tinggi. Fitur target yang diprediksi adalah `Total_Pengeluaran`."""))

nb.cells.append(nbf.v4.new_code_cell("""# Drop kolom TPAK dan Pct_Penduduk_Miskin
df_clean = df.drop(columns=['TPAK', 'Pct_Penduduk_Miskin'])

# Pisahkan juga Pengeluaran_Makanan dan Pengeluaran_Bukan_Makanan agar tidak menjadi leakage (karena jumlahnya pasti sama dengan Total_Pengeluaran)
df_clean = df_clean.drop(columns=['Pengeluaran_Makanan', 'Pengeluaran_Bukan_Makanan'])

print("Missing values setelah pembersihan:")
print(df_clean.isnull().sum())
print("\\nShape data:", df_clean.shape)
df_clean.head()
"""))

# Feature Engineering
nb.cells.append(nbf.v4.new_markdown_cell("""## 3. Advanced Feature Engineering
Kita membuat fitur-fitur baru yang merepresentasikan rasio ekonomi penting:
1. **GDP Deflator proxy**: `PDRB_HargaBerlaku / PDRB_HargaKonstan` (Menunjukkan tingkat harga kumulatif/inflasi regional).
2. **Real UMP**: `UMP / (1 + Inflasi_Rata_Tahunan)` (Menunjukkan daya beli riil upah minimum terhadap inflasi).
3. **PDRB to UMP ratio**: `PDRB_HargaKonstan / UMP` (Indikator produktivitas regional relatif terhadap standar upah minimum).
4. **TPT x UMP interaction**: `TPT * UMP` (Interaksi antara tingkat pengangguran dengan tingkat upah provinsi).
"""))

nb.cells.append(nbf.v4.new_code_cell("""def engineer_features(data):
    df_feat = data.copy()
    # 1. Rasio PDRB Nominal terhadap Riil (GDP Deflator proxy)
    df_feat['GDP_Deflator'] = df_feat['PDRB_HargaBerlaku'] / df_feat['PDRB_HargaKonstan']
    
    # 2. UMP disesuaikan dengan inflasi
    df_feat['Real_UMP'] = df_feat['UMP'] / (1 + df_feat['Inflasi_Rata_Tahunan'])
    
    # 3. Rasio PDRB Riil terhadap UMP
    df_feat['PDRB_to_UMP'] = df_feat['PDRB_HargaKonstan'] / df_feat['UMP']
    
    # 4. Interaksi Pengangguran dengan UMP
    df_feat['TPT_x_UMP'] = df_feat['TPT'] * df_feat['UMP']
    
    return df_feat

df_engineered = engineer_features(df_clean)
print("Fitur baru berhasil dibuat. Shape baru:", df_engineered.shape)
df_engineered.head()
"""))

# Chronological Train-Val-Test Split
nb.cells.append(nbf.v4.new_markdown_cell("""## 4. Chronological Train-Val-Test Split
Untuk mencegah temporal data leakage (data masa depan memprediksi masa lalu), kita melakukan split secara kronologis berdasarkan kolom `Tahun`:
- **Train Set**: Tahun 2021 - 2023 (Data historis awal)
- **Validation Set**: Tahun 2024 (Untuk tuning model)
- **Test Set**: Tahun 2025 (Data teruji paling akhir)
"""))

nb.cells.append(nbf.v4.new_code_cell("""# Pisahkan X dan y
target_col = 'Total_Pengeluaran'
features_list = [col for col in df_engineered.columns if col not in [target_col]]

train_mask = df_engineered['Tahun'] <= 2023
val_mask = df_engineered['Tahun'] == 2024
test_mask = df_engineered['Tahun'] == 2025

X_train_raw = df_engineered[train_mask][features_list]
y_train = df_engineered[train_mask][target_col]

X_val_raw = df_engineered[val_mask][features_list]
y_val = df_engineered[val_mask][target_col]

X_test_raw = df_engineered[test_mask][features_list]
y_test = df_engineered[test_mask][target_col]

print(f"Train Set Shape : {X_train_raw.shape}, y: {y_train.shape} (Tahun 2021-2023)")
print(f"Val Set Shape   : {X_val_raw.shape}, y: {y_val.shape} (Tahun 2024)")
print(f"Test Set Shape  : {X_test_raw.shape}, y: {y_test.shape} (Tahun 2025)")
"""))

# Data Preprocessing Pipeline (Scaling & OneHotEncoding)
nb.cells.append(nbf.v4.new_markdown_cell("""## 5. Pipeline Preprocessing (Scaling & Encoding)
Kita menggunakan `ColumnTransformer` untuk memproses tipe data berbeda:
- Fitur Numerik: Di-scale menggunakan `StandardScaler`.
- Fitur Kategorikal (`Provinsi`): Di-encode menggunakan `OneHotEncoder` (untuk model linier) atau label encoded untuk tree models.
"""))

nb.cells.append(nbf.v4.new_code_cell("""num_features = ['Tahun', 'UMP', 'TPT', 'PDRB_HargaBerlaku', 'PDRB_HargaKonstan', 
                'Inflasi_Rata_Tahunan', 'GDP_Deflator', 'Real_UMP', 'PDRB_to_UMP', 'TPT_x_UMP']
cat_features = ['Provinsi']

# Pipeline untuk model Linier (butuh OneHotEncoder)
preprocessor_linear = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ]
)

# Pipeline untuk model Tree-based (bisa pakai ordinal encoding/label encoding atau OneHot)
# Agar konsisten dan handal, kita gunakan OneHot untuk tree-based klasik juga, 
# kecuali CatBoost yang bisa handle kategorikal string secara langsung.
preprocessor_tree = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_features)
    ]
)

X_train_lin = preprocessor_linear.fit_transform(X_train_raw)
X_val_lin = preprocessor_linear.transform(X_val_raw)
X_test_lin = preprocessor_linear.transform(X_test_raw)

X_train_tree = preprocessor_tree.fit_transform(X_train_raw)
X_val_tree = preprocessor_tree.transform(X_val_raw)
X_test_tree = preprocessor_tree.transform(X_test_raw)

print("Preprocessed linear shape:", X_train_lin.shape)
print("Preprocessed tree shape:", X_train_tree.shape)
"""))

# Model 1: Baseline Linear Regression
nb.cells.append(nbf.v4.new_markdown_cell("""## 6. Model 1: Ridge Regression (Baseline)
Kita menggunakan Ridge Regression (L2 Regularized Linear Regression) sebagai baseline."""))

nb.cells.append(nbf.v4.new_code_cell("""# Inisialisasi dan Train Baseline
baseline_model = Ridge(alpha=1.0)
baseline_model.fit(X_train_lin, y_train)

# Prediksi
y_pred_val_base = baseline_model.predict(X_val_lin)
y_pred_test_base = baseline_model.predict(X_test_lin)

# Evaluasi Test Set
mae_base = mean_absolute_error(y_test, y_pred_test_base)
rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_test_base))
r2_base = r2_score(y_test, y_pred_test_base)

print(f"Baseline Ridge -> Test MAE: {mae_base:.2f}, Test RMSE: {rmse_base:.2f}, R2: {r2_base:.4f}")
"""))

# Model 2: Random Forest Regressor
nb.cells.append(nbf.v4.new_markdown_cell("""## 7. Model 2: Random Forest Regressor
Model ensemble klasik berbasis Bagging."""))

nb.cells.append(nbf.v4.new_code_cell("""rf_model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
rf_model.fit(X_train_tree, y_train)

y_pred_test_rf = rf_model.predict(X_test_tree)

mae_rf = mean_absolute_error(y_test, y_pred_test_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_test_rf))
r2_rf = r2_score(y_test, y_pred_test_rf)

print(f"Random Forest -> Test MAE: {mae_rf:.2f}, Test RMSE: {rmse_rf:.2f}, R2: {r2_rf:.4f}")
"""))

# Model 3: XGBoost Regressor
nb.cells.append(nbf.v4.new_markdown_cell("""## 8. Model 3: XGBoost Regressor
Model Gradient Boosting terkenal yang sangat kuat untuk data tabular."""))

nb.cells.append(nbf.v4.new_code_cell("""xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_model.fit(X_train_tree, y_train)

y_pred_test_xgb = xgb_model.predict(X_test_tree)

mae_xgb = mean_absolute_error(y_test, y_pred_test_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_test_xgb))
r2_xgb = r2_score(y_test, y_pred_test_xgb)

print(f"XGBoost -> Test MAE: {mae_xgb:.2f}, Test RMSE: {rmse_xgb:.2f}, R2: {r2_xgb:.4f}")
"""))

# Model 4: LightGBM Regressor
nb.cells.append(nbf.v4.new_markdown_cell("""## 9. Model 4: LightGBM Regressor
Gradient Boosting yang sangat efisien dan handal dalam kecepatan serta akurasi."""))

nb.cells.append(nbf.v4.new_code_cell("""lgb_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, max_depth=5, subsample=0.8, random_state=42, verbose=-1)
lgb_model.fit(X_train_tree, y_train)

y_pred_test_lgb = lgb_model.predict(X_test_tree)

mae_lgb = mean_absolute_error(y_test, y_pred_test_lgb)
rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_test_lgb))
r2_lgb = r2_score(y_test, y_pred_test_lgb)

print(f"LightGBM -> Test MAE: {mae_lgb:.2f}, Test RMSE: {rmse_lgb:.2f}, R2: {r2_lgb:.4f}")
"""))

# Model 5: CatBoost Regressor
nb.cells.append(nbf.v4.new_markdown_cell("""## 10. Model 5: CatBoost Regressor
CatBoost sangat unggul dalam menangani fitur kategorikal (seperti nama Provinsi) dan mengurangi overfitting secara alami."""))

nb.cells.append(nbf.v4.new_code_cell("""# CatBoost bisa menggunakan data mentah (raw) dengan mendeklarasikan categorical features secara langsung
cat_features_indices = [X_train_raw.columns.get_loc('Provinsi')]

cb_model = CatBoostRegressor(iterations=500, learning_rate=0.03, depth=6, random_seed=42, verbose=0)
cb_model.fit(X_train_raw, y_train, cat_features=cat_features_indices)

y_pred_test_cb = cb_model.predict(X_test_raw)

mae_cb = mean_absolute_error(y_test, y_pred_test_cb)
rmse_cb = np.sqrt(mean_squared_error(y_test, y_pred_test_cb))
r2_cb = r2_score(y_test, y_pred_test_cb)

print(f"CatBoost -> Test MAE: {mae_cb:.2f}, Test RMSE: {rmse_cb:.2f}, R2: {r2_cb:.4f}")
"""))

# Model 6: Ensemble Regressor (Weighted Average)
nb.cells.append(nbf.v4.new_markdown_cell("""## 11. Model 6: Ensemble Regressor
Pendekatan robust dalam pemodelan data tabular adalah menggunakan metode **Ensemble**. Kita menggabungkan prediksi dari XGBoost, LightGBM, dan CatBoost dengan bobot rata-rata agar menghasilkan prediksi yang jauh lebih kokoh (robust)."""))

nb.cells.append(nbf.v4.new_code_cell("""# Kombinasi bobot ensemble
# CatBoost & XGBoost biasanya berkinerja sangat baik, mari beri bobot seimbang
y_pred_test_ensemble = (0.3 * y_pred_test_xgb) + (0.3 * y_pred_test_lgb) + (0.4 * y_pred_test_cb)

mae_ens = mean_absolute_error(y_test, y_pred_test_ensemble)
rmse_ens = np.sqrt(mean_squared_error(y_test, y_pred_test_ensemble))
r2_ens = r2_score(y_test, y_pred_test_ensemble)

print(f"Ensemble Regressor -> Test MAE: {mae_ens:.2f}, Test RMSE: {rmse_ens:.2f}, R2: {r2_ens:.4f}")
"""))

# Komparasi Evaluasi
nb.cells.append(nbf.v4.new_markdown_cell("""## 12. Komparasi Performa Semua Model
Mari bandingkan tingkat error (MAE & RMSE) serta nilai kecocokan model ($R^2$) secara komprehensif."""))

nb.cells.append(nbf.v4.new_code_cell("""models = ['Ridge Baseline', 'Random Forest', 'XGBoost', 'LightGBM', 'CatBoost', 'Ensemble Regressor']
mae_scores = [mae_base, mae_rf, mae_xgb, mae_lgb, mae_cb, mae_ens]
rmse_scores = [rmse_base, rmse_rf, rmse_xgb, rmse_lgb, rmse_cb, rmse_ens]
r2_scores = [r2_base, r2_rf, r2_xgb, r2_lgb, r2_cb, r2_ens]

summary_df = pd.DataFrame({
    'Model': models,
    'MAE (Rupiah)': mae_scores,
    'RMSE (Rupiah)': rmse_scores,
    'R2 Score': r2_scores
}).sort_values('MAE (Rupiah)').reset_index(drop=True)

print(summary_df)

# Plotting Komparasi
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# MAE
sns.barplot(x='MAE (Rupiah)', y='Model', data=summary_df, ax=axes[0], palette='Blues_r')
axes[0].set_title('Mean Absolute Error (Lower is Better)')
axes[0].set_xlabel('MAE (Rupiah)')

# RMSE
sns.barplot(x='RMSE (Rupiah)', y='Model', data=summary_df, ax=axes[1], palette='Oranges_r')
axes[1].set_title('Root Mean Squared Error (Lower is Better)')
axes[1].set_xlabel('RMSE (Rupiah)')

# R2 Score
sns.barplot(x='R2 Score', y='Model', data=summary_df.sort_values('R2 Score', ascending=False), ax=axes[2], palette='Greens_r')
axes[2].set_title('R2 Score (Higher is Better)')
axes[2].set_xlabel('R2 Score')

plt.tight_layout()
plt.show()
"""))

# Visualisasi Hasil Aktual vs Prediksi
nb.cells.append(nbf.v4.new_markdown_cell("""## 13. Visualisasi Hasil Aktual vs Prediksi 2025
Menampilkan perbandingan data pengeluaran aktual tahun 2025 dengan hasil prediksi model terbaik (Ensemble Regressor) untuk beberapa provinsi sampel."""))

nb.cells.append(nbf.v4.new_code_cell("""# Siapkan dataframe untuk plotting
plot_df = pd.DataFrame({
    'Provinsi': X_test_raw['Provinsi'],
    'Aktual': y_test,
    'Prediksi_Ensemble': y_pred_test_ensemble,
    'Prediksi_Ridge': y_pred_test_base
}).reset_index(drop=True)

# Ambil sampel 15 provinsi pertama untuk visualisasi agar tidak terlalu padat
sample_plot = plot_df.head(15)

plt.figure(figsize=(15, 7))
x = np.arange(len(sample_plot))
width = 0.25

plt.bar(x - width, sample_plot['Aktual'], width, label='Aktual (2025)', color='#2ca02c')
plt.bar(x, sample_plot['Prediksi_Ensemble'], width, label='Ensemble Regressor', color='#1f77b4')
plt.bar(x + width, sample_plot['Prediksi_Ridge'], width, label='Ridge Baseline', color='#d62728')

plt.xlabel('Provinsi')
plt.ylabel('Total Pengeluaran per Kapita (Rupiah)')
plt.title('Perbandingan Pengeluaran Aktual vs Prediksi Tahun 2025 (Sampel 15 Provinsi)')
plt.xticks(x, sample_plot['Provinsi'], rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()
"""))

# Simpan notebook
os.makedirs('notebooks', exist_ok=True)
output_nb_name = os.path.join('notebooks', 'analisis_daya_beli_regresi.ipynb')
with open(output_nb_name, 'w') as f:
    nbf.write(nb, f)

print(f"Berhasil membuat notebook {output_nb_name}")
