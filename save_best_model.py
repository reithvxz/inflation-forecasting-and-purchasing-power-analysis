import os
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def main():
    # 1. Load Data
    data_path = os.path.join('datasets', 'processed', 'clean_daya_beli.csv')
    if not os.path.exists(data_path):
        print(f"Error: file {data_path} tidak ditemukan!")
        return
        
    df = pd.read_csv(data_path)
    
    # 2. Pembersihan & Eliminasi fitur TPAK dan Pct_Penduduk_Miskin
    df_clean = df.drop(columns=['TPAK', 'Pct_Penduduk_Miskin', 'Pengeluaran_Makanan', 'Pengeluaran_Bukan_Makanan'])
    
    # 3. Feature Engineering
    df_clean['GDP_Deflator'] = df_clean['PDRB_HargaBerlaku'] / df_clean['PDRB_HargaKonstan']
    df_clean['Real_UMP'] = df_clean['UMP'] / (1 + df_clean['Inflasi_Rata_Tahunan'])
    df_clean['PDRB_to_UMP'] = df_clean['PDRB_HargaKonstan'] / df_clean['UMP']
    df_clean['TPT_x_UMP'] = df_clean['TPT'] * df_clean['UMP']
    
    # 4. Tentukan target dan fitur
    target_col = 'Total_Pengeluaran'
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    # 5. Pipeline Preprocessing
    num_features = ['Tahun', 'UMP', 'TPT', 'PDRB_HargaBerlaku', 'PDRB_HargaKonstan', 
                    'Inflasi_Rata_Tahunan', 'GDP_Deflator', 'Real_UMP', 'PDRB_to_UMP', 'TPT_x_UMP']
    cat_features = ['Provinsi']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ]
    )
    
    # 6. Bangun Pipeline Utuh (Preprocessor + Model Ridge Terbaik)
    # Kita menggunakan alpha=1.0 yang terbukti memberikan performa terbaik pada pengujian
    best_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1.0))
    ])
    
    # 7. Fit model pada SELURUH DATA untuk performa deployment maksimal
    print("Melatih model Ridge pada seluruh dataset...")
    best_pipeline.fit(X, y)
    
    # 8. Buat direktori models dan ekspor model ke file .pkl
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    model_file = os.path.join(models_dir, 'best_daya_beli_ridge.pkl')
    
    with open(model_file, 'wb') as f:
        pickle.dump(best_pipeline, f)
        
    print(f"Berhasil mengekspor model terbaik ke: {model_file}")
    
    # Test loading model
    with open(model_file, 'rb') as f:
        loaded_model = pickle.load(f)
    print("Test load model: SUCCESS!")
    print("Contoh prediksi pertama:", loaded_model.predict(X.iloc[[0]]))

if __name__ == '__main__':
    main()
