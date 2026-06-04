import os
import pickle
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1, :])
        out = self.fc(out)
        return out

LSTM_MODEL = None
LSTM_SCALER_X = None
LSTM_SCALER_Y = None
RIDGE_MODEL = None
INFLASI_PRED_MEM = None
DATA_HISTORIS = None
LSTM_FEATURES = None
RATA_PENGELUARAN = 1450000

def load_models():
    global LSTM_MODEL, LSTM_SCALER_X, LSTM_SCALER_Y, RIDGE_MODEL, INFLASI_PRED_MEM, DATA_HISTORIS, LSTM_FEATURES, RATA_PENGELUARAN
    
    project_root = os.path.dirname(settings.BASE_DIR)
    models_dir = os.path.join(project_root, 'models')
    data_path = os.path.join(project_root, 'datasets', 'processed', 'clean_inflasi_ts.csv')
    
    ridge_path = os.path.join(models_dir, 'best_daya_beli_ridge.pkl')
    if os.path.exists(ridge_path) and RIDGE_MODEL is None:
        with open(ridge_path, 'rb') as f:
            raw = pickle.load(f)
        # Model bisa berupa pipeline langsung atau dictionary bundle
        if isinstance(raw, dict) and 'pipeline' in raw:
            RIDGE_MODEL = raw['pipeline']
        else:
            RIDGE_MODEL = raw
            
    lstm_path = os.path.join(models_dir, 'lstm_model.pt')
    scaler_x_path = os.path.join(models_dir, 'lstm_scaler_x.pkl')
    scaler_y_path = os.path.join(models_dir, 'lstm_scaler_y.pkl')
    
    # Cek apakah file scaler baru ada, jika tidak fallback ke scaler lama
    if not os.path.exists(scaler_x_path):
        scaler_x_path = os.path.join(models_dir, 'lstm_scaler.pkl')
        scaler_y_path = None

    if os.path.exists(lstm_path) and LSTM_MODEL is None:
        # Load Scalers
        if scaler_y_path and os.path.exists(scaler_x_path) and os.path.exists(scaler_y_path):
            with open(scaler_x_path, 'rb') as f:
                LSTM_SCALER_X = pickle.load(f)
            with open(scaler_y_path, 'rb') as f:
                LSTM_SCALER_Y = pickle.load(f)
        elif os.path.exists(scaler_x_path):
            with open(scaler_x_path, 'rb') as f:
                LSTM_SCALER_X = pickle.load(f)
            LSTM_SCALER_Y = LSTM_SCALER_X # Fallback
        else:
            return INFLASI_PRED_MEM, DATA_HISTORIS, RATA_PENGELUARAN
        
        # Load checkpoint
        checkpoint = torch.load(lstm_path, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            input_size = checkpoint.get('input_size', 44)
            seq_length = checkpoint.get('seq_length', 12)
            state_dict = checkpoint['model_state_dict']
            LSTM_FEATURES = checkpoint.get('feature_columns', None)
        else:
            input_size = 44
            seq_length = 12
            state_dict = checkpoint
        
        LSTM_MODEL = LSTMModel(input_size=input_size, hidden_size=128, num_layers=2, output_size=1)
        LSTM_MODEL.load_state_dict(state_dict)
        LSTM_MODEL.eval()
        
        # Load Data
        df = pd.read_csv(data_path)
        df['Tanggal'] = pd.to_datetime(df['Tanggal'])
        df = df.sort_values('Tanggal').reset_index(drop=True)
        df.set_index('Tanggal', inplace=True)
        
        # Imputasi & Feature Engineering (harus identik dengan training)
        df = df.ffill().bfill()
        df['Bulan_Sin'] = np.sin(2 * np.pi * df['Bulan']/12)
        df['Bulan_Cos'] = np.cos(2 * np.pi * df['Bulan']/12)
        if 'Harga_Minyak_USD' in df.columns and 'USD_IDR' in df.columns:
            df['Oil_x_USDIDR'] = df['Harga_Minyak_USD'] * df['USD_IDR']

        # Siapkan fitur (identik dengan training)
        if LSTM_FEATURES is None:
             # Fallback jika feature_columns tidak disimpan
            exclude_cols = ['Bulan', 'Tahun']
            feature_cols = [c for c in df.columns if c not in exclude_cols]
            if 'Inflasi_MoM' in feature_cols: feature_cols.remove('Inflasi_MoM')
            feature_cols = ['Inflasi_MoM'] + feature_cols
        else:
            feature_cols = LSTM_FEATURES

        df_lstm = df[feature_cols].copy()
        
        # Ambil sequence terakhir untuk prediksi
        last_seq = df_lstm.tail(seq_length).values
        X_scaled = LSTM_SCALER_X.transform(last_seq[:, 1:]) # Exclude target
        y_scaled = LSTM_SCALER_Y.transform(last_seq[:, 0].reshape(-1, 1))
        
        # Bentuk tensor: (1, seq_len, features)
        # Perlu reshape X_scaled dan y_scaled menjadi sequence gabungan
        # Karena model dilatih dengan input gabungan (target+exogenous), 
        # kita perlu gabungkan kembali dalam format yang benar.
        # Namun, model kita dilatih dengan input X_scaled (tanpa target) dan target y_scaled terpisah?
        # Tidak, di save_lstm_model.py: X_scaled = scaler_X.transform(X_all) (X_all excludes target)
        # Tapi create_lstm_sequences menerima X_scaled dan y_scaled secara terpisah.
        # Wait, di save_lstm_model.py:
        # X_all = df_lstm.drop('Inflasi_MoM', axis=1).values
        # y_all = df_lstm['Inflasi_MoM'].values.reshape(-1, 1)
        # X_seq, y_seq = create_lstm_sequences(X_scaled, y_scaled, lag_steps)
        # Jadi input model adalah X_scaled.
        
        # Prediksi langkah selanjutnya (Mei 2026)
        X_input = torch.tensor(np.array([X_scaled]), dtype=torch.float32)
        
        with torch.no_grad():
            pred_scaled = LSTM_MODEL(X_input).numpy()
            
        inflasi_pred = float(LSTM_SCALER_Y.inverse_transform(pred_scaled)[0][0])
        INFLASI_PRED_MEM = inflasi_pred
        
        # Recursive Forecast untuk Juni 2026
        # Kita perlu update sequence: geser, buang bulan 1, tambah Mei di akhir
        # Untuk Mei: kita sudah punya X_scaled (12 bulan: Apr 25 - Mar 26? atau May 25 - Apr 26?)
        # Asumsi last_seq adalah May 25 - Apr 26 (12 bulan).
        # Maka prediksi Mei 26.
        # Untuk Juni: input harus Jun 25 - May 26.
        # Kita asumsikan fitur exogenous Juni 26 = Mei 26 (atau flat).
        # Kita update kolom target di sequence lama dengan prediksi baru?
        # TIDAK. Input model kita hanya X (exogenous). Target tidak masuk ke input LSTM!
        # Karena model hanya pakai exogenous features, sequence untuk Juni 26 
        # harus geser 1 bulan dari sequence Mei 26.
        # Kita asumsikan fitur exogenous Juni 26 = Mei 26 (copy baris terakhir).
        
        next_seq_exo = np.vstack([X_scaled[1:], X_scaled[-1:]])
        X_input_next = torch.tensor(np.array([next_seq_exo]), dtype=torch.float32)
        
        with torch.no_grad():
            pred_scaled_next = LSTM_MODEL(X_input_next).numpy()
            
        inflasi_pred_next = LSTM_SCALER_Y.inverse_transform(pred_scaled_next)[0][0]
        
        # Siapkan data historis untuk grafik
        # Tampilkan 24 bulan terakhir (termasuk Mei 2026 yang sudah aktual)
        recent_df = df.tail(24).copy()
        recent_df['Bulan_Tahun'] = recent_df.index.strftime('%b %Y')
        
        labels = recent_df['Bulan_Tahun'].tolist()
        data_actual = recent_df['Inflasi_MoM'].tolist()
        
        # Tambahkan prediksi Juni 2026 saja
        labels.append("Jun 2026 (Pred)")
        data_actual.append(None)
        
        # Garis prediksi: connect dari Mei 2026 aktual ke prediksi Juni
        data_pred = [None] * 24
        data_pred[-1] = recent_df['Inflasi_MoM'].iloc[-1]  # Mei 2026 aktual
        data_pred.append(float(inflasi_pred_next))  # Juni 2026 prediksi
        
        DATA_HISTORIS = {
            'labels': json.dumps(labels),
            'data_actual': json.dumps(data_actual),
            'data_pred': json.dumps(data_pred)
        }
        
    return INFLASI_PRED_MEM, DATA_HISTORIS, RATA_PENGELUARAN


def landing_page(request):
    inflasi_pred, _, rata_pengeluaran = load_models()
    
    # Ambil data tambahan dari dataset clean_daya_beli.csv untuk visualisasi Overview
    project_root = os.path.dirname(settings.BASE_DIR)
    db_path = os.path.join(project_root, 'datasets', 'processed', 'clean_daya_beli.csv')
    
    chart_labels = []
    ump_data = []
    pdrb_data = []
    pie_data = []
    
    if os.path.exists(db_path):
        df_db = pd.read_csv(db_path)
        # Ambil rata-rata UMP per tahun
        yearly_data = df_db.groupby('Tahun').agg({'UMP': 'mean', 'PDRB_HargaKonstan': 'mean'}).reset_index().sort_values('Tahun')
        chart_labels = yearly_data['Tahun'].tolist()
        ump_data = yearly_data['UMP'].tolist()
        pdrb_data = yearly_data['PDRB_HargaKonstan'].tolist()
        
        # Ambil data untuk pie chart proporsi pengeluaran secara keseluruhan
        makanan_mean = df_db['Pengeluaran_Makanan'].mean() if 'Pengeluaran_Makanan' in df_db.columns else 600000
        bukan_makanan_mean = df_db['Pengeluaran_Bukan_Makanan'].mean() if 'Pengeluaran_Bukan_Makanan' in df_db.columns else 850000
        pie_data = [float(makanan_mean), float(bukan_makanan_mean)]

    context = {
        'inflasi_pred': float(inflasi_pred) if inflasi_pred else 0.0,
        'rata_pengeluaran': "{:,.0f}".format(rata_pengeluaran).replace(',', '.'),
        'chart_labels': json.dumps(chart_labels),
        'ump_data': json.dumps(ump_data),
        'pdrb_data': json.dumps(pdrb_data),
        'pie_data': json.dumps(pie_data),
    }
    return render(request, 'predictions/landing.html', context)


def forecasting_page(request):
    inflasi_pred, hist_data, _ = load_models()
    context = {
        'inflasi_pred': float(inflasi_pred) if inflasi_pred else 0.0,
        'labels': hist_data['labels'] if hist_data else '[]',
        'data_actual': hist_data['data_actual'] if hist_data else '[]',
        'data_pred': hist_data['data_pred'] if hist_data else '[]',
    }
    return render(request, 'predictions/forecasting.html', context)


def get_regression_dummy_data(inflasi_val):
    # Model Ridge membutuhkan: Real_UMP, TPT, PDRB_HargaKonstan, Inflasi_Rata_Tahunan, Provinsi
    return pd.DataFrame([{
        'Real_UMP': 3000000.0 / (1 + inflasi_val),
        'TPT': 5.5,
        'PDRB_HargaKonstan': 500000.0,
        'Inflasi_Rata_Tahunan': inflasi_val,
        'Provinsi': 'JAWA TIMUR'
    }])

def daya_beli_page(request):
    load_models()
    
    # Kalkulasi slope eksak untuk slider frontend agar animasi real-time murni dari client side
    base_inflasi = 0.0
    if RIDGE_MODEL is not None:
        val0 = RIDGE_MODEL.predict(get_regression_dummy_data(0.0))[0]
        val1 = RIDGE_MODEL.predict(get_regression_dummy_data(1.0))[0]
        slope_per_percent = float(val1 - val0)
        base_value = float(val0)
    else:
        slope_per_percent = -15000.0
        base_value = 1450000.0
        
    context = {
        'slope': slope_per_percent,
        'base_value': base_value
    }
    return render(request, 'predictions/daya_beli.html', context)

# API endpoint ini mungkin tidak diperlukan lagi jika kita pakai slope JS murni, 
# tapi tetap kita pertahankan untuk kebutuhan lain.
def simulate_daya_beli(request):
    inflasi_val = request.GET.get('inflasi', 0.0)
    try:
        inflasi_val = float(inflasi_val)
    except ValueError:
        return JsonResponse({'error': 'Invalid input'}, status=400)
        
    load_models()
    if RIDGE_MODEL is None:
        return JsonResponse({'error': 'Model belum siap'}, status=500)
        
    dummy_input = get_regression_dummy_data(inflasi_val)
    try:
        val = RIDGE_MODEL.predict(dummy_input)[0]
        if val < 0: val = 0
        return JsonResponse({'predicted_pengeluaran': float(val)})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
