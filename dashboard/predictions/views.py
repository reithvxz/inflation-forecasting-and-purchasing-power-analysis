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
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

LSTM_MODEL = None
LSTM_SCALER = None
RIDGE_MODEL = None
INFLASI_PRED_MEM = None
DATA_HISTORIS = None
RATA_PENGELUARAN = 1450000

def load_models():
    global LSTM_MODEL, LSTM_SCALER, RIDGE_MODEL, INFLASI_PRED_MEM, DATA_HISTORIS, RATA_PENGELUARAN
    
    project_root = os.path.dirname(settings.BASE_DIR)
    models_dir = os.path.join(project_root, 'models')
    data_path = os.path.join(project_root, 'datasets', 'processed', 'clean_inflasi_ts.csv')
    
    ridge_path = os.path.join(models_dir, 'best_daya_beli_ridge.pkl')
    if os.path.exists(ridge_path) and RIDGE_MODEL is None:
        with open(ridge_path, 'rb') as f:
            RIDGE_MODEL = pickle.load(f)
            
    lstm_path = os.path.join(models_dir, 'lstm_model.pt')
    scaler_path = os.path.join(models_dir, 'lstm_scaler.pkl')
    
    if os.path.exists(lstm_path) and os.path.exists(scaler_path) and LSTM_MODEL is None:
        with open(scaler_path, 'rb') as f:
            LSTM_SCALER = pickle.load(f)
        
        LSTM_MODEL = LSTMModel(input_size=4, hidden_size=64, num_layers=2, output_size=1)
        LSTM_MODEL.load_state_dict(torch.load(lstm_path, weights_only=True))
        LSTM_MODEL.eval()
        
        df = pd.read_csv(data_path)
        df['Tanggal'] = pd.to_datetime(df['Tanggal'])
        df = df.sort_values('Tanggal').reset_index(drop=True)
        df.set_index('Tanggal', inplace=True)

        df['USD_IDR'] = df['USD_IDR'].interpolate(method='linear').ffill().bfill()
        df['IHK'] = df['IHK'].interpolate(method='linear').bfill()
        last_known_ihk_idx = df['IHK'].dropna().index[-1]
        for date in df.loc[df.index > last_known_ihk_idx].index:
            prev_date = date - pd.DateOffset(months=1)
            if prev_date not in df.index:
                prev_date = df.index[df.index.get_loc(date) - 1]
            inflasi = df.loc[date, 'Inflasi_MoM']
            df.loc[date, 'IHK'] = df.loc[prev_date, 'IHK'] * (1 + (inflasi / 100))

        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

        features = ['Inflasi_MoM', 'IHK', 'BI_Rate', 'USD_IDR']
        last_12 = df[features].tail(12)
        scaled_last_12 = LSTM_SCALER.transform(last_12)
        X_pred = torch.tensor(np.array([scaled_last_12]), dtype=torch.float32)
        
        with torch.no_grad():
            pred_scaled = LSTM_MODEL(X_pred).numpy()
            
        dummy = np.zeros((1, 4))
        dummy[0, 0] = pred_scaled[0, 0]
        inflasi_pred = LSTM_SCALER.inverse_transform(dummy)[0, 0]
        INFLASI_PRED_MEM = inflasi_pred
        
        # Simpan 24 data terakhir untuk plot grafik di web agar lebih representatif
        recent_df = df.tail(24).copy()
        recent_df['Bulan_Tahun'] = recent_df.index.strftime('%b %Y')
        
        labels = recent_df['Bulan_Tahun'].tolist()
        data_actual = recent_df['Inflasi_MoM'].tolist()
        
        labels.append("Mar 2026 (Prediksi)")
        data_actual.append(None)
        
        data_pred = [None] * 24
        data_pred[-1] = recent_df['Inflasi_MoM'].iloc[-1]
        data_pred.append(inflasi_pred)
        
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
        'inflasi_pred': inflasi_pred or 0.0,
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
        'inflasi_pred': inflasi_pred or 0.0,
        'labels': hist_data['labels'] if hist_data else '[]',
        'data_actual': hist_data['data_actual'] if hist_data else '[]',
        'data_pred': hist_data['data_pred'] if hist_data else '[]',
    }
    return render(request, 'predictions/forecasting.html', context)


def get_regression_dummy_data(inflasi_val):
    return pd.DataFrame([{
        'Tahun': 2024,
        'UMP': 3000000.0, 
        'TPT': 5.5,
        'PDRB_HargaBerlaku': 800000.0,
        'PDRB_HargaKonstan': 500000.0,
        'Inflasi_Rata_Tahunan': inflasi_val,
        'GDP_Deflator': 1.6,
        'Real_UMP': 3000000.0 / (1 + inflasi_val),
        'PDRB_to_UMP': 0.16,
        'TPT_x_UMP': 5.5 * 3000000.0,
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
