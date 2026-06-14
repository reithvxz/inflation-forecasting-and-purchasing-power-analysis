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
from predictions.daya_beli_model import prepare_daya_beli_dataframe

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
RIDGE_MODEL_BUNDLE = None
INFLASI_PRED_MEM = None
DATA_HISTORIS = None
LSTM_FEATURES = None
RATA_PENGELUARAN = 1450000
RIDGE_SIMULATION_DEFAULTS = None
PROVINCE_SIMULATION_BASELINES = None


def _safe_float(value, fallback=0.0):
    try:
        if value is None or pd.isna(value):
            return float(fallback)
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _pct_change(current, previous, fallback=0.0):
    current_val = _safe_float(current, fallback)
    previous_val = _safe_float(previous, 0.0)
    if previous_val == 0:
        return _safe_float(fallback, 0.0)
    return ((current_val - previous_val) / previous_val) * 100.0


def _build_ridge_simulation_defaults(project_root):
    """Build a realistic baseline row that matches Ridge training features."""
    path = os.path.join(project_root, 'datasets', 'processed', 'clean_daya_beli.csv')
    if not os.path.exists(path):
        return None

    df = prepare_daya_beli_dataframe(pd.read_csv(path))
    if df.empty:
        return None

    latest_year = df['Tahun'].max()
    latest = df[df['Tahun'] == latest_year]
    baseline_source = latest if not latest.empty else df
    numeric_defaults = baseline_source.mean(numeric_only=True).to_dict()

    return {
        'Provinsi': baseline_source['Provinsi'].mode().iat[0] if not baseline_source['Provinsi'].mode().empty else 'Jawa Timur',
        'UMP': float(numeric_defaults.get('UMP', 3000000.0)),
        'Inflasi_WB_Annual': float(numeric_defaults.get('Inflasi_WB_Annual', 2.7)),
        'GDP_PerCapita_PPP': float(numeric_defaults.get('GDP_PerCapita_PPP', 13800.0)),
        'Pct_Unemployment_WB': float(numeric_defaults.get('Pct_Unemployment_WB', 3.4)),
        'Poverty_Headcount_Pct': float(numeric_defaults.get('Poverty_Headcount_Pct', 9.4)),
        'RATA_PENGELUARAN': float(numeric_defaults.get('Total_Pengeluaran', 1450000.0)),
        'numeric_defaults': numeric_defaults,
    }


def _build_province_simulation_baselines(project_root):
    path = os.path.join(project_root, 'datasets', 'processed', 'clean_daya_beli.csv')
    if not os.path.exists(path):
        return {}

    df = prepare_daya_beli_dataframe(pd.read_csv(path))
    if df.empty:
        return {}

    latest_rows = df.groupby('Provinsi').tail(1).copy()
    baselines = {}
    for _, row in latest_rows.iterrows():
        province = row.get('Provinsi')
        if not province:
            continue
        row_dict = row.to_dict()
        baselines[province] = {
            'baseline_year': int(_safe_float(row_dict.get('Tahun'), 0)),
            'baseline_pengeluaran': _safe_float(row_dict.get('Total_Pengeluaran'), 1450000.0),
            'fields': row_dict,
        }
    return baselines


def _get_province_simulation_baselines():
    global PROVINCE_SIMULATION_BASELINES
    if PROVINCE_SIMULATION_BASELINES is None:
        project_root = os.path.dirname(settings.BASE_DIR)
        PROVINCE_SIMULATION_BASELINES = _build_province_simulation_baselines(project_root)
    return PROVINCE_SIMULATION_BASELINES or {}


def _build_simulation_input(province, overrides=None):
    load_models()
    baselines = _get_province_simulation_baselines()
    if province not in baselines:
        raise KeyError(f"Provinsi '{province}' tidak ditemukan")

    baseline = baselines[province]
    fields = baseline['fields']
    overrides = overrides or {}

    inflasi_pct = _safe_float(overrides.get('inflasi'), fields.get('Inflasi_Rata_Tahunan', 0.0))
    ump_value = _safe_float(overrides.get('ump'), fields.get('UMP', 3000000.0))
    tpt_value = _safe_float(overrides.get('tpt'), fields.get('TPT', 5.0))
    pdrb_value = _safe_float(overrides.get('pdrb_hargakonstan'), fields.get('PDRB_HargaKonstan', 40000.0))

    denominator = 1 + (inflasi_pct / 100.0)
    real_ump = ump_value / denominator if denominator != 0 else ump_value

    prev_real_ump = _safe_float(fields.get('Prev_Real_UMP'), fields.get('Real_UMP', real_ump))
    prev_pdrb = _safe_float(fields.get('Prev_PDRB_HargaKonstan'), pdrb_value)
    prev_tpt = _safe_float(fields.get('Prev_TPT'), tpt_value)

    feature_row = {
        'Provinsi': province,
        'TPT': tpt_value,
        'PDRB_HargaKonstan': pdrb_value,
        'Inflasi_Rata_Tahunan': inflasi_pct,
        'Gini_Rasio': _safe_float(fields.get('Gini_Rasio'), 0.30),
        'IPM': _safe_float(fields.get('IPM'), 72.4),
        'Garis_Kemiskinan': _safe_float(fields.get('Garis_Kemiskinan'), 609000.0),
        'Jumlah_Penduduk': _safe_float(fields.get('Jumlah_Penduduk'), 8000.0),
        'Pct_Populasi': _safe_float(fields.get('Pct_Populasi'), 2.8),
        'Pct_Akses_Air_Bersih': _safe_float(fields.get('Pct_Akses_Air_Bersih'), 87.7),
        'Protein_gram_per_hari': _safe_float(fields.get('Protein_gram_per_hari'), 62.3),
        'Inflasi_WB_Annual': _safe_float(fields.get('Inflasi_WB_Annual'), 2.7),
        'GDP_PerCapita_PPP': _safe_float(fields.get('GDP_PerCapita_PPP'), 13800.0),
        'Pct_Unemployment_WB': _safe_float(fields.get('Pct_Unemployment_WB'), 3.4),
        'Poverty_Headcount_Pct': _safe_float(fields.get('Poverty_Headcount_Pct'), 9.4),
        'Real_UMP': real_ump,
        'Real_UMP_Growth': _pct_change(real_ump, prev_real_ump, fields.get('Real_UMP_Growth', 0.0)),
        'PDRB_HargaKonstan_Growth': _pct_change(
            pdrb_value,
            prev_pdrb,
            fields.get('PDRB_HargaKonstan_Growth', 0.0),
        ),
        'TPT_Growth': _pct_change(tpt_value, prev_tpt, fields.get('TPT_Growth', 0.0)),
        'UMP_x_PDRB': real_ump * pdrb_value,
        'Inflasi_x_TPT': inflasi_pct * tpt_value,
        'Log_PDRB': float(np.log1p(max(pdrb_value, 0.0))),
        'Log_UMP': float(np.log1p(max(real_ump, 0.0))),
    }

    if RIDGE_MODEL_BUNDLE is not None and 'num_features' in RIDGE_MODEL_BUNDLE:
        for feat in RIDGE_MODEL_BUNDLE.get('num_features', []):
            if feat not in feature_row:
                feature_row[feat] = _safe_float(fields.get(feat), 0.0)

    return pd.DataFrame([feature_row]), baseline

def load_models():
    global LSTM_MODEL, LSTM_SCALER_X, LSTM_SCALER_Y, RIDGE_MODEL, RIDGE_MODEL_BUNDLE, INFLASI_PRED_MEM, DATA_HISTORIS, LSTM_FEATURES, RATA_PENGELUARAN, RIDGE_SIMULATION_DEFAULTS
    
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
            RIDGE_MODEL_BUNDLE = raw
        else:
            RIDGE_MODEL = raw
            RIDGE_MODEL_BUNDLE = {'legacy_artifact': True}

    if RIDGE_SIMULATION_DEFAULTS is None:
        RIDGE_SIMULATION_DEFAULTS = _build_ridge_simulation_defaults(project_root)
        if RIDGE_SIMULATION_DEFAULTS is not None:
            RATA_PENGELUARAN = RIDGE_SIMULATION_DEFAULTS['RATA_PENGELUARAN']
            
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

        # Pastikan semua kolom fitur ada (imputasi 0 untuk kolom yang hilang)
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0

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
        'ridge_test_r2': round(_safe_float((RIDGE_MODEL_BUNDLE or {}).get('test_r2'), 0.0) * 100, 1),
        'chart_labels': json.dumps(chart_labels),
        'ump_data': json.dumps(ump_data),
        'pdrb_data': json.dumps(pdrb_data),
        'pie_data': json.dumps(pie_data),
    }
    return render(request, 'predictions/landing.html', context)


def forecasting_page(request):
    inflasi_pred, _, _ = load_models()
    load_ensemble()
    
    project_root = os.path.dirname(settings.BASE_DIR)
    data_path = os.path.join(project_root, 'datasets', 'processed', 'clean_inflasi_ts.csv')
    
    # Load full historical data
    df = pd.read_csv(data_path, parse_dates=['Tanggal'])
    df = df.sort_values('Tanggal').reset_index(drop=True)
    
    # Default range: last 5 years
    default_start_year = max(df['Tanggal'].dt.year.min(), df['Tanggal'].dt.year.max() - 4)
    default_end_year = df['Tanggal'].dt.year.max()
    
    # Build full history JSON
    history = {
        'labels': df['Tanggal'].dt.strftime('%Y-%m').tolist(),
        'data_mom': [round(float(v), 2) if not pd.isna(v) else None for v in df['Inflasi_MoM']],
        'data_yoy': [round(float(v), 2) if not pd.isna(v) else None for v in df['Inflasi_YoY']],
    }
    
    # Year range options
    year_min = int(df['Tanggal'].dt.year.min())
    year_max = int(df['Tanggal'].dt.year.max())
    
    # Get next-month prediction
    last_date = df['Tanggal'].iloc[-1]
    next_date = (last_date + pd.DateOffset(months=1)).strftime('%Y-%m')
    last_value = float(df['Inflasi_MoM'].iloc[-1])
    
    # Get ensemble forecast (1 month)
    ensemble_pred = float(inflasi_pred) if inflasi_pred else 0.0
    
    # Get other model predictions
    model_preds = {
        'lstm': ensemble_pred,
    }
    if ENSEMBLE_FORECAST is not None:
        # Format ensemble_forecast.pkl: flat keys 'lstm_forecast', 'arima_forecast', etc.
        lstm_fc = ENSEMBLE_FORECAST.get('lstm_forecast', [])
        arima_fc = ENSEMBLE_FORECAST.get('arima_forecast', [])
        prophet_fc = ENSEMBLE_FORECAST.get('prophet_forecast', [])
        ensemble_fc = ENSEMBLE_FORECAST.get('ensemble_forecast', [])
        if lstm_fc and len(lstm_fc) > 0:
            model_preds['lstm'] = float(lstm_fc[0])
        if arima_fc and len(arima_fc) > 0:
            model_preds['arima'] = float(arima_fc[0])
        if prophet_fc and len(prophet_fc) > 0:
            model_preds['prophet'] = float(prophet_fc[0])
        if ensemble_fc and len(ensemble_fc) > 0:
            model_preds['ensemble'] = float(ensemble_fc[0])
    
    context = {
        'inflasi_pred': ensemble_pred,
        'last_value': last_value,
        'last_date': last_date.strftime('%Y-%m'),
        'next_date': next_date,
        'history': json.dumps(history),
        'model_preds': json.dumps(model_preds),
        'year_min': year_min,
        'year_max': year_max,
        'default_start_year': int(default_start_year),
        'default_end_year': int(default_end_year),
    }
    return render(request, 'predictions/forecasting.html', context)


def get_regression_dummy_data(inflasi_val):
    default_province = (RIDGE_SIMULATION_DEFAULTS or {}).get('Provinsi', 'Jawa Timur')
    dummy_input, _ = _build_simulation_input(default_province, {'inflasi': inflasi_val})
    return dummy_input

def daya_beli_page(request):
    load_models()

    baselines = _get_province_simulation_baselines()
    province_names = sorted(baselines.keys())
    default_province = (RIDGE_SIMULATION_DEFAULTS or {}).get('Provinsi')
    if default_province not in baselines:
        default_province = province_names[0] if province_names else ''

    province_defaults = {}
    for province in province_names:
        baseline = baselines[province]
        fields = baseline['fields']
        province_defaults[province] = {
            'year': baseline['baseline_year'],
            'baseline_pengeluaran': round(_safe_float(baseline['baseline_pengeluaran']), 2),
            'inflasi': round(_safe_float(fields.get('Inflasi_Rata_Tahunan')), 2),
            'ump': round(_safe_float(fields.get('UMP')), 2),
            'tpt': round(_safe_float(fields.get('TPT')), 3),
            'pdrb_hargakonstan': round(_safe_float(fields.get('PDRB_HargaKonstan')), 2),
        }
    context = {
        'provinces': province_names,
        'default_province': default_province,
        'province_defaults_json': json.dumps(province_defaults),
    }
    return render(request, 'predictions/daya_beli.html', context)
def simulate_daya_beli(request):
    provinsi = request.GET.get('provinsi', '').strip()
    if not provinsi:
        return JsonResponse({'error': 'Provinsi wajib dipilih'}, status=400)

    inflasi_val = request.GET.get('inflasi')
    ump_val = request.GET.get('ump')
    tpt_val = request.GET.get('tpt')
    pdrb_val = request.GET.get('pdrb_hargakonstan')

    overrides = {}
    try:
        if inflasi_val not in (None, ''):
            overrides['inflasi'] = float(inflasi_val)
        if ump_val not in (None, ''):
            overrides['ump'] = float(ump_val)
        if tpt_val not in (None, ''):
            overrides['tpt'] = float(tpt_val)
        if pdrb_val not in (None, ''):
            overrides['pdrb_hargakonstan'] = float(pdrb_val)
    except ValueError:
        return JsonResponse({'error': 'Input numerik tidak valid'}, status=400)

    load_models()
    if RIDGE_MODEL is None:
        return JsonResponse({'error': 'Model belum siap'}, status=500)
    if (RIDGE_MODEL_BUNDLE or {}).get('legacy_artifact'):
        return JsonResponse(
            {'error': 'Artifact model daya beli lama tidak kompatibel dengan inference per provinsi terbaru.'},
            status=500,
        )

    try:
        dummy_input, baseline = _build_simulation_input(provinsi, overrides)
        val = float(RIDGE_MODEL.predict(dummy_input)[0])
        predicted_pengeluaran = max(val, 0.0)
        baseline_pengeluaran = _safe_float(baseline['baseline_pengeluaran'], 0.0)
        delta_pct = _pct_change(predicted_pengeluaran, baseline_pengeluaran, 0.0)

        if delta_pct > 3:
            status_label = 'meningkat'
        elif delta_pct < -3:
            status_label = 'menurun'
        else:
            status_label = 'stabil'

        inflasi_used = _safe_float(dummy_input.at[0, 'Inflasi_Rata_Tahunan'])
        inputs_used = {
            'provinsi': provinsi,
            'inflasi': round(inflasi_used, 2),
            'ump': round(_safe_float(dummy_input.at[0, 'Real_UMP']) * (1 + (inflasi_used / 100.0)), 2),
            'tpt': round(_safe_float(dummy_input.at[0, 'TPT']), 3),
            'pdrb_hargakonstan': round(_safe_float(dummy_input.at[0, 'PDRB_HargaKonstan']), 2),
        }
        return JsonResponse({
            'predicted_pengeluaran': round(predicted_pengeluaran, 2),
            'province': provinsi,
            'baseline_year': int(baseline['baseline_year']),
            'baseline_pengeluaran': round(baseline_pengeluaran, 2),
            'inputs_used': inputs_used,
            'status_label': status_label,
        })
    except KeyError:
        return JsonResponse({'error': 'Provinsi tidak ditemukan'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def home_page(request):
    load_models()
    context = {
        'ridge_test_r2': round(_safe_float((RIDGE_MODEL_BUNDLE or {}).get('test_r2'), 0.0) * 100, 1),
        'province_count': len(_get_province_simulation_baselines()),
    }
    return render(request, 'predictions/home.html', context)


def guide_page(request):
    return render(request, 'predictions/guide.html')


# --- Dataset Explorer ---
def datasets_page(request):
    return render(request, 'predictions/datasets.html')


# --- Province Comparison ---
def compare_page(request):
    return render(request, 'predictions/compare.html')


# --- What-If Scenarios ---
def scenarios_page(request):
    return render(request, 'predictions/scenarios.html')


# ============================================================
# API ENDPOINTS FOR REAL DATA
# ============================================================

def _is_number(s):
    """Check if string looks like a number (int or float)."""
    try:
        float(str(s).replace(',', '').replace(' ', ''))
        return True
    except (ValueError, TypeError):
        return False


def api_dataset_sample(request):
    """Return sample rows and column info from a dataset file.

    Supports two modes:
    - ?file=<relative path under datasets/> → read that specific CSV
    - ?dataset=<name> → legacy alias for processed datasets (daya_beli / inflasi)
    """
    n_rows = int(request.GET.get('n', 8))
    project_root = os.path.dirname(settings.BASE_DIR)
    datasets_root = os.path.join(project_root, 'datasets')

    file_param = request.GET.get('file', '').strip()
    dataset = request.GET.get('dataset', '').strip()

    if file_param:
        # Whitelist: only allow CSV files under datasets/ (no path traversal)
        rel = file_param.replace('\\', '/').lstrip('/')
        if '..' in rel.split('/'):
            return JsonResponse({'error': 'Invalid path'}, status=400)
        if not rel.lower().endswith('.csv'):
            return JsonResponse({'error': 'Only CSV files are previewable'}, status=400)
        path = os.path.join(datasets_root, rel)
        if not os.path.isfile(path):
            return JsonResponse({
                'error': 'File not found. Dataset ini mungkin berformat XLSX/XLS atau terdiri dari banyak file (satu CSV per tahun).',
                'file_format': 'XLSX' if 'XLSX' in rel.upper() or 'XLS' in rel.upper() else 'multi-file'
            }, status=404)
    elif dataset == 'daya_beli':
        path = os.path.join(datasets_root, 'processed', 'clean_daya_beli.csv')
    elif dataset == 'inflasi':
        path = os.path.join(datasets_root, 'processed', 'clean_inflasi_ts.csv')
    else:
        return JsonResponse({'error': 'Unknown dataset. Provide ?file= or ?dataset='}, status=400)

    if not os.path.exists(path):
        return JsonResponse({'error': 'File not found'}, status=404)

    try:
        # Try to detect multi-row headers (BPS files often have 1-3 metadata rows)
        # Strategy: read first few lines, pick the first row with most non-empty cells
        # that is "header-like" (mostly text, no numeric values).
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            sample_lines = [f.readline() for _ in range(8)]

        best_row = 0
        best_score = -1
        for i, line in enumerate(sample_lines):
            cells = [c.strip() for c in line.rstrip('\n').split(',') if c.strip()]
            if not cells:
                continue
            # Count cells that are clearly text (not parseable as float)
            text_cells = sum(1 for c in cells if not _is_number(c))
            # Heuristic score: prefer rows where many cells are text and row width is high
            score = text_cells * 10 + len(cells)
            # Penalize early empty rows
            if i < 2 and text_cells <= 1:
                score = -1
            if score > best_score:
                best_score = score
                best_row = i

        # If best score is too low, just use row 0
        if best_score < 0:
            best_row = 0

        df = pd.read_csv(path, header=best_row)

        # Rename first column if it's "Unnamed: 0" (typical for BPS wide-format files)
        if len(df.columns) > 0 and 'Unnamed' in str(df.columns[0]):
            df = df.rename(columns={df.columns[0]: 'Wilayah'})

        # Drop remaining fully-empty columns (Unnamed: X) and fully-empty rows
        df = df.loc[:, ~df.columns.str.contains(r'^Unnamed', na=False)]
        df = df.dropna(how='all').reset_index(drop=True)

        col_names = []
        col_types = []
        for col in df.columns:
            col_names.append(col)
            if pd.api.types.is_numeric_dtype(df[col]):
                col_types.append('number')
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                col_types.append('date')
            else:
                col_types.append('text')

        # Sample first n_rows, fill NaN
        rows = df.head(n_rows).fillna('').astype(str).to_dict('records')

        return JsonResponse({
            'columns': col_names,
            'types': col_types,
            'rows': rows,
            'total_rows': len(df),
            'source_file': os.path.relpath(path, project_root).replace('\\', '/')
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def api_province_list(request):
    """Return list of provinces from daya_beli CSV."""
    project_root = os.path.dirname(settings.BASE_DIR)
    path = os.path.join(project_root, 'datasets', 'processed', 'clean_daya_beli.csv')
    if not os.path.exists(path):
        return JsonResponse({'provinces': []})
    try:
        df = pd.read_csv(path)
        if 'Provinsi' in df.columns:
            provinces = sorted(df['Provinsi'].unique().tolist())
        else:
            provinces = []
        return JsonResponse({'provinces': provinces})
    except Exception:
        return JsonResponse({'provinces': []})


def api_province_data(request):
    """Return data for selected provinces and metric."""
    provinces = request.GET.getlist('provinsi')
    metric = request.GET.get('metric', 'Total_Pengeluaran')
    project_root = os.path.dirname(settings.BASE_DIR)
    path = os.path.join(project_root, 'datasets', 'processed', 'clean_daya_beli.csv')
    
    if not os.path.exists(path):
        return JsonResponse({'error': 'Data not found'}, status=404)
    
    try:
        df = pd.read_csv(path)
        if 'Provinsi' not in df.columns or 'Tahun' not in df.columns:
            return JsonResponse({'error': 'Required columns missing'}, status=500)
        
        if metric not in df.columns:
            metric = 'Total_Pengeluaran'
        
        # Filter by provinces if specified
        if provinces:
            df = df[df['Provinsi'].isin(provinces)]
        
        # Group by province and year
        result = {}
        for prov in df['Provinsi'].unique():
            prov_df = df[df['Provinsi'] == prov].sort_values('Tahun')
            result[prov] = {
                'years': prov_df['Tahun'].tolist(),
                'values': prov_df[metric].fillna(0).tolist()
            }
        
        # Metric info
        metric_info = {
            'Total_Pengeluaran': {'label': 'Daya Beli (Total Pengeluaran)', 'unit': 'Rp', 'format': 'currency'},
            'UMP': {'label': 'Upah Minimum Provinsi', 'unit': 'Rp', 'format': 'currency'},
            'PDRB_HargaKonstan': {'label': 'PDRB Per Kapita', 'unit': 'Rp', 'format': 'currency'},
            'TPT': {'label': 'Tingkat Pengangguran Terbuka', 'unit': '%', 'format': 'percent'},
            'IPM': {'label': 'Indeks Pembangunan Manusia', 'unit': '', 'format': 'number'},
            'Gini_Rasio': {'label': 'Gini Ratio', 'unit': '', 'format': 'number'},
            'Garis_Kemiskinan': {'label': 'Garis Kemiskinan', 'unit': 'Rp', 'format': 'currency'},
            'Pct_Penduduk_Miskin': {'label': '% Penduduk Miskin', 'unit': '%', 'format': 'percent'},
            'Jumlah_Penduduk': {'label': 'Jumlah Penduduk', 'unit': '', 'format': 'number'},
            'Inflasi_Rata_Tahunan': {'label': 'Inflasi Rata-rata Tahunan', 'unit': '%', 'format': 'percent'},
        }
        
        return JsonResponse({
            'data': result,
            'metric_info': metric_info.get(metric, {'label': metric, 'unit': '', 'format': 'number'})
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def api_commodity_prices(request):
    """Return commodity prices and predicted inflation for Rupiah Purchasing Power feature."""
    load_models()

    # Check if World Bank CSV exists
    project_root = os.path.dirname(settings.BASE_DIR)
    csv_path = os.path.join(project_root, 'datasets', 'raw', 'CMO-April-2026.csv')

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # TODO: Parse real commodity data from CSV
            pass
        except Exception:
            pass

    # Simulated realistic commodity prices
    commodities = {
        "beras": {"name": "Beras", "price": 12500, "unit": "Rp/kg", "change_pct": 2.3},
        "gula": {"name": "Gula", "price": 16000, "unit": "Rp/kg", "change_pct": -0.5},
        "minyak_goreng": {"name": "Minyak Goreng", "price": 14500, "unit": "Rp/liter", "change_pct": 1.8},
        "telur": {"name": "Telur", "price": 28000, "unit": "Rp/kg", "change_pct": 3.1},
        "bbm_pertalite": {"name": "BBM Pertalite", "price": 10000, "unit": "Rp/liter", "change_pct": 0.0},
        "daging_ayam": {"name": "Daging Ayam", "price": 35000, "unit": "Rp/kg", "change_pct": 1.2},
    }

    inflasi_val = float(INFLASI_PRED_MEM) if INFLASI_PRED_MEM is not None else 2.5

    return JsonResponse({
        "commodities": commodities,
        "inflasi_prediksi": inflasi_val,
        "base_pengeluaran": RATA_PENGELUARAN,
    })


def api_all_metrics_latest(request):
    """Return latest year metrics for all provinces (for radar chart)."""
    project_root = os.path.dirname(settings.BASE_DIR)
    path = os.path.join(project_root, 'datasets', 'processed', 'clean_daya_beli.csv')
    if not os.path.exists(path):
        return JsonResponse({'error': 'Data not found'}, status=404)
    try:
        df = pd.read_csv(path)
        latest_year = df['Tahun'].max()
        latest = df[df['Tahun'] == latest_year]
        
        metrics = ['Total_Pengeluaran', 'UMP', 'PDRB_HargaKonstan', 'TPT', 'IPM', 'Gini_Rasio', 
                   'Pct_Penduduk_Miskin', 'Inflasi_Rata_Tahunan']
        available = [m for m in metrics if m in latest.columns]
        
        result = {}
        for _, row in latest.iterrows():
            prov = row['Provinsi']
            result[prov] = {m: float(row[m]) if pd.notna(row[m]) else 0 for m in available}
            result[prov]['Tahun'] = int(row['Tahun'])
        
        return JsonResponse({'latest_year': int(latest_year), 'provinces': result, 'metrics': available})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def api_usd_idr_latest(request):
    """Return the latest USD/IDR rate and its previous daily observation."""
    import urllib.request
    import json as json_lib
    from datetime import date, timedelta
    
    project_root = os.path.dirname(settings.BASE_DIR)
    path = os.path.join(project_root, 'datasets', 'processed', 'clean_inflasi_ts.csv')
    
    # Frankfurter returns one consistent daily series and skips non-trading days.
    daily_rate = None
    daily_date = None
    previous_rate = None
    previous_date = None
    daily_history = []
    try:
        end_date = date.today()
        start_date = end_date - timedelta(days=14)
        url = (
            "https://api.frankfurter.dev/v1/"
            f"{start_date.isoformat()}..{end_date.isoformat()}"
            "?base=USD&symbols=IDR"
        )
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json_lib.loads(resp.read().decode())
            observations = sorted(
                (
                    observation_date,
                    rates.get('IDR'),
                )
                for observation_date, rates in data.get('rates', {}).items()
                if rates.get('IDR') is not None
            )
            if observations:
                daily_date, daily_rate = observations[-1]
                daily_rate = round(float(daily_rate), 2)
                daily_history = [round(float(rate), 2) for _, rate in observations[-10:]]
            if len(observations) >= 2:
                previous_date, previous_rate = observations[-2]
                previous_rate = round(float(previous_rate), 2)
    except Exception:
        pass
    
    # 2. Load monthly data from processed CSV
    monthly_rate = None
    monthly_date = None
    monthly_history = []
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if 'USD_IDR' in df.columns and 'Tanggal' in df.columns:
                df = df.dropna(subset=['USD_IDR'])
                if len(df) > 0:
                    monthly_rate = round(float(df['USD_IDR'].iloc[-1]), 2)
                    monthly_date = str(df['Tanggal'].iloc[-1])
                    monthly_history = df['USD_IDR'].tail(12).tolist()
    except Exception:
        pass
    
    # Use daily data when available, otherwise retain the existing monthly fallback.
    latest = daily_rate if daily_rate else (monthly_rate if monthly_rate else 18050)
    change_pct = 0
    if daily_rate is not None and previous_rate:
        change_pct = ((daily_rate - previous_rate) / previous_rate) * 100
    elif monthly_history and len(monthly_history) >= 2:
        prev = monthly_history[-2]
        if prev > 0:
            change_pct = ((monthly_history[-1] - prev) / prev) * 100

    history = daily_history if daily_history else monthly_history
    
    return JsonResponse({
        'latest': latest,
        'daily_rate': daily_rate,
        'daily_date': daily_date,
        'previous_rate': previous_rate,
        'previous_date': previous_date,
        'monthly_rate': monthly_rate,
        'monthly_date': monthly_date,
        'change_pct': round(change_pct, 2),
        'history': history,
        'source': 'Frankfurter (central bank reference rates)' if daily_rate else 'BPS (monthly avg)',
        'data_type': 'daily' if daily_rate else 'monthly_avg'
    })


# ============================================================
# ARIMA MODEL
# ============================================================

ARIMA_MODEL = None
ARIMA_FORECAST = None

def load_arima():
    """Load ARIMA model and forecast data."""
    global ARIMA_MODEL, ARIMA_FORECAST
    
    project_root = os.path.dirname(settings.BASE_DIR)
    models_dir = os.path.join(project_root, 'models')
    
    arima_path = os.path.join(models_dir, 'arima_inflasi.pkl')
    forecast_path = os.path.join(models_dir, 'arima_forecast.pkl')
    
    if os.path.exists(arima_path) and ARIMA_MODEL is None:
        try:
            with open(arima_path, 'rb') as f:
                ARIMA_MODEL = pickle.load(f)
        except Exception:
            ARIMA_MODEL = None
    
    if os.path.exists(forecast_path) and ARIMA_FORECAST is None:
        try:
            with open(forecast_path, 'rb') as f:
                ARIMA_FORECAST = pickle.load(f)
        except Exception:
            ARIMA_FORECAST = None


def api_arima_forecast(request):
    """Return ARIMA forecast data."""
    load_arima()
    
    if ARIMA_FORECAST is None:
        return JsonResponse({
            'available': False,
            'message': 'ARIMA model belum di-train. Jalankan save_arima_model.py terlebih dahulu.'
        })
    
    return JsonResponse({
        'available': True,
        'forecast': ARIMA_FORECAST.get('forecast', {}),
        'order': str(ARIMA_FORECAST.get('order', 'N/A')),
        'last_date': ARIMA_FORECAST.get('last_date', 'N/A'),
        'last_value': ARIMA_FORECAST.get('last_value', 0)
    })


# ============================================================
# ENSEMBLE FORECAST API (LSTM + ARIMA + Prophet)
# ============================================================
ENSEMBLE_FORECAST = None
ENSEMBLE_METRICS = None

def load_ensemble():
    """Load ensemble forecast & metrics."""
    global ENSEMBLE_FORECAST, ENSEMBLE_METRICS
    project_root = os.path.dirname(settings.BASE_DIR)
    models_dir = os.path.join(project_root, 'models')
    
    forecast_path = os.path.join(models_dir, 'ensemble_forecast.pkl')
    metrics_path = os.path.join(models_dir, 'ensemble_metrics.pkl')
    
    if os.path.exists(forecast_path) and ENSEMBLE_FORECAST is None:
        try:
            with open(forecast_path, 'rb') as f:
                ENSEMBLE_FORECAST = pickle.load(f)
        except Exception:
            ENSEMBLE_FORECAST = None
    
    if os.path.exists(metrics_path) and ENSEMBLE_METRICS is None:
        try:
            with open(metrics_path, 'rb') as f:
                ENSEMBLE_METRICS = pickle.load(f)
        except Exception:
            ENSEMBLE_METRICS = None


def api_ensemble_forecast(request):
    """Return ensemble forecast (LSTM + ARIMA + Prophet) + comparison metrics."""
    load_ensemble()
    
    if ENSEMBLE_FORECAST is None:
        return JsonResponse({
            'available': False,
            'message': 'Ensemble model belum di-train. Jalankan train_ensemble.py terlebih dahulu.'
        })
    
    # Build comparison
    comparison = {}
    if ENSEMBLE_METRICS is not None:
        for m in ['naive', 'arima', 'lstm', 'prophet', 'ensemble']:
            if m in ENSEMBLE_METRICS:
                r = ENSEMBLE_METRICS[m]
                comparison[m] = {
                    'mae': round(r.get('mae', 0), 4),
                    'rmse': round(r.get('rmse', 0), 4),
                    'smape': round(r.get('smape', 0), 2),
                    'n_test': r.get('n_test', 0)
                }
    
    return JsonResponse({
        'available': True,
        'forecast': {
            'lstm': ENSEMBLE_FORECAST.get('lstm_forecast', []),
            'arima': ENSEMBLE_FORECAST.get('arima_forecast', []),
            'prophet': ENSEMBLE_FORECAST.get('prophet_forecast', []),
            'ensemble': ENSEMBLE_FORECAST.get('ensemble_forecast', [])
        },
        'weights': ENSEMBLE_FORECAST.get('weights', {}),
        'last_date': ENSEMBLE_FORECAST.get('last_date', 'N/A'),
        'last_value': ENSEMBLE_FORECAST.get('last_value', 0),
        'comparison': comparison,
        'best_model': 'ensemble' if comparison.get('ensemble', {}).get('mae', 99) < min(
            [comparison.get(m, {}).get('mae', 99) for m in ['arima', 'lstm', 'prophet']]
        ) else 'individual'
    })


# ============================================================
# INFLASI SUMMARY API (M-to-M, Y-o-Y, Y-to-D)
# ============================================================

INFLASI_SUMMARY_CACHE = None

def api_inflasi_summary(request):
    """Return ringkasan inflasi: M-to-M, Y-o-Y, Y-to-D, dan histori 24 bulan."""
    global INFLASI_SUMMARY_CACHE
    
    if INFLASI_SUMMARY_CACHE is not None:
        return JsonResponse(INFLASI_SUMMARY_CACHE)
    
    project_root = os.path.dirname(settings.BASE_DIR)
    data_path = os.path.join(project_root, 'datasets', 'processed', 'clean_inflasi_ts.csv')
    
    if not os.path.exists(data_path):
        return JsonResponse({'error': 'Data file not found'}, status=404)
    
    try:
        df = pd.read_csv(data_path, parse_dates=['Tanggal'])
        df = df.sort_values('Tanggal').reset_index(drop=True)
        
        # Data bulan terakhir
        latest = df.iloc[-1]
        last_date = latest['Tanggal'].strftime('%Y-%m-%d')
        
        # M-to-M: perubahan dari bulan lalu
        prev = df.iloc[-2]
        mom_change = float(latest['Inflasi_MoM'] - prev['Inflasi_MoM'])
        
        # Y-o-Y: sudah di-preprocess sebagai kolom Inflasi_YoY
        yoy = float(latest.get('Inflasi_YoY', 0)) if not pd.isna(latest.get('Inflasi_YoY')) else None
        ytd = float(latest.get('Inflasi_YtD', 0)) if not pd.isna(latest.get('Inflasi_YtD')) else None
        
        # Y-o-Y bulan lalu untuk perbandingan
        yoy_prev = float(prev.get('Inflasi_YoY', 0)) if not pd.isna(prev.get('Inflasi_YoY')) else None
        yoy_change = (yoy - yoy_prev) if (yoy is not None and yoy_prev is not None) else None
        
        # Y-o-Y setahun lalu (12 bulan lalu) untuk konteks
        if len(df) >= 13:
            year_ago = df.iloc[-13]
            yoy_year_ago = float(year_ago.get('Inflasi_YoY', 0)) if not pd.isna(year_ago.get('Inflasi_YoY')) else None
        else:
            yoy_year_ago = None
        
        # Histori 24 bulan terakhir
        recent = df.tail(24).copy()
        history = {
            'labels': recent['Tanggal'].dt.strftime('%b %Y').tolist(),
            'mom': recent['Inflasi_MoM'].round(2).tolist(),
            'yoy': [round(float(v), 2) if not pd.isna(v) else None 
                    for v in recent.get('Inflasi_YoY', [None]*len(recent))],
            'ytd': [round(float(v), 2) if not pd.isna(v) else None 
                    for v in recent.get('Inflasi_YtD', [None]*len(recent))]
        }
        
        # Statistik ringkasan
        full_yoy = df['Inflasi_YoY'].dropna()
        stats = {
            'yoy_mean_12m': round(float(full_yoy.tail(12).mean()), 2) if len(full_yoy) >= 12 else None,
            'yoy_min_12m': round(float(full_yoy.tail(12).min()), 2) if len(full_yoy) >= 12 else None,
            'yoy_max_12m': round(float(full_yoy.tail(12).max()), 2) if len(full_yoy) >= 12 else None
        }
        
        # Status klasifikasi
        if yoy is not None:
            if yoy < 2.5:
                status = 'Terkendali'
                status_color = 'positive'
            elif yoy < 4.0:
                status = 'Waspada'
                status_color = 'warning'
            else:
                status = 'Tinggi'
                status_color = 'negative'
        else:
            status = 'Tidak tersedia'
            status_color = 'neutral'
        
        INFLASI_SUMMARY_CACHE = {
            'as_of': last_date,
            'mom': {
                'value': round(float(latest['Inflasi_MoM']), 2),
                'change': round(mom_change, 2),
                'description': 'Month-to-Month (bulanan)'
            },
            'yoy': {
                'value': round(yoy, 2) if yoy is not None else None,
                'change': round(yoy_change, 2) if yoy_change is not None else None,
                'year_ago': round(yoy_year_ago, 2) if yoy_year_ago is not None else None,
                'description': 'Year-on-Year (vs 12 bulan lalu)'
            },
            'ytd': {
                'value': round(ytd, 2) if ytd is not None else None,
                'description': 'Year-to-Date (vs Januari tahun ini)'
            },
            'status': status,
            'status_color': status_color,
            'stats': stats,
            'history': history
        }
        
        return JsonResponse(INFLASI_SUMMARY_CACHE)
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


# ============================================================
# MAP PAGE
# ============================================================

def map_page(request):
    """Indonesia choropleth map page."""
    return render(request, 'predictions/map.html')
