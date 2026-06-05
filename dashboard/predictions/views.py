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
RIDGE_MODEL_BUNDLE = None
INFLASI_PRED_MEM = None
DATA_HISTORIS = None
LSTM_FEATURES = None
RATA_PENGELUARAN = 1450000

def load_models():
    global LSTM_MODEL, LSTM_SCALER_X, LSTM_SCALER_Y, RIDGE_MODEL, RIDGE_MODEL_BUNDLE, INFLASI_PRED_MEM, DATA_HISTORIS, LSTM_FEATURES, RATA_PENGELUARAN
    
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
            RIDGE_MODEL_BUNDLE = None
            
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
    """
    Bangun dummy input untuk Ridge model.
    Menggunakan metadata dari model bundle jika tersedia, agar otomatis support fitur baru.
    """
    # Default base values
    base_values = {
        'Real_UMP': 3000000.0 / (1 + inflasi_val),
        'TPT': 5.5,
        'PDRB_HargaKonstan': 500000.0,
        'Inflasi_Rata_Tahunan': inflasi_val,
        'Provinsi': 'JAWA TIMUR',
        # World Bank features (nasional, gunakan nilai default tengah)
        'Inflasi_WB_Annual': 3.5,  # ~inflasi normal Indonesia
        'GDP_PerCapita_PPP': 13000.0,  # PDB per kapita tengah
        'Pct_Unemployment_WB': 5.0,  # pengangguran normal
        'Poverty_Headcount_Pct': 10.0,  # kemiskinan rata-rata
    }
    
    # Jika model bundle punya info fitur, tambahkan default values untuk fitur lain
    if RIDGE_MODEL_BUNDLE is not None and 'num_features' in RIDGE_MODEL_BUNDLE:
        try:
            num_features = RIDGE_MODEL_BUNDLE.get('num_features', [])
            for feat in num_features:
                if feat not in base_values:
                    # Default values untuk fitur yang belum ter-handle
                    defaults_map = {
                        'Gini_Rasio': 0.35,
                        'IPM': 72.0,
                        'Garis_Kemiskinan': 500000.0,
                        'Jumlah_Penduduk': 5000000.0,
                        'Pct_Populasi': 2.5,  # % share of total population
                        'Pct_Akses_Air_Bersih': 80.0,
                        'Protein_gram_per_hari': 60.0,
                        'Jumlah_Rumah_Tangga': 1500000.0,
                    }
                    base_values[feat] = defaults_map.get(feat, 0.0)
        except Exception:
            pass
    
    return pd.DataFrame([base_values])

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


# --- Landing Page (new) ---
def home_page(request):
    return render(request, 'predictions/home.html')


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

def api_dataset_sample(request):
    """Return sample rows and column info from processed CSV files."""
    dataset = request.GET.get('dataset', 'daya_beli')
    n_rows = int(request.GET.get('n', 8))
    project_root = os.path.dirname(settings.BASE_DIR)
    
    if dataset == 'daya_beli':
        path = os.path.join(project_root, 'datasets', 'processed', 'clean_daya_beli.csv')
    elif dataset == 'inflasi':
        path = os.path.join(project_root, 'datasets', 'processed', 'clean_inflasi_ts.csv')
    else:
        return JsonResponse({'error': 'Unknown dataset'}, status=400)
    
    if not os.path.exists(path):
        return JsonResponse({'error': 'File not found'}, status=404)
    
    try:
        df = pd.read_csv(path)
        # Get column names and types
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
        
        # Sample rows (first n_rows)
        rows = df.head(n_rows).fillna('').astype(str).to_dict('records')
        
        return JsonResponse({
            'columns': col_names,
            'types': col_types,
            'rows': rows,
            'total_rows': len(df),
            'dataset': dataset
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
    """Return latest USD/IDR exchange rate with daily + monthly data."""
    import urllib.request
    import json as json_lib
    from datetime import datetime
    
    project_root = os.path.dirname(settings.BASE_DIR)
    path = os.path.join(project_root, 'datasets', 'processed', 'clean_inflasi_ts.csv')
    
    # 1. Try to fetch daily rate from external API
    daily_rate = None
    daily_date = None
    try:
        url = "https://open.er-api.com/v6/latest/USD"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json_lib.loads(resp.read().decode())
            if data.get('result') == 'success' and 'IDR' in data.get('rates', {}):
                daily_rate = round(data['rates']['IDR'], 2)
                daily_date = data.get('time_last_update_utc', datetime.now().strftime('%Y-%m-%d'))
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
    
    # 3. Prepare response
    # Use daily rate if available, otherwise fallback to monthly
    latest = daily_rate if daily_rate else (monthly_rate if monthly_rate else 18050)
    date_str = daily_date if daily_date else (monthly_date if monthly_date else 'N/A')
    
    # Calculate change from monthly data
    change_pct = 0
    if monthly_history and len(monthly_history) >= 2:
        prev = monthly_history[-2]
        if prev > 0:
            change_pct = ((monthly_history[-1] - prev) / prev) * 100
    
    return JsonResponse({
        'latest': latest,
        'daily_rate': daily_rate,
        'daily_date': daily_date,
        'monthly_rate': monthly_rate,
        'monthly_date': monthly_date,
        'change_pct': round(change_pct, 2),
        'history': monthly_history,
        'source': 'open.er-api.com' if daily_rate else 'BPS (monthly avg)',
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
# MAP PAGE
# ============================================================

def map_page(request):
    """Indonesia choropleth map page."""
    return render(request, 'predictions/map.html')
