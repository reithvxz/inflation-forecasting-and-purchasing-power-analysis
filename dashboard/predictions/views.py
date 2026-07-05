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
from predictions.daya_beli_model import (
    DERIVED_NUMERIC_FEATURES,
    NOMINAL_TARGET_COLUMN,
    TARGET_COLUMN,
    TARGET_LABEL,
    prepare_daya_beli_dataframe,
)
from predictions.inflation_forecast import (
    SARIMAX_REGRESSOR_SHORTLIST,
    inflation_dataset_path,
    load_saved_forecast_payload,
    load_saved_sarimax_feature_audit,
)

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
INFLATION_FORECAST_PAYLOAD = None
SARIMAX_FEATURE_AUDIT = None

SARIMAX_FEATURE_REASON_MAP = {
    "USD_IDR": {
        "label": "USD/IDR",
        "reason": "Menangkap pass-through kurs terhadap harga impor, bahan baku, dan biaya produksi.",
    },
    "Brent_USD": {
        "label": "Brent Oil",
        "reason": "Merepresentasikan tekanan biaya energi dan logistik yang dapat merambat ke harga domestik.",
    },
    "BI_Rate": {
        "label": "BI Rate",
        "reason": "Mewakili kanal kebijakan moneter dan respons suku bunga terhadap tekanan inflasi.",
    },
    "DXY": {
        "label": "US Dollar Index",
        "reason": "Memberi konteks kekuatan dolar global, terpisah dari pergerakan rupiah yang spesifik ke Indonesia.",
    },
    "FAO_FPI": {
        "label": "FAO Food Price Index",
        "reason": "Mewakili tekanan harga pangan global, yang penting untuk struktur inflasi Indonesia.",
    },
}

RIDGE_FEATURE_REASON_MAP = {
    "UMP": {
        "label": "Upah minimum provinsi",
        "reason": "Menangkap level upah nominal yang menjadi titik awal pembentukan daya beli riil.",
    },
    "Real_UMP": {
        "label": "Upah minimum riil",
        "reason": "Merepresentasikan kapasitas upah setelah dikoreksi tekanan inflasi tahunan.",
    },
    "TPT": {
        "label": "Tingkat pengangguran terbuka",
        "reason": "Menggambarkan tekanan pasar kerja yang berpengaruh pada ruang belanja rumah tangga.",
    },
    "TPAK": {
        "label": "Tingkat partisipasi angkatan kerja",
        "reason": "Memberi konteks kedalaman partisipasi tenaga kerja di setiap wilayah.",
    },
    "PDRB_HargaKonstan": {
        "label": "PDRB harga konstan",
        "reason": "Mewakili kapasitas ekonomi riil daerah yang menopang konsumsi per kapita.",
    },
    "PDRB_HargaBerlaku": {
        "label": "PDRB harga berlaku",
        "reason": "Menambah konteks level ekonomi nominal antarwilayah.",
    },
    "Inflasi_Rata_Tahunan": {
        "label": "Inflasi rata-rata tahunan",
        "reason": "Mewakili tekanan harga domestik yang memengaruhi konsumsi riil rumah tangga.",
    },
    "Inflasi_WB_Annual": {
        "label": "Inflasi tahunan acuan",
        "reason": "Menjadi sumber inflasi tahunan utama untuk pembentukan deflator pada artefak aktif.",
    },
    "Inflation_Deflator": {
        "label": "Deflator inflasi",
        "reason": "Mengubah pengeluaran nominal menjadi pengeluaran riil yang lebih relevan untuk interpretasi daya beli.",
    },
    "Prev_Total_Pengeluaran_Riil": {
        "label": "Pengeluaran riil tahun sebelumnya",
        "reason": "Memberi jangkar historis agar model tidak kehilangan konteks level konsumsi terakhir.",
    },
    "Real_UMP_Growth": {
        "label": "Pertumbuhan upah riil",
        "reason": "Menangkap perubahan tahunan daya dorong upah setelah koreksi inflasi.",
    },
    "PDRB_HargaKonstan_Growth": {
        "label": "Pertumbuhan PDRB riil",
        "reason": "Mengukur arah perubahan kapasitas ekonomi daerah dari tahun sebelumnya.",
    },
    "TPT_Growth": {
        "label": "Perubahan TPT",
        "reason": "Menambah sinyal perubahan kondisi pasar kerja, bukan hanya levelnya.",
    },
    "UMP_x_PDRB": {
        "label": "Interaksi upah dan PDRB",
        "reason": "Membaca apakah efek upah menjadi berbeda pada wilayah dengan skala ekonomi yang berbeda.",
    },
    "Inflasi_x_TPT": {
        "label": "Interaksi inflasi dan TPT",
        "reason": "Menggambarkan tekanan gabungan antara kenaikan harga dan pelemahan pasar kerja.",
    },
    "Log_PDRB": {
        "label": "Log PDRB",
        "reason": "Menstabilkan skala PDRB agar hubungan non-linear lebih mudah dibaca model linear.",
    },
    "Log_UMP": {
        "label": "Log upah riil",
        "reason": "Menstabilkan skala upah dan membantu model membaca perubahan relatif.",
    },
    "Gini_Rasio": {
        "label": "Rasio gini",
        "reason": "Memberi konteks distribusi pengeluaran yang dapat memengaruhi rata-rata konsumsi per kapita.",
    },
    "IPM": {
        "label": "Indeks pembangunan manusia",
        "reason": "Mewakili kualitas pembangunan manusia yang berkorelasi dengan struktur konsumsi wilayah.",
    },
    "Pct_Penduduk_Miskin": {
        "label": "Persentase penduduk miskin",
        "reason": "Menggambarkan tekanan kesejahteraan yang relevan terhadap konsumsi rata-rata.",
    },
    "Garis_Kemiskinan": {
        "label": "Garis kemiskinan",
        "reason": "Memberi titik acuan biaya minimum hidup antarwilayah.",
    },
    "Jumlah_Penduduk": {
        "label": "Jumlah penduduk",
        "reason": "Menambah konteks ukuran wilayah dan skala rumah tangga yang terwakili.",
    },
    "Pct_Populasi": {
        "label": "Proporsi populasi",
        "reason": "Menunjukkan bobot penduduk relatif antarwilayah pada data aktif.",
    },
    "Pct_Akses_Air_Bersih": {
        "label": "Akses air bersih",
        "reason": "Menjadi proksi kualitas layanan dasar yang ikut memengaruhi profil kesejahteraan wilayah.",
    },
    "Protein_gram_per_hari": {
        "label": "Konsumsi protein harian",
        "reason": "Menambah konteks kualitas konsumsi rumah tangga, bukan hanya nominal pengeluaran.",
    },
    "GDP_PerCapita_PPP": {
        "label": "GDP per capita PPP",
        "reason": "Memberi referensi daya beli makro yang bisa membantu pembacaan skala konsumsi antarwilayah.",
    },
    "Pct_Unemployment_WB": {
        "label": "Pengangguran acuan internasional",
        "reason": "Menambah pembanding eksternal untuk pasar kerja ketika data tersedia.",
    },
    "Poverty_Headcount_Pct": {
        "label": "Poverty headcount",
        "reason": "Memberi sinyal kemiskinan relatif sebagai pembanding tambahan lintas sumber.",
    },
    "Year_Index": {
        "label": "Indeks tahun",
        "reason": "Menjaga jejak waktu agar model bisa membaca pergeseran antarperiode secara eksplisit.",
    },
}


def _safe_float(value, fallback=0.0):
    try:
        if value is None or pd.isna(value):
            return float(fallback)
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _json_no_store(payload, status=200):
    response = JsonResponse(payload, status=status)
    response["Cache-Control"] = "no-store, no-cache, max-age=0, must-revalidate"
    response["Pragma"] = "no-cache"
    response["Expires"] = "0"
    return response


def _get_feature_note(feature_name):
    return {
        "feature": feature_name,
        "label": (SARIMAX_FEATURE_REASON_MAP.get(feature_name) or {}).get("label", feature_name),
        "reason": (SARIMAX_FEATURE_REASON_MAP.get(feature_name) or {}).get(
            "reason",
            "Fitur ini ikut dipakai oleh artefak aktif dan tetap diaudit kontribusinya secara out-of-sample.",
        ),
    }


def _get_active_sarimax_feature_keys():
    payload = _get_sarimax_feature_audit_payload() or {}
    horizons = (payload.get("horizons") or {}).values()
    discovered = []
    for horizon_payload in horizons:
        for feature in horizon_payload.get("base_regressors", []):
            if feature not in discovered:
                discovered.append(feature)
    return discovered or list(SARIMAX_REGRESSOR_SHORTLIST)


def _get_sarimax_feature_notes():
    return [_get_feature_note(feature_name) for feature_name in _get_active_sarimax_feature_keys()]


def _get_model_family_notes():
    active_sarimax_labels = [item["label"] for item in _get_sarimax_feature_notes()]
    sarimax_inputs = ", ".join(active_sarimax_labels) if active_sarimax_labels else "regressor makroekonomi aktif"

    return [
        {
            "name": "ARIMA",
            "family": "Statistical time-series",
            "inputs": "Hanya seri target Inflasi_MoM historis.",
            "when_used": "Menjadi baseline kuat saat pola utama cukup tertangkap dari histori inflasi itu sendiri.",
            "strength": "Sederhana, mudah diaudit, dan sering kompetitif bila sinyal exogenous tambahan tidak terlalu kuat.",
        },
        {
            "name": "SARIMAX",
            "family": "Econometric time-series with exogenous regressors",
            "inputs": f"Seri target Inflasi_MoM ditambah regressor aktif pada artefak saat ini: {sarimax_inputs}.",
            "when_used": "Dipakai saat tekanan inflasi diduga juga dipengaruhi shock kurs, energi, pangan global, dan kebijakan moneter.",
            "strength": "Menjaga interpretabilitas sambil tetap menangkap pengaruh variabel eksternal yang relevan.",
        },
        {
            "name": "LSTM / Bi-LSTM",
            "family": "Sequence deep learning",
            "inputs": "Riwayat beberapa langkah waktu dengan feature space yang lebih luas dibanding model statistik.",
            "when_used": "Berguna saat pola non-linear dan interaksi antarfitur waktu lebih kompleks daripada yang mudah ditangkap model klasik.",
            "strength": "Lebih fleksibel untuk pola non-linear, tetapi validasinya harus ketat karena lebih mudah overfit.",
        },
        {
            "name": "Prophet",
            "family": "Additive trend-seasonality model with regressors",
            "inputs": "Tanggal, seri target, dan regressor tambahan yang sama dengan shortlist exogenous aktif.",
            "when_used": "Cocok untuk baseline yang eksplisit memisahkan tren, musiman, dan efek regressor.",
            "strength": "Cepat dibaca, relatif stabil, dan berguna sebagai pembanding yang transparan.",
        },
    ]


def _get_ridge_feature_note(feature_name):
    info = RIDGE_FEATURE_REASON_MAP.get(feature_name, {})
    return {
        "feature": feature_name,
        "label": info.get("label", feature_name.replace("_", " ")),
        "reason": info.get(
            "reason",
            "Fitur ini dipertahankan pada artefak aktif karena masih memberi sinyal terhadap estimasi pengeluaran riil per kapita.",
        ),
        "group": "derived" if feature_name in DERIVED_NUMERIC_FEATURES else "core",
    }


def _build_ridge_model_guide_context():
    load_models(load_inflation=False)
    bundle = RIDGE_MODEL_BUNDLE or {}
    if not bundle or bundle.get("legacy_artifact"):
        return {
            "available": False,
            "message": "Artefak model daya beli aktif belum menyediakan metadata teknis yang lengkap untuk halaman panduan.",
        }

    selected_features = [_get_ridge_feature_note(feature) for feature in bundle.get("num_features", [])]
    core_features = [item for item in selected_features if item["group"] == "core"]
    derived_features = [item for item in selected_features if item["group"] == "derived"]
    split_strategy = bundle.get("split_strategy") or {}
    data_scope = bundle.get("data_scope") or {}
    walk_forward = (bundle.get("walk_forward") or {}).get("mean") or {}

    return {
        "available": True,
        "target_label": bundle.get("target_label", TARGET_LABEL),
        "target_type": bundle.get("target_type", "real"),
        "core_features": core_features,
        "derived_features": derived_features,
        "categorical_features": [
            {
                "feature": "Provinsi",
                "label": "Wilayah",
                "reason": "Kategori wilayah membantu model membedakan karakter dasar masing-masing provinsi dan agregat nasional Indonesia.",
            }
        ],
        "data_scope": data_scope,
        "split_strategy": split_strategy,
        "validation_strategy": bundle.get("validation_strategy") or {},
        "test_metrics": {
            "r2": round(_safe_float(bundle.get("test_r2")), 3),
            "mae": round(_safe_float(bundle.get("test_mae")), 0),
            "rmse": round(_safe_float(bundle.get("test_rmse")), 0),
            "smape": round(_safe_float(bundle.get("test_smape")), 2),
        },
        "walk_forward": {
            "r2": round(_safe_float(walk_forward.get("r2")), 3),
            "mae": round(_safe_float(walk_forward.get("mae")), 0),
            "rmse": round(_safe_float(walk_forward.get("rmse")), 0),
            "smape": round(_safe_float(walk_forward.get("smape")), 2),
        },
        "model_note": bundle.get("model_note", ""),
        "limitations": [
            f"Cakupan data aktif berada pada {int(_safe_float(data_scope.get('year_min'), 2021))}-{int(_safe_float(data_scope.get('year_max'), 2025))} dengan observasi tahunan lintas wilayah, sehingga resolusi waktunya memang tidak setajam model inflasi bulanan.",
            "Output paling tepat dibaca sebagai estimasi pengeluaran riil per kapita per bulan yang dipakai sebagai proksi daya beli, bukan ukuran daya beli teoritis murni.",
            "Nilai model lebih kuat untuk membaca arah perubahan antar-skenario dibanding mengejar presisi nominal pada level yang sangat detail.",
        ],
    }


def _get_sarimax_feature_audit_payload():
    global SARIMAX_FEATURE_AUDIT
    if SARIMAX_FEATURE_AUDIT is None:
        project_root = os.path.dirname(settings.BASE_DIR)
        SARIMAX_FEATURE_AUDIT = load_saved_sarimax_feature_audit(project_root)
    return SARIMAX_FEATURE_AUDIT


def _build_sarimax_feature_audit_context():
    payload = _get_sarimax_feature_audit_payload()
    notes = _get_sarimax_feature_notes()
    note_lookup = {item["feature"]: item for item in notes}

    if not payload:
        return {
            "available": False,
            "notes": notes,
            "message": "Artefak audit fitur SARIMAX belum tersedia. Jalankan train_inflation_multihorizon.py untuk menghasilkan hasil ablation out-of-sample terbaru.",
        }

    horizons = []
    for horizon_key, horizon_payload in (payload.get("horizons") or {}).items():
        rows = []
        for row in horizon_payload.get("drop_one_tests", []):
            note = note_lookup.get(row.get("feature"), {})
            rows.append(
                {
                    "feature": row.get("feature"),
                    "label": note.get("label", row.get("feature")),
                    "reason": note.get("reason", ""),
                    "status": row.get("status", "skipped"),
                    "dropped_mae": ((row.get("dropped_model_metrics") or {}).get("mae")),
                    "delta_mae": row.get("delta_mae"),
                    "interpretation": row.get("interpretation") or row.get("reason") or "Audit belum tersedia.",
                }
            )
        horizons.append(
            {
                "key": horizon_key,
                "label": horizon_payload.get("label", horizon_key),
                "base_regressors": horizon_payload.get("base_regressors", []),
                "base_metrics": horizon_payload.get("base_metrics", {}),
                "rows": rows,
            }
        )

    return {
        "available": True,
        "generated_at": payload.get("generated_at"),
        "methodology": payload.get("methodology", {}),
        "notes": notes,
        "horizons": horizons,
    }


def _pct_change(current, previous, fallback=0.0):
    current_val = _safe_float(current, fallback)
    previous_val = _safe_float(previous, 0.0)
    if previous_val == 0:
        return _safe_float(fallback, 0.0)
    return ((current_val - previous_val) / previous_val) * 100.0


def _format_compact_rupiah(value):
    numeric = _safe_float(value, 0.0)
    if abs(numeric) >= 1_000_000_000:
        return f"Rp {numeric / 1_000_000_000:.2f} M"
    if abs(numeric) >= 1_000_000:
        return f"Rp {numeric / 1_000_000:.2f} jt"
    return f"Rp {numeric:,.0f}".replace(",", ".")


def _format_compact_number(value):
    numeric = _safe_float(value, 0.0)
    if abs(numeric) >= 1_000_000_000:
        return f"{numeric / 1_000_000_000:.2f} M"
    if abs(numeric) >= 1_000_000:
        return f"{numeric / 1_000_000:.2f} jt"
    return f"{numeric:,.0f}".replace(",", ".")


def _format_percent(value, digits=2):
    numeric = _safe_float(value, 0.0)
    return f"{numeric:.{digits}f}%"


def _format_decimal(value, digits=2):
    numeric = _safe_float(value, 0.0)
    return f"{numeric:.{digits}f}"


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
        'RATA_PENGELUARAN': float(numeric_defaults.get(TARGET_COLUMN, 1450000.0)),
        'numeric_defaults': numeric_defaults,
    }


def _build_province_simulation_baselines(project_root):
    path = os.path.join(project_root, 'datasets', 'processed', 'clean_daya_beli.csv')
    if not os.path.exists(path):
        return {}

    df = prepare_daya_beli_dataframe(pd.read_csv(path))
    if df.empty:
        return {}

    def _aggregate_national_row(frame):
        if frame.empty:
            return None
        numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()
        weights = frame.get('Jumlah_Penduduk')
        use_weights = weights is not None and weights.notna().any() and float(weights.fillna(0).sum()) > 0

        aggregated = {'Provinsi': 'Indonesia'}
        for column in numeric_columns:
            series = frame[column].astype(float)
            valid_mask = series.notna()
            if not valid_mask.any():
                continue
            if use_weights:
                local_weights = weights[valid_mask].fillna(0).astype(float)
                if float(local_weights.sum()) > 0:
                    aggregated[column] = float(np.average(series[valid_mask], weights=local_weights))
                    continue
            aggregated[column] = float(series[valid_mask].mean())
        return aggregated

    latest_rows = df.groupby('Provinsi').tail(1).copy()
    baselines = {}
    for _, row in latest_rows.iterrows():
        province = row.get('Provinsi')
        if not province:
            continue
        row_dict = row.to_dict()
        baselines[province] = {
            'baseline_year': int(_safe_float(row_dict.get('Tahun'), 0)),
            'baseline_pengeluaran': _safe_float(row_dict.get(TARGET_COLUMN), 1450000.0),
            'baseline_pengeluaran_nominal': _safe_float(row_dict.get(NOMINAL_TARGET_COLUMN), 1450000.0),
            'fields': row_dict,
        }

    latest_year = int(df['Tahun'].max())
    national_latest = _aggregate_national_row(df[df['Tahun'] == latest_year].copy())
    if national_latest is not None:
        previous_year = latest_year - 1
        national_previous = _aggregate_national_row(df[df['Tahun'] == previous_year].copy()) or {}
        for key in (
            'Prev_Real_UMP',
            'Prev_PDRB_HargaKonstan',
            'Prev_TPT',
            'Prev_Total_Pengeluaran_Riil',
            'Real_UMP_Growth',
            'PDRB_HargaKonstan_Growth',
            'TPT_Growth',
        ):
            if key not in national_latest and key in national_previous:
                national_latest[key] = national_previous[key]
        national_latest['Tahun'] = latest_year
        baselines['Indonesia'] = {
            'baseline_year': latest_year,
            'baseline_pengeluaran': _safe_float(national_latest.get(TARGET_COLUMN), 1450000.0),
            'baseline_pengeluaran_nominal': _safe_float(national_latest.get(NOMINAL_TARGET_COLUMN), 1450000.0),
            'fields': national_latest,
        }
    return baselines


def _get_province_simulation_baselines():
    global PROVINCE_SIMULATION_BASELINES
    if PROVINCE_SIMULATION_BASELINES is None:
        project_root = os.path.dirname(settings.BASE_DIR)
        PROVINCE_SIMULATION_BASELINES = _build_province_simulation_baselines(project_root)
    return PROVINCE_SIMULATION_BASELINES or {}


def _get_actual_province_count():
    baselines = _get_province_simulation_baselines()
    return len([name for name in baselines.keys() if name != 'Indonesia'])


SCENARIO_LIBRARY = {
    'inflation_shock': {
        'title': 'Tekanan inflasi tahunan naik 1,5 poin',
        'description': 'Skenario ini membaca dampak ketika tekanan harga meningkat sementara variabel pendapatan dan aktivitas ekonomi mengikuti baseline aktif.',
        'accent': 'amber',
        'assumptions': [
            'Inflasi tahunan +1,5 poin',
            'UMP mengikuti baseline wilayah',
            'TPT tetap',
            'PDRB harga konstan tetap',
        ],
        'overrides': {'inflasi_delta': 1.5},
        'interpretation_template': 'Kenaikan tekanan harga tanpa penyesuaian pendapatan biasanya mempersempit ruang konsumsi riil, sehingga proksi daya beli cenderung {direction}.',
    },
    'income_support': {
        'title': 'Kompensasi pendapatan: UMP naik 8%',
        'description': 'Skenario ini membaca respons proksi daya beli ketika pendapatan minimum wilayah menguat, dengan tekanan harga mengikuti baseline aktif.',
        'accent': 'teal',
        'assumptions': [
            'UMP +8%',
            'Inflasi tahunan tetap',
            'TPT tetap',
            'PDRB harga konstan tetap',
        ],
        'overrides': {'ump_multiplier': 1.08},
        'interpretation_template': 'Penguatan pendapatan minimum memberi bantalan pada konsumsi riil, sehingga proksi daya beli nasional cenderung {direction}.',
    },
    'balanced_support': {
        'title': 'Pendapatan menguat, inflasi ikut naik',
        'description': 'Skenario gabungan untuk menguji apakah penguatan pendapatan masih mampu menutup tambahan tekanan harga dalam horizon pendek.',
        'accent': 'blue',
        'assumptions': [
            'UMP +8%',
            'Inflasi tahunan +1,0 poin',
            'TPT tetap',
            'PDRB harga konstan tetap',
        ],
        'overrides': {'ump_multiplier': 1.08, 'inflasi_delta': 1.0},
        'interpretation_template': 'Ketika pendapatan dan inflasi naik bersama, hasil akhir bergantung pada apakah pertumbuhan pendapatan bersih masih cukup untuk menjaga konsumsi riil. Dalam skenario aktif, arah nasional cenderung {direction}.',
    },
    'growth_slowdown': {
        'title': 'Perlambatan pertumbuhan wilayah',
        'description': 'Skenario ini menekan output riil wilayah dan menaikkan TPT untuk membaca tekanan simultan dari sisi pertumbuhan dan pasar tenaga kerja.',
        'accent': 'violet',
        'assumptions': [
            'PDRB harga konstan -8%',
            'TPT +0,8 poin',
            'Inflasi tahunan tetap',
            'UMP tetap',
        ],
        'overrides': {'pdrb_multiplier': 0.92, 'tpt_delta': 0.8},
        'interpretation_template': 'Ketika output riil melemah dan pengangguran naik, ruang konsumsi rumah tangga biasanya tertekan. Dalam simulasi aktif, proksi daya beli nasional cenderung {direction}.',
    },
}


def _build_scenario_overrides(province, scenario_spec):
    baselines = _get_province_simulation_baselines()
    baseline_fields = (baselines.get(province) or {}).get('fields') or {}
    scenario_rules = scenario_spec.get('overrides') or {}
    overrides = {}

    baseline_annual_inflation = _safe_float(
        baseline_fields.get('Inflasi_WB_Annual'),
        baseline_fields.get('Inflasi_Rata_Tahunan'),
    )
    baseline_ump = _safe_float(baseline_fields.get('UMP'), 0.0)
    baseline_tpt = _safe_float(baseline_fields.get('TPT'), 0.0)
    baseline_pdrb = _safe_float(baseline_fields.get('PDRB_HargaKonstan'), 0.0)

    if 'inflasi_abs' in scenario_rules:
        overrides['inflasi'] = _safe_float(scenario_rules.get('inflasi_abs'))
    elif 'inflasi_delta' in scenario_rules:
        overrides['inflasi'] = baseline_annual_inflation + _safe_float(scenario_rules.get('inflasi_delta'))

    if 'ump_multiplier' in scenario_rules:
        overrides['ump'] = max(0.0, baseline_ump * _safe_float(scenario_rules.get('ump_multiplier'), 1.0))
    elif 'ump_delta_pct' in scenario_rules:
        overrides['ump'] = max(0.0, baseline_ump * (1 + (_safe_float(scenario_rules.get('ump_delta_pct')) / 100.0)))

    if 'tpt_abs' in scenario_rules:
        overrides['tpt'] = max(0.0, _safe_float(scenario_rules.get('tpt_abs')))
    elif 'tpt_delta' in scenario_rules:
        overrides['tpt'] = max(0.0, baseline_tpt + _safe_float(scenario_rules.get('tpt_delta')))

    if 'pdrb_multiplier' in scenario_rules:
        overrides['pdrb_hargakonstan'] = max(0.0, baseline_pdrb * _safe_float(scenario_rules.get('pdrb_multiplier'), 1.0))
    elif 'pdrb_delta_pct' in scenario_rules:
        overrides['pdrb_hargakonstan'] = max(0.0, baseline_pdrb * (1 + (_safe_float(scenario_rules.get('pdrb_delta_pct')) / 100.0)))

    return overrides


def _predict_simulation_value(province, overrides=None):
    dummy_input, baseline = _build_simulation_input(province, overrides or {})
    predicted_value = max(float(RIDGE_MODEL.predict(dummy_input)[0]), 0.0)
    return predicted_value, baseline, dummy_input


def _scenario_direction_label(change_pct):
    if change_pct >= 2.5:
        return 'menguat'
    if change_pct <= -2.5:
        return 'melemah'
    return 'relatif stabil'


def _build_simulation_input(province, overrides=None):
    load_models(load_inflation=False)
    baselines = _get_province_simulation_baselines()
    if province not in baselines:
        raise KeyError(f"Provinsi '{province}' tidak ditemukan")

    baseline = baselines[province]
    fields = baseline['fields']
    overrides = overrides or {}

    baseline_local_inflation = _safe_float(fields.get('Inflasi_Rata_Tahunan'), 0.0)
    baseline_annual_inflation = _safe_float(
        fields.get('Inflasi_WB_Annual'),
        baseline_local_inflation,
    )
    scenario_annual_inflation = _safe_float(
        overrides.get('inflasi'),
        baseline_annual_inflation,
    )
    if abs(baseline_annual_inflation) > 1e-9:
        local_to_annual_ratio = baseline_local_inflation / baseline_annual_inflation
        scenario_local_inflation = scenario_annual_inflation * local_to_annual_ratio
    else:
        scenario_local_inflation = _safe_float(
            overrides.get('inflasi'),
            baseline_local_inflation,
        )

    ump_value = _safe_float(overrides.get('ump'), fields.get('UMP', 3000000.0))
    tpt_value = _safe_float(overrides.get('tpt'), fields.get('TPT', 5.0))
    pdrb_value = _safe_float(overrides.get('pdrb_hargakonstan'), fields.get('PDRB_HargaKonstan', 40000.0))

    denominator = 1 + (scenario_annual_inflation / 100.0)
    real_ump = ump_value / denominator if denominator != 0 else ump_value

    prev_real_ump = _safe_float(fields.get('Prev_Real_UMP'), fields.get('Real_UMP', real_ump))
    prev_pdrb = _safe_float(fields.get('Prev_PDRB_HargaKonstan'), pdrb_value)
    prev_tpt = _safe_float(fields.get('Prev_TPT'), tpt_value)
    baseline_year = int(_safe_float(fields.get('Tahun'), 0))

    feature_row = {
        'Provinsi': province,
        'Year_Index': _safe_float(fields.get('Year_Index'), baseline_year),
        'TPT': tpt_value,
        'TPAK': _safe_float(fields.get('TPAK'), 68.0),
        'PDRB_HargaKonstan': pdrb_value,
        'Inflasi_Rata_Tahunan': scenario_local_inflation,
        'Inflation_Deflator': denominator,
        'Gini_Rasio': _safe_float(fields.get('Gini_Rasio'), 0.30),
        'IPM': _safe_float(fields.get('IPM'), 72.4),
        'Garis_Kemiskinan': _safe_float(fields.get('Garis_Kemiskinan'), 609000.0),
        'Jumlah_Penduduk': _safe_float(fields.get('Jumlah_Penduduk'), 8000.0),
        'Pct_Populasi': _safe_float(fields.get('Pct_Populasi'), 2.8),
        'Pct_Akses_Air_Bersih': _safe_float(fields.get('Pct_Akses_Air_Bersih'), 87.7),
        'Protein_gram_per_hari': _safe_float(fields.get('Protein_gram_per_hari'), 62.3),
        'Inflasi_WB_Annual': scenario_annual_inflation,
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
        'Inflasi_x_TPT': scenario_local_inflation * tpt_value,
        'Log_PDRB': float(np.log1p(max(pdrb_value, 0.0))),
        'Log_UMP': float(np.log1p(max(real_ump, 0.0))),
        'Prev_Total_Pengeluaran_Riil': _safe_float(fields.get('Prev_Total_Pengeluaran_Riil'), baseline['baseline_pengeluaran']),
    }

    if RIDGE_MODEL_BUNDLE is not None and 'num_features' in RIDGE_MODEL_BUNDLE:
        for feat in RIDGE_MODEL_BUNDLE.get('num_features', []):
            if feat not in feature_row:
                feature_row[feat] = _safe_float(fields.get(feat), 0.0)

    return pd.DataFrame([feature_row]), baseline

def load_models(load_inflation=True):
    global LSTM_MODEL, LSTM_SCALER_X, LSTM_SCALER_Y, RIDGE_MODEL, RIDGE_MODEL_BUNDLE, INFLASI_PRED_MEM, DATA_HISTORIS, LSTM_FEATURES, RATA_PENGELUARAN, RIDGE_SIMULATION_DEFAULTS
    
    project_root = os.path.dirname(settings.BASE_DIR)
    models_dir = os.path.join(project_root, 'models')
    data_path = inflation_dataset_path(project_root)
    
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

    if not load_inflation:
        return INFLASI_PRED_MEM, DATA_HISTORIS, RATA_PENGELUARAN
            
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
        checkpoint = torch.load(lstm_path, map_location=torch.device('cpu'), weights_only=False)
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
    load_models(load_inflation=False)
    forecast_payload = _get_inflation_forecast_payload()
    public_horizon = ((forecast_payload or {}).get('horizons') or {}).get('1m', {})
    top_models = public_horizon.get('top_models') or []
    top_model = top_models[0] if top_models else {}
    inflasi_pred = _safe_float(public_horizon.get('headline_forecast'))
    headline_interval = public_horizon.get('headline_interval') or {}
    rata_pengeluaran = RATA_PENGELUARAN
    
    # Ambil data tambahan dari dataset clean_daya_beli.csv untuk visualisasi Overview
    project_root = os.path.dirname(settings.BASE_DIR)
    db_path = os.path.join(project_root, 'datasets', 'processed', 'clean_daya_beli.csv')
    
    chart_labels = []
    ump_data = []
    pdrb_data = []
    pie_data = []
    regional_snapshot_cards = []
    regional_snapshot_brief = None
    regional_metric_explorer = {}
    structural_insight_cards = []
    quality_of_life_cards = []
    quality_trend_explorer = {}
    dashboard_snapshot_year = None
    
    if os.path.exists(db_path):
        df_db = prepare_daya_beli_dataframe(pd.read_csv(db_path))
        # Ambil rata-rata UMP per tahun
        yearly_data = df_db.groupby('Tahun').agg({'UMP': 'mean', 'PDRB_HargaKonstan': 'mean'}).reset_index().sort_values('Tahun')
        chart_labels = yearly_data['Tahun'].tolist()
        ump_data = yearly_data['UMP'].tolist()
        pdrb_data = yearly_data['PDRB_HargaKonstan'].tolist()
        
        # Ambil data untuk pie chart proporsi pengeluaran secara keseluruhan
        makanan_mean = df_db['Pengeluaran_Makanan'].mean() if 'Pengeluaran_Makanan' in df_db.columns else 600000
        bukan_makanan_mean = df_db['Pengeluaran_Bukan_Makanan'].mean() if 'Pengeluaran_Bukan_Makanan' in df_db.columns else 850000
        pie_data = [float(makanan_mean), float(bukan_makanan_mean)]

        if not df_db.empty and 'Tahun' in df_db.columns:
            dashboard_snapshot_year = int(df_db['Tahun'].max())
            snapshot = df_db[df_db['Tahun'] == dashboard_snapshot_year].copy()
            snapshot = snapshot[snapshot['Provinsi'].notna()]
            snapshot = snapshot[snapshot['Provinsi'] != 'Indonesia']

            if not snapshot.empty:
                def _snapshot_card(metric, *, ascending=False, title='', value_prefix='', formatter='number', tone='primary'):
                    if metric not in snapshot.columns:
                        return None
                    data = snapshot[['Provinsi', metric]].dropna()
                    if data.empty:
                        return None
                    ranked = data.sort_values(metric, ascending=ascending)
                    row = ranked.iloc[0]
                    metric_value = _safe_float(row[metric])
                    if formatter == 'currency':
                        formatted_value = _format_compact_rupiah(metric_value)
                    elif formatter == 'percent':
                        formatted_value = f"{metric_value:.2f}%"
                    else:
                        formatted_value = _format_compact_number(metric_value)
                    prefix = f"{value_prefix} " if value_prefix else ""
                    return {
                        'tone': tone,
                        'title': title,
                        'province': row['Provinsi'],
                        'value': formatted_value,
                        'note': f"{prefix}{formatted_value} pada snapshot wilayah {dashboard_snapshot_year}.",
                    }

                def _snapshot_point(metric, *, ascending=False, formatter='number'):
                    if metric not in snapshot.columns:
                        return None
                    data = snapshot[['Provinsi', metric]].dropna()
                    if data.empty:
                        return None
                    ranked = data.sort_values(metric, ascending=ascending)
                    row = ranked.iloc[0]
                    metric_value = _safe_float(row[metric])
                    if formatter == 'currency':
                        formatted_value = _format_compact_rupiah(metric_value)
                    elif formatter == 'percent':
                        formatted_value = f"{metric_value:.2f}%"
                    else:
                        formatted_value = _format_compact_number(metric_value)
                    return {
                        'province': row['Provinsi'],
                        'value': formatted_value,
                    }

                def _metric_explorer(metric, *, label, ascending=False, formatter='number'):
                    if metric not in snapshot.columns:
                        return None
                    data = snapshot[['Provinsi', metric]].dropna()
                    if data.empty:
                        return None
                    ranked = data.sort_values(metric, ascending=ascending).head(5)
                    values = [_safe_float(value) for value in ranked[metric].tolist()]
                    if formatter == 'currency':
                        formatted_values = [_format_compact_rupiah(value) for value in values]
                    elif formatter == 'percent':
                        formatted_values = [f"{value:.2f}%" for value in values]
                    else:
                        formatted_values = [_format_compact_number(value) for value in values]
                    return {
                        'label': label,
                        'metric': metric,
                        'formatter': formatter,
                        'direction': 'ascending' if ascending else 'descending',
                        'provinces': ranked['Provinsi'].tolist(),
                        'values': values,
                        'formatted_values': formatted_values,
                    }

                def _latest_row(metric, *, ascending=False):
                    if metric not in snapshot.columns:
                        return None
                    data = snapshot[['Provinsi', metric]].dropna()
                    if data.empty:
                        return None
                    ranked = data.sort_values(metric, ascending=ascending)
                    return ranked.iloc[0]

                def _snapshot_average(metric):
                    if metric not in snapshot.columns:
                        return None
                    data = snapshot[metric].dropna()
                    if data.empty:
                        return None
                    return float(data.mean())

                def _trend_payload(metric, *, label, formatter='number'):
                    if metric not in df_db.columns:
                        return None
                    series = (
                        df_db.groupby('Tahun')[metric]
                        .mean()
                        .dropna()
                        .reset_index()
                        .sort_values('Tahun')
                    )
                    if series.empty:
                        return None
                    values = [_safe_float(value) for value in series[metric].tolist()]
                    if formatter == 'percent':
                        formatted_values = [_format_percent(value) for value in values]
                    elif formatter == 'currency':
                        formatted_values = [_format_compact_rupiah(value) for value in values]
                    else:
                        formatted_values = [_format_decimal(value) for value in values]
                    return {
                        'label': label,
                        'metric': metric,
                        'formatter': formatter,
                        'years': [int(year) for year in series['Tahun'].tolist()],
                        'values': values,
                        'formatted_values': formatted_values,
                    }

                candidates = [
                    _snapshot_card(
                        TARGET_COLUMN,
                        ascending=False,
                        title='Proksi daya beli tertinggi',
                        value_prefix='Pengeluaran riil',
                        formatter='currency',
                        tone='primary',
                    ),
                    _snapshot_card(
                        'Pct_Penduduk_Miskin',
                        ascending=True,
                        title='Kemiskinan terendah',
                        value_prefix='Tingkat kemiskinan',
                        formatter='percent',
                        tone='amber',
                    ),
                    _snapshot_card(
                        'IPM',
                        ascending=False,
                        title='IPM tertinggi',
                        value_prefix='Indeks pembangunan manusia',
                        formatter='number',
                        tone='teal',
                    ),
                    _snapshot_card(
                        'Rerata_Lama_Sekolah',
                        ascending=False,
                        title='Lama sekolah tertinggi',
                        value_prefix='Rata-rata lama sekolah',
                        formatter='number',
                        tone='violet',
                    ),
                    _snapshot_card(
                        'Pct_Sanitasi_Layak',
                        ascending=False,
                        title='Sanitasi layak tertinggi',
                        value_prefix='Akses sanitasi',
                        formatter='percent',
                        tone='teal',
                    ),
                ]
                regional_snapshot_cards = [card for card in candidates if card]

                top_spending = _snapshot_point(TARGET_COLUMN, ascending=False, formatter='currency')
                low_poverty = _snapshot_point('Pct_Penduduk_Miskin', ascending=True, formatter='percent')
                high_ipm = _snapshot_point('IPM', ascending=False, formatter='number')
                if top_spending and low_poverty and high_ipm:
                    regional_snapshot_brief = {
                        'headline': (
                            f"Pada snapshot {dashboard_snapshot_year}, {top_spending['province']} memimpin proksi daya beli, "
                            f"{low_poverty['province']} mencatat kemiskinan terendah, dan {high_ipm['province']} berada di posisi teratas untuk IPM."
                        ),
                        'detail': (
                            f"Pembacaan ini membantu memisahkan kekuatan konsumsi ({top_spending['value']}), tekanan sosial "
                            f"({low_poverty['value']} penduduk miskin), dan kualitas pembangunan manusia ({high_ipm['value']})."
                        ),
                    }

                explorer_candidates = {
                    'spending': _metric_explorer(
                        TARGET_COLUMN,
                        label='Top proksi daya beli',
                        ascending=False,
                        formatter='currency',
                    ),
                    'ipm': _metric_explorer(
                        'IPM',
                        label='Top IPM',
                        ascending=False,
                        formatter='number',
                    ),
                    'poverty': _metric_explorer(
                        'Pct_Penduduk_Miskin',
                        label='Kemiskinan terendah',
                        ascending=True,
                        formatter='percent',
                    ),
                    'sanitation': _metric_explorer(
                        'Pct_Sanitasi_Layak',
                        label='Sanitasi layak tertinggi',
                        ascending=False,
                        formatter='percent',
                    ),
                    'formal': _metric_explorer(
                        'Pct_Pekerja_Formal',
                        label='Pekerja formal tertinggi',
                        ascending=False,
                        formatter='percent',
                    ),
                }
                regional_metric_explorer = {key: value for key, value in explorer_candidates.items() if value}

                highest_gini = _latest_row('Gini_Rasio', ascending=False)
                lowest_poverty = _latest_row('Pct_Penduduk_Miskin', ascending=True)
                highest_formal = _latest_row('Pct_Pekerja_Formal', ascending=False)
                highest_tpak = _latest_row('TPAK', ascending=False)
                highest_pdrb = _latest_row('PDRB_HargaKonstan', ascending=False)
                highest_investment = _latest_row('Realisasi_Investasi_PMDN', ascending=False)
                highest_air = _latest_row('Pct_Akses_Air_Bersih', ascending=False)
                highest_calorie = _latest_row('Kalori_kkal_per_hari', ascending=False)
                highest_school = _latest_row('Rerata_Lama_Sekolah', ascending=False)
                highest_sanitation = _latest_row('Pct_Sanitasi_Layak', ascending=False)
                highest_ipm = _latest_row('IPM', ascending=False)

                gini_avg = _snapshot_average('Gini_Rasio')
                poverty_avg = _snapshot_average('Pct_Penduduk_Miskin')
                formal_avg = _snapshot_average('Pct_Pekerja_Formal')
                pdrb_avg = _snapshot_average('PDRB_HargaKonstan')

                structural_candidates = [
                    {
                        'tone': 'amber',
                        'title': 'Ketimpangan dan tekanan sosial',
                        'value': _format_decimal(gini_avg, 3) if gini_avg is not None else '-',
                        'value_label': 'Rata-rata gini ratio',
                        'reference': (
                            f"Gini tertinggi di {highest_gini['Provinsi']} dan kemiskinan terendah di {lowest_poverty['Provinsi']}."
                            if highest_gini is not None and lowest_poverty is not None else ''
                        ),
                        'note': (
                            f"Rata-rata kemiskinan wilayah aktif berada di {_format_percent(poverty_avg)} sehingga pembacaan distribusi pengeluaran perlu dibaca bersama tekanan sosial."
                            if poverty_avg is not None else ''
                        ),
                    },
                    {
                        'tone': 'teal',
                        'title': 'Kualitas pasar kerja',
                        'value': _format_percent(formal_avg) if formal_avg is not None else '-',
                        'value_label': 'Rata-rata pekerja formal',
                        'reference': (
                            f"Pekerja formal tertinggi berada di {highest_formal['Provinsi']} dan TPAK tertinggi di {highest_tpak['Provinsi']}."
                            if highest_formal is not None and highest_tpak is not None else ''
                        ),
                        'note': 'Kualitas pasar kerja dibaca dari kombinasi formalitas tenaga kerja, partisipasi angkatan kerja, dan tekanan pengangguran wilayah.',
                    },
                    {
                        'tone': 'primary',
                        'title': 'Skala ekonomi dan investasi',
                        'value': _format_compact_rupiah(pdrb_avg) if pdrb_avg is not None else '-',
                        'value_label': 'Rata-rata PDRB konstan',
                        'reference': (
                            f"PDRB konstan tertinggi berada di {highest_pdrb['Provinsi']} dan investasi PMDN tertinggi di {highest_investment['Provinsi']}."
                            if highest_pdrb is not None and highest_investment is not None else ''
                        ),
                        'note': 'Pasangan indikator ini membantu membedakan wilayah yang besar secara ekonomi dari wilayah yang sedang menarik ekspansi investasi domestik.',
                    },
                ]
                structural_insight_cards = [card for card in structural_candidates if card.get('value') != '-']

                quality_candidates = [
                    {
                        'tone': 'teal',
                        'title': 'IPM tertinggi',
                        'province': highest_ipm['Provinsi'] if highest_ipm is not None else '-',
                        'value': _format_decimal(highest_ipm['IPM'], 0) if highest_ipm is not None else '-',
                        'note': 'Posisi teratas pada snapshot kualitas hidup',
                        'detail': 'Posisi ini menjadi acuan kualitas pembangunan manusia pada snapshot wilayah aktif.',
                    },
                    {
                        'tone': 'violet',
                        'title': 'Akses dasar terkuat',
                        'province': highest_air['Provinsi'] if highest_air is not None else '-',
                        'value': _format_percent(highest_air['Pct_Akses_Air_Bersih']) if highest_air is not None else '-',
                        'note': 'Akses air bersih tertinggi pada wilayah aktif',
                        'detail': (
                            f"Sanitasi tertinggi juga berada di {highest_sanitation['Provinsi']}."
                            if highest_sanitation is not None else 'Akses dasar dibaca dari air bersih dan sanitasi layak.'
                        ),
                    },
                    {
                        'tone': 'amber',
                        'title': 'Pola konsumsi paling kuat',
                        'province': highest_calorie['Provinsi'] if highest_calorie is not None else '-',
                        'value': _format_decimal(highest_calorie['Kalori_kkal_per_hari'], 0) if highest_calorie is not None else '-',
                        'note': 'Konsumsi energi harian tertinggi pada snapshot aktif',
                        'detail': (
                            f"Rata-rata lama sekolah tertinggi berada di {highest_school['Provinsi']} untuk konteks kualitas modal manusia."
                            if highest_school is not None else 'Konsumsi pangan dilihat sebagai konteks kesejahteraan dasar rumah tangga.'
                        ),
                    },
                ]
                quality_of_life_cards = [card for card in quality_candidates if card.get('value') != '-']

                trend_candidates = {
                    'ipm': _trend_payload('IPM', label='IPM nasional rata-rata', formatter='number'),
                    'poverty': _trend_payload('Pct_Penduduk_Miskin', label='Kemiskinan wilayah rata-rata', formatter='percent'),
                    'formal': _trend_payload('Pct_Pekerja_Formal', label='Pekerja formal rata-rata', formatter='percent'),
                    'school': _trend_payload('Rerata_Lama_Sekolah', label='Rata-rata lama sekolah', formatter='number'),
                }
                quality_trend_explorer = {key: value for key, value in trend_candidates.items() if value}

    context = {
        'inflasi_pred': float(inflasi_pred) if inflasi_pred else 0.0,
        'inflasi_model_name': top_model.get('name', 'Model publik aktif'),
        'inflasi_interval_lower': top_model.get('ci_lower', headline_interval.get('lower')),
        'inflasi_interval_upper': top_model.get('ci_upper', headline_interval.get('upper')),
        'public_horizon_label': public_horizon.get('label', '1 Bulan'),
        'forecast_generated_at': (forecast_payload or {}).get('generated_at'),
        'rata_pengeluaran': "{:,.0f}".format(rata_pengeluaran).replace(',', '.'),
        'ridge_test_r2': round(_safe_float((RIDGE_MODEL_BUNDLE or {}).get('test_r2'), 0.0), 3),
        'ridge_test_mae': round(_safe_float((RIDGE_MODEL_BUNDLE or {}).get('test_mae'), 0.0), 0),
        'ridge_target_label': (RIDGE_MODEL_BUNDLE or {}).get('target_label', TARGET_LABEL),
        'province_count': _get_actual_province_count(),
        'chart_labels': json.dumps(chart_labels),
        'ump_data': json.dumps(ump_data),
        'pdrb_data': json.dumps(pdrb_data),
        'pie_data': json.dumps(pie_data),
        'regional_snapshot_cards': regional_snapshot_cards,
        'regional_snapshot_brief': regional_snapshot_brief,
        'regional_metric_explorer_json': json.dumps(regional_metric_explorer),
        'structural_insight_cards': structural_insight_cards,
        'quality_of_life_cards': quality_of_life_cards,
        'quality_trend_explorer_json': json.dumps(quality_trend_explorer),
        'dashboard_snapshot_year': dashboard_snapshot_year,
    }
    return render(request, 'predictions/landing.html', context)


def forecasting_page(request):
    forecast_payload = _get_inflation_forecast_payload()
    context = {
        'forecast_payload_json': json.dumps(forecast_payload or {}),
    }
    return render(request, 'predictions/forecasting.html', context)


def get_regression_dummy_data(inflasi_val):
    default_province = (RIDGE_SIMULATION_DEFAULTS or {}).get('Provinsi', 'Jawa Timur')
    dummy_input, _ = _build_simulation_input(default_province, {'inflasi': inflasi_val})
    return dummy_input

def daya_beli_page(request):
    load_models(load_inflation=False)

    baselines = _get_province_simulation_baselines()
    province_names = sorted((name for name in baselines.keys() if name != 'Indonesia'))
    if 'Indonesia' in baselines:
        province_names = ['Indonesia'] + province_names
    default_province = (RIDGE_SIMULATION_DEFAULTS or {}).get('Provinsi')
    if default_province not in baselines:
        default_province = 'Indonesia' if 'Indonesia' in baselines else (province_names[0] if province_names else '')

    province_defaults = {}
    for province in province_names:
        baseline = baselines[province]
        fields = baseline['fields']
        province_defaults[province] = {
            'year': baseline['baseline_year'],
            'baseline_pengeluaran': round(_safe_float(baseline['baseline_pengeluaran']), 2),
            'baseline_pengeluaran_nominal': round(_safe_float(baseline['baseline_pengeluaran_nominal']), 2),
            'inflasi': round(_safe_float(fields.get('Inflasi_WB_Annual'), fields.get('Inflasi_Rata_Tahunan')), 2),
            'ump': round(_safe_float(fields.get('UMP')), 2),
            'tpt': round(_safe_float(fields.get('TPT')), 3),
            'pdrb_hargakonstan': round(_safe_float(fields.get('PDRB_HargaKonstan')), 2),
        }
    context = {
        'provinces': province_names,
        'default_province': default_province,
        'province_defaults_json': json.dumps(province_defaults),
        'ridge_target_label': (RIDGE_MODEL_BUNDLE or {}).get('target_label', TARGET_LABEL),
        'ridge_test_r2': round(_safe_float((RIDGE_MODEL_BUNDLE or {}).get('test_r2'), 0.0), 3),
        'ridge_test_mae': round(_safe_float((RIDGE_MODEL_BUNDLE or {}).get('test_mae'), 0.0), 0),
    }
    return render(request, 'predictions/daya_beli.html', context)
def simulate_daya_beli(request):
    provinsi = request.GET.get('provinsi', '').strip()
    if not provinsi:
        return JsonResponse({'error': 'Wilayah wajib dipilih'}, status=400)

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

    load_models(load_inflation=False)
    if RIDGE_MODEL is None:
        return JsonResponse({'error': 'Model belum siap'}, status=500)
    if (RIDGE_MODEL_BUNDLE or {}).get('legacy_artifact'):
        return JsonResponse(
            {'error': 'Artifact model lama tidak kompatibel dengan inference per wilayah terbaru.'},
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

        annual_inflation_used = _safe_float(
            overrides.get('inflasi'),
            dummy_input.at[0, 'Inflasi_WB_Annual'],
        )
        inflation_deflator = _safe_float(dummy_input.at[0, 'Inflation_Deflator'], 1.0)
        inputs_used = {
            'provinsi': provinsi,
            'inflasi': round(annual_inflation_used, 2),
            'ump': round(_safe_float(dummy_input.at[0, 'Real_UMP']) * inflation_deflator, 2),
            'tpt': round(_safe_float(dummy_input.at[0, 'TPT']), 3),
            'pdrb_hargakonstan': round(_safe_float(dummy_input.at[0, 'PDRB_HargaKonstan']), 2),
        }
        return JsonResponse({
            'predicted_pengeluaran': round(predicted_pengeluaran, 2),
            'province': provinsi,
            'baseline_year': int(baseline['baseline_year']),
            'baseline_pengeluaran': round(baseline_pengeluaran, 2),
            'baseline_pengeluaran_nominal': round(_safe_float(baseline['baseline_pengeluaran_nominal']), 2),
            'inputs_used': inputs_used,
            'status_label': status_label,
            'target_label': (RIDGE_MODEL_BUNDLE or {}).get('target_label', TARGET_LABEL),
            'target_type': (RIDGE_MODEL_BUNDLE or {}).get('target_type', 'real'),
        })
    except KeyError:
        return JsonResponse({'error': 'Provinsi tidak ditemukan'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def home_page(request):
    load_models(load_inflation=False)
    forecast_payload = _get_inflation_forecast_payload()
    public_horizon = ((forecast_payload or {}).get('horizons') or {}).get('1m', {})
    top_models = public_horizon.get('top_models') or []
    top_model = top_models[0] if top_models else {}
    inflasi_pred = _safe_float(public_horizon.get('headline_forecast'))
    headline_interval = public_horizon.get('headline_interval') or {}
    context = {
        'inflasi_pred': float(inflasi_pred) if inflasi_pred else 0.0,
        'inflasi_model_name': top_model.get('name', 'Model publik aktif'),
        'inflasi_interval_lower': top_model.get('ci_lower', headline_interval.get('lower')),
        'inflasi_interval_upper': top_model.get('ci_upper', headline_interval.get('upper')),
        'public_horizon_label': public_horizon.get('label', '1 Bulan'),
        'forecast_generated_at': (forecast_payload or {}).get('generated_at'),
        'ridge_test_r2': round(_safe_float((RIDGE_MODEL_BUNDLE or {}).get('test_r2'), 0.0), 3),
        'ridge_test_mae': round(_safe_float((RIDGE_MODEL_BUNDLE or {}).get('test_mae'), 0.0), 0),
        'ridge_target_label': (RIDGE_MODEL_BUNDLE or {}).get('target_label', TARGET_LABEL),
        'province_count': _get_actual_province_count(),
    }
    return render(request, 'predictions/home.html', context)


def guide_page(request):
    load_models(load_inflation=False)
    return render(
        request,
        'predictions/guide.html',
        {
            'sarimax_feature_audit': _build_sarimax_feature_audit_context(),
            'model_family_notes': _get_model_family_notes(),
            'ridge_model_summary': _build_ridge_model_guide_context(),
        },
    )


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
        path = inflation_dataset_path(os.path.dirname(settings.BASE_DIR))
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
        df = prepare_daya_beli_dataframe(pd.read_csv(path))
        if 'Provinsi' in df.columns:
            provinces = sorted([name for name in df['Provinsi'].unique().tolist() if name != 'Indonesia'])
        else:
            provinces = []
        return JsonResponse({'provinces': provinces})
    except Exception:
        return JsonResponse({'provinces': []})


def api_province_data(request):
    """Return data for selected provinces and metric."""
    provinces = request.GET.getlist('provinsi')
    metric = request.GET.get('metric', TARGET_COLUMN)
    project_root = os.path.dirname(settings.BASE_DIR)
    path = os.path.join(project_root, 'datasets', 'processed', 'clean_daya_beli.csv')
    
    if not os.path.exists(path):
        return JsonResponse({'error': 'Data not found'}, status=404)
    
    try:
        df = prepare_daya_beli_dataframe(pd.read_csv(path))
        if 'Provinsi' not in df.columns or 'Tahun' not in df.columns:
            return JsonResponse({'error': 'Required columns missing'}, status=500)

        if metric not in df.columns:
            metric = TARGET_COLUMN

        df = df[df['Provinsi'] != 'Indonesia']

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
            TARGET_COLUMN: {'label': 'Daya Beli Riil', 'unit': 'Rp', 'format': 'currency'},
            NOMINAL_TARGET_COLUMN: {'label': 'Pengeluaran Nominal', 'unit': 'Rp', 'format': 'currency'},
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
    load_models(load_inflation=False)

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

    public_horizon = _get_public_horizon_forecast('1m')
    inflasi_val = _safe_float(public_horizon.get('headline_forecast'), 2.5)

    return JsonResponse({
        "commodities": commodities,
        "inflasi_prediksi": inflasi_val,
        "base_pengeluaran": RATA_PENGELUARAN,
    })


def api_all_metrics_latest(request):
    """Return yearly metrics for all provinces."""
    project_root = os.path.dirname(settings.BASE_DIR)
    path = os.path.join(project_root, 'datasets', 'processed', 'clean_daya_beli.csv')
    if not os.path.exists(path):
        return _json_no_store({'error': 'Data not found'}, status=404)
    try:
        df = prepare_daya_beli_dataframe(pd.read_csv(path))
        available_years = sorted(int(year) for year in df['Tahun'].dropna().unique().tolist())
        latest_year = max(available_years)
        selected_year = latest_year
        requested_year = request.GET.get('year')
        if requested_year not in (None, ''):
            try:
                parsed_year = int(requested_year)
            except ValueError:
                parsed_year = latest_year
            if parsed_year in available_years:
                selected_year = parsed_year

        provincial_df = df[df['Provinsi'] != 'Indonesia']
        selected_rows = provincial_df[provincial_df['Tahun'] == selected_year]

        metrics = [
            TARGET_COLUMN,
            NOMINAL_TARGET_COLUMN,
            'UMP',
            'PDRB_HargaKonstan',
            'PDRB_HargaBerlaku',
            'TPT',
            'TPAK',
            'IPM',
            'Gini_Rasio',
            'Pct_Penduduk_Miskin',
            'Garis_Kemiskinan',
            'NTP',
            'Pct_Sanitasi_Layak',
            'Rerata_Lama_Sekolah',
            'Inflasi_Rata_Tahunan',
        ]
        available = [m for m in metrics if m in selected_rows.columns]
        
        result = {}
        for _, row in selected_rows.iterrows():
            prov = row['Provinsi']
            result[prov] = {m: float(row[m]) if pd.notna(row[m]) else 0 for m in available}
            result[prov]['Tahun'] = int(row['Tahun'])

        all_provinces = sorted(provincial_df['Provinsi'].dropna().unique().tolist())
        selected_provinces = sorted(selected_rows['Provinsi'].dropna().unique().tolist())
        missing_provinces = [name for name in all_provinces if name not in selected_provinces]

        return _json_no_store(
            {
                'latest_year': int(latest_year),
                'selected_year': int(selected_year),
                'available_years': available_years,
                'provinces': result,
                'metrics': available,
                'coverage_count': len(selected_provinces),
                'coverage_total': len(all_provinces),
                'missing_provinces': missing_provinces,
            }
        )
    except Exception as e:
        return _json_no_store({'error': str(e)}, status=500)


def api_scenario_analysis(request):
    """Return deterministic scenario analysis derived from the active ridge model."""
    load_models(load_inflation=False)
    if RIDGE_MODEL is None:
        return _json_no_store({'error': 'Model proksi daya beli belum siap.'}, status=500)
    if (RIDGE_MODEL_BUNDLE or {}).get('legacy_artifact'):
        return _json_no_store({'error': 'Artifact model lama tidak kompatibel dengan analisis skenario aktif.'}, status=500)

    scenario_id = request.GET.get('scenario_id', 'inflation_shock')
    scenario_spec = SCENARIO_LIBRARY.get(scenario_id)
    if scenario_spec is None:
        return _json_no_store(
            {
                'error': 'Scenario tidak ditemukan',
                'available_scenarios': list(SCENARIO_LIBRARY.keys()),
            },
            status=404,
        )

    baselines = _get_province_simulation_baselines()
    province_names = sorted([name for name in baselines.keys() if name != 'Indonesia'])

    try:
        baseline_value, baseline_meta, _ = _predict_simulation_value('Indonesia')
        national_overrides = _build_scenario_overrides('Indonesia', scenario_spec)
        scenario_value, _, _ = _predict_simulation_value('Indonesia', national_overrides)
        national_change_pct = _pct_change(scenario_value, baseline_value, 0.0)

        province_impacts = []
        for province in province_names:
            province_baseline_value, _, _ = _predict_simulation_value(province)
            province_overrides = _build_scenario_overrides(province, scenario_spec)
            province_scenario_value, _, _ = _predict_simulation_value(province, province_overrides)
            province_change_pct = _pct_change(province_scenario_value, province_baseline_value, 0.0)
            province_impacts.append(
                {
                    'province': province,
                    'baseline_value': round(province_baseline_value, 2),
                    'scenario_value': round(province_scenario_value, 2),
                    'change_pct': round(province_change_pct, 2),
                }
            )

        province_impacts.sort(key=lambda item: abs(item['change_pct']), reverse=True)
        featured_impacts = province_impacts[:8]
        direction_label = _scenario_direction_label(national_change_pct)
        featured_year = int(_safe_float(baseline_meta.get('baseline_year'), 0))

        payload = {
            'scenario_id': scenario_id,
            'title': scenario_spec['title'],
            'description': scenario_spec['description'],
            'baseline_year': featured_year,
            'baseline_value': round(baseline_value, 2),
            'scenario_value': round(scenario_value, 2),
            'change_pct': round(national_change_pct, 2),
            'status_label': direction_label,
            'coverage_label': f'Agregat nasional dengan pembanding {len(province_impacts)} provinsi.',
            'assumptions': scenario_spec.get('assumptions', []),
            'series': {
                'labels': [item['province'] for item in featured_impacts],
                'baseline': [item['baseline_value'] for item in featured_impacts],
                'scenario': [item['scenario_value'] for item in featured_impacts],
                'change_pct': [item['change_pct'] for item in featured_impacts],
                'focus': 'Provinsi dengan perubahan relatif paling menonjol',
            },
            'province_impacts': province_impacts,
            'interpretation': scenario_spec['interpretation_template'].format(direction=direction_label),
            'limitations': [
                'Analisis memakai model ridge aktif dengan baseline wilayah terbaru yang tersedia pada data 2021-2025.',
                'Output dibaca sebagai estimasi pengeluaran riil per kapita per bulan, lalu diinterpretasikan sebagai proksi daya beli.',
                'Skenario ini membantu membaca arah dan sensitivitas relatif, bukan menetapkan angka kebijakan sampai digit terakhir.',
            ],
        }
        return _json_no_store(payload)
    except KeyError as error:
        return _json_no_store({'error': str(error)}, status=404)
    except Exception as error:
        return _json_no_store({'error': str(error)}, status=500)


def api_usd_idr_latest(request):
    """Return the latest USD/IDR rate with month-to-date comparison data."""
    import urllib.request
    import json as json_lib
    from datetime import date, datetime, timedelta
    
    project_root = os.path.dirname(settings.BASE_DIR)
    path = inflation_dataset_path(project_root)
    
    # Prefer the latest daily reference first, then enrich it with current-month history.
    daily_rate = None
    daily_date = None
    previous_rate = None
    previous_date = None
    month_start_rate = None
    month_start_date = None
    daily_history = []
    source_label = None

    try:
        latest_url = "https://open.er-api.com/v6/latest/USD"
        req = urllib.request.Request(latest_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json_lib.loads(resp.read().decode())
            idr_rate = (data.get('rates') or {}).get('IDR')
            if idr_rate is not None:
                daily_rate = round(float(idr_rate), 2)
                source_label = 'open.er-api.com (latest daily reference)'
                updated_unix = data.get('time_last_update_unix')
                if updated_unix:
                    daily_date = datetime.utcfromtimestamp(int(updated_unix)).date().isoformat()
    except Exception:
        pass

    try:
        end_date = date.today()
        start_date = end_date.replace(day=1)
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
                daily_history = [round(float(rate), 2) for _, rate in observations]
                month_start_date, month_start_rate = observations[0]
                month_start_rate = round(float(month_start_rate), 2)
                if daily_rate is None:
                    daily_date, daily_rate = observations[-1]
                    daily_rate = round(float(daily_rate), 2)
                    source_label = 'Frankfurter (central bank reference rates)'
                previous_date = month_start_date
                previous_rate = month_start_rate
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
    if daily_rate is not None and month_start_rate:
        change_pct = ((daily_rate - month_start_rate) / month_start_rate) * 100
    elif monthly_history and len(monthly_history) >= 2:
        prev = monthly_history[-2]
        if prev > 0:
            change_pct = ((monthly_history[-1] - prev) / prev) * 100

    history = daily_history if daily_history else monthly_history
    
    return _json_no_store({
        'latest': latest,
        'daily_rate': daily_rate,
        'daily_date': daily_date,
        'previous_rate': previous_rate,
        'previous_date': previous_date,
        'month_start_rate': month_start_rate,
        'month_start_date': month_start_date,
        'monthly_rate': monthly_rate,
        'monthly_date': monthly_date,
        'change_pct': round(change_pct, 2),
        'history': history,
        'source': source_label if daily_rate else 'BPS (monthly avg)',
        'data_type': 'daily' if daily_rate else 'monthly_avg',
        'generated_at': datetime.utcnow().isoformat() + 'Z',
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


def _get_inflation_forecast_payload(force_refresh=False):
    global INFLATION_FORECAST_PAYLOAD
    if force_refresh or INFLATION_FORECAST_PAYLOAD is None:
        project_root = os.path.dirname(settings.BASE_DIR)
        INFLATION_FORECAST_PAYLOAD = load_saved_forecast_payload(project_root)
    return INFLATION_FORECAST_PAYLOAD


def _get_public_horizon_forecast(horizon_key='1m'):
    payload = _get_inflation_forecast_payload()
    horizons = (payload or {}).get('horizons') or {}
    return horizons.get(horizon_key) or {}


def _build_legacy_ensemble_payload():
    payload = _get_inflation_forecast_payload() or {}
    horizon = (payload.get('horizons') or {}).get('1m') or {}
    if not horizon:
        return None

    top_models = horizon.get('top_models') or []
    comparison_rows = (payload.get('comparison_summary') or {}).get('1m', [])
    comparison = {}
    for row in comparison_rows:
        metrics = row.get('metrics') or {}
        if row.get('status') == 'ok':
            comparison[row['id']] = {
                'mae': round(_safe_float(metrics.get('mae')), 4),
                'rmse': round(_safe_float(metrics.get('rmse')), 4),
                'smape': round(_safe_float(metrics.get('smape')), 2),
                'n_test': int(metrics.get('n_test', 0)),
            }

    forecast = {}
    for model in top_models:
        forecast[model['id']] = [model.get('point_forecast')]
    return {
        'available': True,
        'forecast': forecast,
        'weights': {},
        'last_date': (payload.get('history') or {}).get('last_date'),
        'last_value': (payload.get('history') or {}).get('last_actual_mom', 0),
        'comparison': comparison,
        'best_model': horizon.get('headline_model'),
    }


def api_ensemble_forecast(request):
    """Return ensemble forecast (LSTM + ARIMA + Prophet) + comparison metrics."""
    legacy_payload = _build_legacy_ensemble_payload()
    if legacy_payload is not None:
        return _json_no_store(legacy_payload)

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


def api_inflation_forecast(request):
    payload = _get_inflation_forecast_payload(force_refresh=True)
    if payload is None:
        return _json_no_store(
            {'error': 'Artefak forecast multi-horizon belum tersedia. Jalankan train_inflation_multihorizon.py terlebih dahulu.'},
            status=503,
        )
    return _json_no_store(payload)


# ============================================================
# INFLASI SUMMARY API (M-to-M, Y-o-Y, Y-to-D)
# ============================================================

def api_inflasi_summary(request):
    """Return ringkasan inflasi: M-to-M, Y-o-Y, Y-to-D, dan histori 24 bulan."""
    project_root = os.path.dirname(settings.BASE_DIR)
    data_path = inflation_dataset_path(project_root)
    
    if not os.path.exists(data_path):
        return _json_no_store({'error': 'Data file not found'}, status=404)
    
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
        
        payload = {
            'as_of': last_date,
            'date': last_date,
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
        
        return _json_no_store(payload)
    
    except Exception as e:
        return _json_no_store({'error': str(e)}, status=500)


# ============================================================
# MAP PAGE
# ============================================================

def map_page(request):
    """Indonesia choropleth map page."""
    return render(request, 'predictions/map.html')
