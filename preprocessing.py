# -*- coding: utf-8 -*-
"""
============================================================================
  PREPROCESSING PIPELINE (v3)
  Proyek: Prediksi Inflasi dan Dampaknya terhadap Daya Beli
  Kelompok E – Machine Learning SD-A1, Universitas Airlangga
============================================================================

Dataset yang digunakan (23 dataset):
  --- LOKAL (BPS, Bank Indonesia, dll) ---
   1.  Indeks Harga Konsumen (Umum)                    – BPS, 2005–2023
   2.  Inflasi Bulanan (M-to-M)                        – BPS, 2005–2026-02
   3.  Tingkat Inflasi Tahun Kalender (Y-to-D)         – BPS, referensi
   4.  BI Rate / Data Inflasi BI                       – Bank Indonesia
   5.  Upah Minimum Provinsi (UMP)                     – BPS Jateng, 2021–2025
   6.  Rata-rata Pengeluaran per Kapita                 – BPS, 2017–2025
   7.  Data Historis USD/IDR                           – Investing.com, bulanan
   8.  Tingkat Pengangguran Terbuka (Semester+Provinsi) – Open Data Jabar
   9.  TPT & TPAK Menurut Provinsi                     – BPS, 2017–2025
  10.  PDRB Per Kapita (Ribu Rupiah)                   – BPS, 2010–2025
  11.  Persentase Penduduk Miskin per Provinsi          – BPS, 2010–2024
  12.  Inflasi Umum, Inti, Harga Diatur, Bergejolak    – BPS, 2009–2026-05
  13.  Harga Bulanan Minyak Mentah (USD/Barel)          – IndexMundi, 2001–2026-03
  14.  USD/IDR Harian (Jan–Mei 2026)                   – Yahoo Finance

  --- INTERNASIONAL (Baru v3) ---
  15.  Crude Oil Brent (USD/Barel)                      – Yahoo Finance, 2007–2026
  16.  Indeks Dollar AS (DXY)                           – Yahoo Finance, 2003–2026
  17.  The Fed Funds Rate (%)                           – FRED, 2003–2026
  18.  Gold Price (USD/oz)                              – Yahoo Finance, 2003–2026
  19.  CPO Price (USD/mt)                               – Yahoo Finance, 2010–2026
  20.  Geopolitical Risk Index (GPR)                    – Caldara & Iacoviello
  21.  FAO Food Price Index (template, manual)         – FAO
  22.  Rice Price Thailand 5% (template, manual)        – World Bank Pink Sheet

  --- TODO (perlu download manual dari BPS) ---
  23.  Indeks Gini per Provinsi                         – BPS
  24.  IPM per Provinsi                                 – BPS

Output:
  1. datasets/processed/clean_inflasi_ts.csv  → Model 1 (LSTM Forecasting)
  2. datasets/processed/clean_daya_beli.csv   → Model 2 (Regresi Daya Beli)
============================================================================
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

BASE = "datasets"
OUT_DIR = os.path.join(BASE, "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
BULAN_MAP = {
    "Januari": 1, "Februari": 2, "Maret": 3, "April": 4,
    "Mei": 5, "Juni": 6, "Juli": 7, "Agustus": 8,
    "September": 9, "Oktober": 10, "November": 11, "Desember": 12,
}

BULAN_MAP_EN = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def _parse_indo_date(s: str) -> pd.Timestamp:
    """Parse tanggal format 'Januari 2024' atau 'Jan 2024'."""
    try:
        parts = str(s).strip().split()
        if len(parts) == 2:
            bulan = BULAN_MAP.get(parts[0]) or BULAN_MAP_EN.get(parts[0])
            if bulan:
                return pd.Timestamp(year=int(parts[1]), month=bulan, day=1)
    except Exception:
        pass
    return pd.NaT


def _to_float_id(val) -> float:
    """Konversi angka format Indonesia (1.234,56 → 1234.56) ke float."""
    try:
        s = str(val).strip()
        if s in ("-", "", "nan", "None", "-"):
            return np.nan
        s = s.replace("%", "").replace(" ", "")
        # Format Indonesia: titik = ribuan, koma = desimal
        if "," in s and "." in s:
            s = s.replace(".", "").replace(",", ".")
        elif "," in s and "." not in s:
            s = s.replace(",", ".")
        elif "." in s and s.count(".") > 1:
            s = s.replace(".", "")
        return float(s)
    except Exception:
        return np.nan


def _extract_year(filename: str):
    """Ekstrak tahun dari nama file (misalnya '...2024.csv' → 2024)."""
    try:
        stem = os.path.splitext(os.path.basename(filename))[0]
        # Handle '(1)' suffix: ambil token numerik 4-digit terakhir
        for part in reversed(stem.replace("(", " ").replace(")", " ").split()):
            if part.isdigit() and len(part) == 4:
                return int(part)
    except Exception:
        pass
    return None


def _find_indonesia(df: pd.DataFrame, col: int = 0):
    """Cari baris dengan nilai 'INDONESIA' di kolom tertentu."""
    mask = df.iloc[:, col].astype(str).str.upper().str.strip() == "INDONESIA"
    return df[mask].iloc[0] if mask.any() else None


def _normalize_prov(name: str) -> str:
    """Normalisasi nama provinsi ke Title Case standar."""
    mapping = {
        "DKI JAKARTA": "DKI Jakarta",
        "DI YOGYAKARTA": "DI Yogyakarta",
        "ACEH": "Aceh",
        "KEP. BANGKA BELITUNG": "Kepulauan Bangka Belitung",
        "KEP. RIAU": "Kepulauan Riau",
        "KEPULAUAN BANGKA BELITUNG": "Kepulauan Bangka Belitung",
        "KEPULAUAN RIAU": "Kepulauan Riau",
    }
    u = str(name).strip().upper()
    if u in mapping:
        return mapping[u]
    return str(name).strip().title()


# ===========================================================================
# LOADERS
# ===========================================================================

# ---------------------------------------------------------------------------
# [1] Inflasi Bulanan (M-to-M) — backbone time series
# ---------------------------------------------------------------------------
def load_inflasi_mom() -> pd.DataFrame:
    """Inflasi Bulanan M-to-M → Series bulanan level INDONESIA."""
    print("  [1/13] Inflasi Bulanan M-to-M...", end=" ")
    files = glob.glob(os.path.join(BASE, "Inflasi Bulanan", "*.csv"))
    records = []
    for f in sorted(files):
        tahun = _extract_year(f)
        if not tahun:
            continue
        try:
            df = pd.read_csv(f, skiprows=3, header=0, dtype=str, on_bad_lines="skip")
            df.rename(columns={df.columns[0]: "Kota"}, inplace=True)
            row = _find_indonesia(df)
            if row is None:
                continue
            for nama, angka in BULAN_MAP.items():
                if nama in df.columns:
                    val = _to_float_id(row[nama])
                    if not np.isnan(val):
                        records.append({"Tanggal": pd.Timestamp(tahun, angka, 1),
                                        "Inflasi_MoM": val})
        except Exception:
            pass
    df_out = (pd.DataFrame(records)
              .sort_values("Tanggal")
              .drop_duplicates("Tanggal")
              .set_index("Tanggal"))
    print(f"{len(df_out)} baris ({df_out.index.min().year}–{df_out.index.max().year})")
    return df_out


# ---------------------------------------------------------------------------
# [2] Indeks Harga Konsumen (IHK) — level nasional
# ---------------------------------------------------------------------------
def load_ihk() -> pd.DataFrame:
    """IHK Nasional (Umum) → Series bulanan."""
    print("  [2/13] Indeks Harga Konsumen (IHK)...", end=" ")
    files = glob.glob(os.path.join(BASE, "Indeks Harga Konsumen (Umum)", "*.csv"))
    records = []
    for f in sorted(files):
        tahun = _extract_year(f)
        if not tahun:
            continue
        try:
            df = pd.read_csv(f, skiprows=3, header=0, dtype=str, on_bad_lines="skip")
            df.rename(columns={df.columns[0]: "Kota"}, inplace=True)
            row = _find_indonesia(df)
            if row is None:
                continue
            for nama, angka in BULAN_MAP.items():
                if nama in df.columns:
                    val = _to_float_id(row[nama])
                    if not np.isnan(val):
                        records.append({"Tanggal": pd.Timestamp(tahun, angka, 1),
                                        "IHK": val})
        except Exception:
            pass
    df_out = (pd.DataFrame(records)
              .sort_values("Tanggal")
              .drop_duplicates("Tanggal")
              .set_index("Tanggal"))
    print(f"{len(df_out)} baris ({df_out.index.min().year}–{df_out.index.max().year})")
    return df_out


# ---------------------------------------------------------------------------
# [4] BI Rate — suku bunga acuan
# ---------------------------------------------------------------------------
def load_bi_rate() -> pd.DataFrame:
    """BI Rate / Data Inflasi BI → Series bulanan."""
    print("  [4/13] BI Rate (Data Inflasi BI)...", end=" ")
    path = os.path.join(BASE, "BI Rate (Data Inflasi)", "Data Inflasi.xlsx")
    try:
        df = pd.read_excel(path, skiprows=3, header=0)
        if "Periode" not in df.columns:
            df.columns = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)
        df = df[["Periode", "Data Inflasi"]].dropna()
        df["Data Inflasi"] = df["Data Inflasi"].apply(_to_float_id)
        df["Tanggal"] = df["Periode"].apply(_parse_indo_date)
        df = (df.dropna(subset=["Tanggal", "Data Inflasi"])
              .sort_values("Tanggal")
              .set_index("Tanggal")
              [["Data Inflasi"]]
              .rename(columns={"Data Inflasi": "BI_Rate"}))
        print(f"{len(df)} baris ({df.index.min().year}–{df.index.max().year})")
        return df
    except Exception as e:
        print(f"GAGAL – {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# [7] Kurs USD/IDR — bulanan
# ---------------------------------------------------------------------------
def load_usd_idr() -> pd.DataFrame:
    """USD/IDR kurs bulanan dari Investing.com."""
    print("  [7/13] Kurs USD/IDR (bulanan)...", end=" ")
    folder = os.path.join(BASE, "Data Historis USD_IDR")
    # Cari file CSV apapun dalam folder (nama bisa berubah)
    candidates = glob.glob(os.path.join(folder, "*.csv"))
    if not candidates:
        print("GAGAL – tidak ada file CSV")
        return pd.DataFrame()
    path = candidates[0]
    try:
        df = pd.read_csv(path, dtype=str)
        # Rename kolom pertama & kedua
        df.rename(columns={df.columns[0]: "Tanggal", df.columns[1]: "Kurs"}, inplace=True)
        # Parse tanggal format dd/mm/yyyy
        df["Tanggal"] = pd.to_datetime(df["Tanggal"], format="%d/%m/%Y", errors="coerce")
        df["Kurs"] = df["Kurs"].apply(_to_float_id)
        df = df.dropna(subset=["Tanggal", "Kurs"]).set_index("Tanggal").sort_index()
        # Data sudah bulanan - normalize ke awal bulan
        df.index = df.index.normalize()  # set to start of day
        # Resample ke frekuensi bulanan, ambil rata-rata
        df_monthly = df[["Kurs"]].resample("MS").mean().rename(columns={"Kurs": "USD_IDR"})
        print(f"{len(df_monthly)} bulan ({df_monthly.index.min().year}–{df_monthly.index.max().year})")
        return df_monthly
    except Exception as e:
        print(f"GAGAL – {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# [12] Inflasi Umum, Inti, Harga Diatur, Bergejolak — komponen inflasi bulanan
# ---------------------------------------------------------------------------
def load_inflasi_komponen() -> pd.DataFrame:
    """
    Inflasi Umum, Inti, Harga Diatur Pemerintah, dan Bergejolak (M-to-M).
    Format: hierarkis Tahun → Bulan dengan 4 kolom nilai.
    """
    print("  [12/13] Inflasi Komponen (Umum/Inti/Diatur/Bergejolak)...", end=" ")
    folder = "Inflasi Umum, Inti, Harga Diatur Pemerintah, dan Bergejolak Nasional (M-to-M dan Y-to-D)"
    files = glob.glob(os.path.join(BASE, folder, "*.csv"))
    if not files:
        print("GAGAL – file tidak ditemukan")
        return pd.DataFrame()
    records = []
    for fpath in files:
        try:
            df = pd.read_csv(fpath, header=None, dtype=str, on_bad_lines="skip")
            current_year = None
            for _, row in df.iterrows():
                col0 = str(row.iloc[0]).strip() if len(row) > 0 else ""
                col1 = str(row.iloc[1]).strip() if len(row) > 1 else ""
                # Baris tahun: kolom pertama adalah angka 4-digit, kolom kedua kosong/NaN
                if col0.isdigit() and len(col0) == 4:
                    current_year = int(col0)
                    continue
                # Baris bulan: kolom pertama kosong, kolom kedua adalah nama bulan Indonesia
                if col0 in ("", "nan") and col1 in BULAN_MAP and current_year is not None:
                    bulan = BULAN_MAP[col1]
                    vals = []
                    for i in [2, 3, 4, 5]:
                        v = _to_float_id(row.iloc[i]) if len(row) > i else np.nan
                        vals.append(v)
                    records.append({
                        "Tanggal": pd.Timestamp(current_year, bulan, 1),
                        "Inflasi_Umum_MoM": vals[0],
                        "Inflasi_Inti_MoM": vals[1],
                        "Inflasi_HargaDiatur_MoM": vals[2],
                        "Inflasi_Bergejolak_MoM": vals[3],
                    })
        except Exception as err:
            print(f"  (parse error: {err})")
            pass
    if not records:
        print("GAGAL – tidak ada data terparsing")
        return pd.DataFrame()
    df_out = (pd.DataFrame(records)
              .sort_values("Tanggal")
              .drop_duplicates("Tanggal")
              .set_index("Tanggal"))
    print(f"{len(df_out)} baris ({df_out.index.min().year}–{df_out.index.max().year})")
    return df_out


# ---------------------------------------------------------------------------
# [13] Harga Minyak Mentah — bulanan USD/barel
# ---------------------------------------------------------------------------
def load_harga_minyak() -> pd.DataFrame:
    """Harga Bulanan Minyak Mentah (USD/Barel) dari IndexMundi.

    v3: Append data April–Mei 2026 dari Yahoo Finance (WTI) jika tersedia.
    """
    print("  [13/13] Harga Minyak Mentah (USD/Barel)...", end=" ")
    folder = "Harga Bulanan Minyak Mentah (minyak bumi) - Dolar AS per Barel"
    files = glob.glob(os.path.join(BASE, folder, "*.csv"))
    if not files:
        print("GAGAL – file tidak ditemukan")
        return pd.DataFrame()
    try:
        df = pd.read_csv(files[0], dtype=str)
        df["Tanggal"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
        df["Harga_Minyak_USD"] = df["crude_oil_price_usd_per_barrel"].apply(_to_float_id)
        df = (df.dropna(subset=["Tanggal", "Harga_Minyak_USD"])
              .sort_values("Tanggal")
              .set_index("Tanggal")
              [["Harga_Minyak_USD"]])
        # Resample ke awal bulan
        df_monthly = df.resample("MS").mean()

        # Append data April–Mei 2026 dari Yahoo Finance
        wti_path = os.path.join(BASE, "international", "wti_apr_may_2026.csv")
        if os.path.exists(wti_path):
            wti_df = pd.read_csv(wti_path)
            if "Tanggal" in wti_df.columns and "Harga" in wti_df.columns:
                wti_df["Tanggal"] = pd.to_datetime(wti_df["Tanggal"], errors="coerce")
                wti_df = wti_df.dropna(subset=["Tanggal"]).set_index("Tanggal")
                wti_df = wti_df.rename(columns={"Harga": "Harga_Minyak_USD"})
                # Concat (append)
                df_monthly = pd.concat([df_monthly, wti_df[["Harga_Minyak_USD"]]])
                df_monthly = df_monthly[~df_monthly.index.duplicated(keep="last")].sort_index()

        print(f"{len(df_monthly)} bulan ({df_monthly.index.min().year}–{df_monthly.index.max().year})")
        return df_monthly
    except Exception as e:
        print(f"GAGAL – {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# [14] USD/IDR Harian 2026 (Jan–Mei) — append dari Yahoo Finance
# ---------------------------------------------------------------------------
def load_usd_idr_2026() -> pd.DataFrame:
    """USD/IDR harian 2026 (Jan–Mei) dari Yahoo Finance, di-resample ke bulanan."""
    path = os.path.join(BASE, "international", "usd_idr_2026.csv")
    if not os.path.exists(path):
        print("  (file 2026 belum ada, skip)")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        # Rename Date ke Tanggal jika perlu
        if "Tanggal" not in df.columns and "Date" in df.columns:
            df = df.rename(columns={"Date": "Tanggal"})
        if "Tanggal" not in df.columns:
            print("  GAGAL – kolom 'Tanggal'/'Date' tidak ditemukan")
            return pd.DataFrame()
        df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
        # Cari kolom harga
        if "Close" in df.columns:
            price_col = "Close"
        elif "Terakhir" in df.columns:
            price_col = "Terakhir"
        else:
            print("  GAGAL – kolom harga tidak ditemukan")
            return pd.DataFrame()
        df[price_col] = df[price_col].apply(_to_float_id)
        df = df.dropna(subset=["Tanggal", price_col])
        # Resample ke bulanan (ambil rata-rata)
        df_monthly = (df.set_index("Tanggal")[[price_col]]
                      .resample("MS").mean()
                      .rename(columns={price_col: "USD_IDR"}))
        print(f"{len(df_monthly)} bulan ({df_monthly.index.min().year}–{df_monthly.index.max().year})")
        return df_monthly
    except Exception as e:
        print(f"GAGAL – {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# [15] Crude Oil Brent — dari Yahoo Finance (BZ=F)
# ---------------------------------------------------------------------------
def load_brent_oil() -> pd.DataFrame:
    """Crude Oil Brent (USD/Barel) dari Yahoo Finance."""
    path = os.path.join(BASE, "international", "crude_oil_brent.csv")
    if not os.path.exists(path):
        print("  (file belum ada, skip)")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if "Tanggal" not in df.columns and "Date" in df.columns:
            df = df.rename(columns={"Date": "Tanggal"})
        df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
        val_col = [c for c in df.columns if c != "Tanggal"][0]
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        df = df.dropna(subset=["Tanggal", val_col]).set_index("Tanggal")
        df = df.rename(columns={val_col: "Brent_USD"})
        print(f"{len(df)} bulan ({df.index.min().year}–{df.index.max().year})")
        return df[["Brent_USD"]]
    except Exception as e:
        print(f"GAGAL – {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# [16] Indeks Dollar AS (DXY) — dari Yahoo Finance (DX-Y.NYB)
# ---------------------------------------------------------------------------
def load_dxy() -> pd.DataFrame:
    """Indeks Dollar AS (DXY) dari Yahoo Finance."""
    path = os.path.join(BASE, "international", "dxy_dollar_index.csv")
    if not os.path.exists(path):
        print("  (file belum ada, skip)")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if "Tanggal" not in df.columns and "Date" in df.columns:
            df = df.rename(columns={"Date": "Tanggal"})
        df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
        val_col = [c for c in df.columns if c != "Tanggal"][0]
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        df = df.dropna(subset=["Tanggal", val_col]).set_index("Tanggal")
        df = df.rename(columns={val_col: "DXY"})
        print(f"{len(df)} bulan ({df.index.min().year}–{df.index.max().year})")
        return df[["DXY"]]
    except Exception as e:
        print(f"GAGAL – {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# [17] The Fed Funds Rate — dari FRED
# ---------------------------------------------------------------------------
def load_fed_rate() -> pd.DataFrame:
    """The Fed Funds Rate (%) dari FRED."""
    path = os.path.join(BASE, "international", "fed_funds_rate.csv")
    if not os.path.exists(path):
        print("  (file belum ada, skip)")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
        val_col = [c for c in df.columns if c != "Tanggal"][0]
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        df = df.dropna(subset=["Tanggal", val_col]).set_index("Tanggal")
        df = df.rename(columns={val_col: "FedRate_Pct"})
        print(f"{len(df)} bulan ({df.index.min().year}–{df.index.max().year})")
        return df[["FedRate_Pct"]]
    except Exception as e:
        print(f"GAGAL – {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# [18] Gold Price — dari Yahoo Finance (GC=F)
# ---------------------------------------------------------------------------
def load_gold() -> pd.DataFrame:
    """Gold Price (USD/oz) dari Yahoo Finance."""
    path = os.path.join(BASE, "international", "gold_price.csv")
    if not os.path.exists(path):
        print("  (file belum ada, skip)")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if "Tanggal" not in df.columns and "Date" in df.columns:
            df = df.rename(columns={"Date": "Tanggal"})
        df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
        val_col = [c for c in df.columns if c != "Tanggal"][0]
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        df = df.dropna(subset=["Tanggal", val_col]).set_index("Tanggal")
        df = df.rename(columns={val_col: "Gold_USD"})
        print(f"{len(df)} bulan ({df.index.min().year}–{df.index.max().year})")
        return df[["Gold_USD"]]
    except Exception as e:
        print(f"GAGAL – {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# [19] CPO Price (Crude Palm Oil) — dari Yahoo Finance (CPO=F)
# ---------------------------------------------------------------------------
def load_cpo() -> pd.DataFrame:
    """CPO Price (USD/mt) dari Yahoo Finance."""
    path = os.path.join(BASE, "international", "cpo_price.csv")
    if not os.path.exists(path):
        print("  (file belum ada, skip)")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if "Tanggal" not in df.columns and "Date" in df.columns:
            df = df.rename(columns={"Date": "Tanggal"})
        df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
        val_col = [c for c in df.columns if c != "Tanggal"][0]
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        df = df.dropna(subset=["Tanggal", val_col]).set_index("Tanggal")
        df = df.rename(columns={val_col: "CPO_USD"})
        print(f"{len(df)} bulan ({df.index.min().year}–{df.index.max().year})")
        return df[["CPO_USD"]]
    except Exception as e:
        print(f"GAGAL – {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# [20] Geopolitical Risk Index (GPR) — Caldara & Iacoviello
# ---------------------------------------------------------------------------
def load_gpr() -> pd.DataFrame:
    """Geopolitical Risk Index (GPR) dari Caldara & Iacoviello (policyuncertainty.com).
    File CSV dari: https://www.policyuncertainty.com/gpr.html
    Format: kolom 'month' (m/d/yyyy) dan 'GPR' (nilai indeks).
    """
    path = os.path.join(BASE, "international", "data_gpr_export.csv")
    if not os.path.exists(path):
        # Fallback ke template
        path = os.path.join(BASE, "international", "gpr_index.csv")
        if not os.path.exists(path):
            print("  (file belum ada, skip)")
            return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        # Cari kolom tanggal dan GPR
        date_col = None
        for c in ["month", "Month", "Date", "Tanggal"]:
            if c in df.columns:
                date_col = c
                break
        gpr_col = None
        for c in ["GPR", "gpr", "GPR_Index"]:
            if c in df.columns:
                gpr_col = c
                break
        if date_col is None or gpr_col is None:
            print(f"  WARNING – kolom tanggal/GPR tidak ditemukan ({list(df.columns)[:5]})")
            return pd.DataFrame()
        df = df[[date_col, gpr_col]].copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[gpr_col] = pd.to_numeric(df[gpr_col], errors="coerce")
        df = df.dropna(subset=[date_col, gpr_col])
        # Set tanggal ke awal bulan
        df[date_col] = df[date_col].dt.to_period("M").dt.to_timestamp()
        df = df.drop_duplicates(subset=[date_col], keep="last")
        df = df.set_index(date_col)
        df = df.rename(columns={gpr_col: "GPR_Index"})
        print(f"{len(df)} bulan ({df.index.min().year}–{df.index.max().year})")
        return df[["GPR_Index"]]
    except Exception as e:
        print(f"GAGAL – {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# [21] FAO Food Price Index — dari FAO
# ---------------------------------------------------------------------------
def load_fao_fpi() -> pd.DataFrame:
    """FAO Food Price Index dari FAO (ffpi-data-*.xlsx).
    File Excel dari: https://www.fao.org/worldfoodsituation/foodpricesindex/en/
    Format: header di row 2, data mulai row 4, kolom 0=Tanggal, kolom 1=Food Price Index.
    """
    path = os.path.join(BASE, "international", "ffpi-data-2026-05.xlsx")
    if not os.path.exists(path):
        # Fallback ke template CSV
        path = os.path.join(BASE, "international", "fao_food_price_index.csv")
        if not os.path.exists(path):
            print("  (file belum ada, skip)")
            return pd.DataFrame()
    try:
        if path.endswith(".xlsx"):
            df = pd.read_excel(path, header=2)
        else:
            df = pd.read_csv(path, comment="#")
        # Cari kolom Tanggal dan Food Price Index
        date_col = None
        for c in ["Date", "Tanggal", "date", "Month"]:
            if c in df.columns:
                date_col = c
                break
        fpi_col = None
        for c in ["Food Price Index", "FAO_Food_Price_Index", "FPI", "Index"]:
            if c in df.columns:
                fpi_col = c
                break
        if date_col is None or fpi_col is None:
            print(f"  WARNING – kolom Tanggal/FPI tidak ditemukan ({list(df.columns)[:5]})")
            return pd.DataFrame()
        df = df[[date_col, fpi_col]].copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[fpi_col] = pd.to_numeric(df[fpi_col], errors="coerce")
        df = df.dropna(subset=[date_col, fpi_col])
        # Set tanggal ke awal bulan
        df[date_col] = df[date_col].dt.to_period("M").dt.to_timestamp()
        df = df.drop_duplicates(subset=[date_col], keep="last")
        df = df.set_index(date_col)
        df = df.rename(columns={fpi_col: "FAO_FPI"})
        print(f"{len(df)} bulan ({df.index.min().year}–{df.index.max().year})")
        return df[["FAO_FPI"]]
    except Exception as e:
        print(f"GAGAL – {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# [22] Semua Komoditas World Bank CMO (Palm Oil, Coal, Coffee, Wheat, dll)
# ---------------------------------------------------------------------------
# Mapping: nama kolom di CMO -> nama kolom di output
CMO_COLUMNS = {
    "Palm oil":                   "CMO_PalmOil_USD",
    "Coal, Australian":           "CMO_Coal_AU_USD",
    "Coal, South African **":     "CMO_Coal_SA_USD",
    "Coffee, Robusta":            "CMO_Coffee_Robusta_USD",
    "Coffee, Arabica":            "CMO_Coffee_Arabica_USD",
    "Wheat, US SRW":              "CMO_Wheat_SRW_USD",
    "Wheat, US HRW":              "CMO_Wheat_HRW_USD",
    "Soybeans":                   "CMO_Soybeans_USD",
    "Soybean oil":                "CMO_SoybeanOil_USD",
    "Sugar, world":               "CMO_Sugar_USD",
    "Rubber, TSR20 **":           "CMO_Rubber_TSR20_USD",
    "Rubber, RSS3":               "CMO_Rubber_RSS3_USD",
    "Cotton, A Index":            "CMO_Cotton_USD",
    "Rice, Thai 5% ":             "CMO_Rice_Thailand_USD",
    "Coconut oil":                "CMO_CoconutOil_USD",
    "Groundnuts":                 "CMO_Groundnuts_USD",
    "Fish meal":                  "CMO_FishMeal_USD",
    "Maize":                      "CMO_Maize_USD",
    "Tin":                        "CMO_Tin_USD",
    "Nickel":                     "CMO_Nickel_USD",
    "Copper":                     "CMO_Copper_USD",
    "Aluminum":                   "CMO_Aluminum_USD",
    "Iron ore, cfr spot":         "CMO_IronOre_USD",
    "Natural gas, US":            "CMO_NatGas_USD",
    "Natural gas, Europe":        "CMO_NatGas_EU_USD",
    "Liquefied natural gas, Japan": "CMO_LNG_Japan_USD",
}


def load_cmo_commodities() -> pd.DataFrame:
    """Load semua komoditas World Bank CMO (Commodity Markets) sekaligus.
    File: CMO-Historical-Data-Monthly.xlsx
    Sheet: Monthly Prices
    Format tanggal: '1960M01', dst.
    Output: DataFrame dengan kolom per komoditas (USD/mt atau satuan World Bank).
    """
    path = os.path.join(BASE, "international", "CMO-Historical-Data-Monthly.xlsx")
    if not os.path.exists(path):
        print("  (file CMO belum ada, skip)")
        return pd.DataFrame()
    try:
        df = pd.read_excel(path, sheet_name="Monthly Prices", header=4)
        date_col = df.columns[0]

        # Ambil kolom tanggal
        result = pd.DataFrame()
        result["Tanggal"] = pd.to_datetime(df[date_col], format="%YM%m", errors="coerce")
        result = result.dropna(subset=["Tanggal"]).set_index("Tanggal").sort_index()

        loaded = 0
        for cmo_col, out_col in CMO_COLUMNS.items():
            if cmo_col in df.columns:
                vals = pd.to_numeric(df[cmo_col].values, errors="coerce")
                # Buat series dengan tanggal yang sama
                series = pd.Series(vals[:len(result)], index=result.index, name=out_col)
                result[out_col] = series
                loaded += 1

        result = result.dropna(how="all")
        # Forward fill & backward fill untuk setiap kolom
        for col in result.columns:
            result[col] = result[col].ffill().bfill()

        print(f"{loaded} komoditas, {len(result)} bulan ({result.index.min().year}–{result.index.max().year})")
        return result
    except Exception as e:
        print(f"GAGAL – {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# [5] Upah Minimum Provinsi (UMP)
# ---------------------------------------------------------------------------
def load_ump() -> pd.DataFrame:
    """UMP per Provinsi per Tahun."""
    print("  [5/13] Upah Minimum Provinsi (UMP)...", end=" ")
    files = glob.glob(os.path.join(BASE, "Upah Minimum Provinsi", "*.csv"))
    records = []
    for f in sorted(files):
        tahun = _extract_year(f)
        if not tahun:
            continue
        try:
            df = pd.read_csv(f, skiprows=2, header=0, dtype=str, on_bad_lines="skip")
            df.rename(columns={df.columns[0]: "Provinsi", df.columns[1]: "UMP"}, inplace=True)
            df = df.dropna(subset=["Provinsi", "UMP"])
            df = df[~df["Provinsi"].str.strip().str.upper()
                    .isin(["PROVINSI", "INDONESIA", "NASIONAL", ""])]
            df["Provinsi"] = df["Provinsi"].apply(_normalize_prov)
            df["UMP"] = df["UMP"].apply(_to_float_id)
            df["Tahun"] = tahun
            records.append(df[["Provinsi", "UMP", "Tahun"]].dropna())
        except Exception:
            pass
    df_out = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    print(f"{len(df_out)} baris, {df_out['Tahun'].nunique()} tahun")
    return df_out


# ---------------------------------------------------------------------------
# [6] Rata-rata Pengeluaran per Kapita
# ---------------------------------------------------------------------------
def load_pengeluaran() -> pd.DataFrame:
    """Rata-rata Pengeluaran per Kapita per Provinsi per Tahun."""
    print("  [6/13] Pengeluaran per Kapita...", end=" ")
    folder = "Rata-rata Pengeluaran per Kapita Sebulan Makanan dan Bukan Makanan"
    files = glob.glob(os.path.join(BASE, folder, "*.csv"))
    records = []
    for f in sorted(files):
        tahun = _extract_year(f)
        if not tahun:
            continue
        try:
            df = pd.read_csv(f, header=0, dtype=str, on_bad_lines="skip")
            df.rename(columns={df.columns[0]: "Provinsi"}, inplace=True)
            df = df[~df["Provinsi"].str.strip().str.upper()
                    .isin(["PROVINSI", "", "INDONESIA"])]
            df["Provinsi"] = df["Provinsi"].apply(_normalize_prov)
            if len(df.columns) >= 4:
                col_makanan = df.columns[1]
                col_bukan = df.columns[2]
                col_total = df.columns[3]
                df["Pengeluaran_Makanan"] = df[col_makanan].apply(_to_float_id)
                df["Pengeluaran_Bukan_Makanan"] = df[col_bukan].apply(_to_float_id)
                df["Total_Pengeluaran"] = df[col_total].apply(_to_float_id)
            elif len(df.columns) == 3:
                # Beberapa file mungkin hanya punya 3 kolom
                col_makanan = df.columns[1]
                col_bukan = df.columns[2]
                df["Pengeluaran_Makanan"] = df[col_makanan].apply(_to_float_id)
                df["Pengeluaran_Bukan_Makanan"] = df[col_bukan].apply(_to_float_id)
                df["Total_Pengeluaran"] = df["Pengeluaran_Makanan"] + df["Pengeluaran_Bukan_Makanan"]
            df["Tahun"] = tahun
            records.append(df[["Provinsi", "Pengeluaran_Makanan",
                                "Pengeluaran_Bukan_Makanan", "Total_Pengeluaran",
                                "Tahun"]].dropna())
        except Exception:
            pass
    df_out = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    print(f"{len(df_out)} baris, {df_out['Tahun'].nunique()} tahun")
    return df_out


# ---------------------------------------------------------------------------
# [8] Tingkat Pengangguran Terbuka — Semester & Provinsi (Open Data Jabar)
# ---------------------------------------------------------------------------
def load_pengangguran_semester() -> pd.DataFrame:
    """TPT per Provinsi per Tahun (rata-rata Feb+Agustus) — Open Data Jabar."""
    print("  [8/13] TPT Semester & Provinsi (Open Data Jabar)...", end=" ")
    path = os.path.join(
        BASE,
        "Tingkat Pengangguran Terbuka Berdasarkan Semester dan Provinsi di Indonesia",
        "disnakertrans-od_21012_tingkat_pengangguran_terbuka_brdsrkn_semester_prov_v1_data.csv"
    )
    try:
        df = pd.read_csv(path, dtype=str)
        df["tingkat_pengangguran_terbuka"] = df["tingkat_pengangguran_terbuka"].apply(_to_float_id)
        df["tahun"] = df["tahun"].apply(lambda x: int(x) if str(x).isdigit() else np.nan)
        df["nama_provinsi"] = df["nama_provinsi"].apply(_normalize_prov)
        agg = (df.dropna(subset=["tingkat_pengangguran_terbuka", "tahun"])
               .groupby(["nama_provinsi", "tahun"])["tingkat_pengangguran_terbuka"]
               .mean()
               .reset_index()
               .rename(columns={"nama_provinsi": "Provinsi",
                                "tahun": "Tahun",
                                "tingkat_pengangguran_terbuka": "TPT"}))
        agg["Tahun"] = agg["Tahun"].astype(int)
        print(f"{len(agg)} baris, {agg['Tahun'].nunique()} tahun "
              f"({int(agg['Tahun'].min())}–{int(agg['Tahun'].max())})")
        return agg
    except Exception as e:
        print(f"GAGAL – {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# [9] TPT & TPAK Menurut Provinsi — BPS per tahun
# ---------------------------------------------------------------------------
def load_tpt_tpak() -> pd.DataFrame:
    """
    TPT & TPAK per Provinsi per Tahun dari BPS.
    Rata-rata Feb + Agustus untuk TPT; rata-rata Feb + Agustus untuk TPAK.
    """
    print("  [9/13] TPT & TPAK Menurut Provinsi (BPS)...", end=" ")
    folder = "Tingkat Pengangguran Terbuka (TPT) dan Tingkat Partisipasi Angkatan Kerja (TPAK) Menurut Provinsi"
    files = glob.glob(os.path.join(BASE, folder, "*.csv"))
    records = []
    for f in sorted(files):
        tahun = _extract_year(f)
        if not tahun:
            continue
        try:
            df = pd.read_csv(f, header=0, dtype=str, on_bad_lines="skip")
            df.rename(columns={df.columns[0]: "Provinsi"}, inplace=True)
            # Filter baris meta/catatan
            df = df[~df["Provinsi"].str.strip().str.upper()
                    .isin(["PROVINSI", "", "INDONESIA", "CATATAN"])]
            df = df[~df["Provinsi"].str.startswith("<sup", na=False)]
            df = df[~df["Provinsi"].str.startswith("Catatan", na=False)]
            df["Provinsi"] = df["Provinsi"].apply(_normalize_prov)
            # Cari kolom TPT dan TPAK
            cols = df.columns.tolist()
            tpt_cols = [c for c in cols if "TPT" in str(c).upper() or "Pengangguran" in str(c)]
            tpak_cols = [c for c in cols if "TPAK" in str(c).upper() or "Partisipasi" in str(c)]
            # Hitung rata-rata dari semua kolom TPT dan TPAK
            for _, row in df.iterrows():
                prov = row["Provinsi"]
                tpt_vals = [_to_float_id(row[c]) for c in tpt_cols if c in row]
                tpak_vals = [_to_float_id(row[c]) for c in tpak_cols if c in row]
                tpt_mean = np.nanmean(tpt_vals) if tpt_vals else np.nan
                tpak_mean = np.nanmean(tpak_vals) if tpak_vals else np.nan
                if not np.isnan(tpt_mean) or not np.isnan(tpak_mean):
                    records.append({
                        "Provinsi": prov,
                        "Tahun": tahun,
                        "TPT_BPS": round(tpt_mean, 4) if not np.isnan(tpt_mean) else np.nan,
                        "TPAK_BPS": round(tpak_mean, 4) if not np.isnan(tpak_mean) else np.nan,
                    })
        except Exception:
            pass
    df_out = (pd.DataFrame(records)
              .dropna(subset=["Provinsi"])
              .drop_duplicates(["Provinsi", "Tahun"])
              .sort_values(["Tahun", "Provinsi"])
              .reset_index(drop=True))
    if not df_out.empty:
        print(f"{len(df_out)} baris, {df_out['Tahun'].nunique()} tahun "
              f"({int(df_out['Tahun'].min())}–{int(df_out['Tahun'].max())})")
    else:
        print("GAGAL – tidak ada data")
    return df_out


# ---------------------------------------------------------------------------
# [10] PDRB Per Kapita (Ribu Rupiah) — BPS per tahun
# ---------------------------------------------------------------------------
def load_pdrb() -> pd.DataFrame:
    """PDRB Per Kapita per Provinsi per Tahun (Harga Berlaku & Konstan)."""
    print("  [10/13] PDRB Per Kapita (Ribu Rp)...", end=" ")
    folder = "Produk Domestik Regional Bruto Per Kapita (Ribu Rupiah)"
    files = glob.glob(os.path.join(BASE, folder, "*.csv"))
    records = []
    for f in sorted(files):
        tahun = _extract_year(f)
        if not tahun:
            continue
        try:
            df = pd.read_csv(f, skiprows=4, header=None, dtype=str, on_bad_lines="skip")
            if df.empty or len(df.columns) < 2:
                continue
            df.rename(columns={df.columns[0]: "Provinsi"}, inplace=True)
            df = df[df["Provinsi"].str.strip() != ""]
            df = df[~df["Provinsi"].str.strip().str.upper()
                    .isin(["INDONESIA", ""])]
            df["Provinsi"] = df["Provinsi"].apply(_normalize_prov)
            # Kolom 1 = Harga Berlaku, Kolom 2 = Harga Konstan 2010
            col_berlaku = df.columns[1] if len(df.columns) > 1 else None
            col_konstan = df.columns[2] if len(df.columns) > 2 else None
            for _, row in df.iterrows():
                prov = row["Provinsi"]
                berlaku = _to_float_id(row[col_berlaku]) if col_berlaku else np.nan
                konstan = _to_float_id(row[col_konstan]) if col_konstan else np.nan
                if not np.isnan(berlaku) or not np.isnan(konstan):
                    records.append({
                        "Provinsi": prov,
                        "Tahun": tahun,
                        "PDRB_HargaBerlaku": berlaku,
                        "PDRB_HargaKonstan": konstan,
                    })
        except Exception:
            pass
    df_out = (pd.DataFrame(records)
              .dropna(subset=["Provinsi"])
              .drop_duplicates(["Provinsi", "Tahun"])
              .sort_values(["Tahun", "Provinsi"])
              .reset_index(drop=True))
    if not df_out.empty:
        print(f"{len(df_out)} baris, {df_out['Tahun'].nunique()} tahun "
              f"({int(df_out['Tahun'].min())}–{int(df_out['Tahun'].max())})")
    else:
        print("GAGAL – tidak ada data")
    return df_out


# ---------------------------------------------------------------------------
# [11] Persentase Penduduk Miskin per Provinsi
# ---------------------------------------------------------------------------
def load_penduduk_miskin() -> pd.DataFrame:
    """Persentase Penduduk Miskin per Provinsi per Tahun."""
    print("  [11/13] Persentase Penduduk Miskin...", end=" ")
    folder = "Persentase Penduduk Miskin Berdasarkan Provinsi di Indonesia"
    files = glob.glob(os.path.join(BASE, folder, "*_data.csv"))
    if not files:
        files = glob.glob(os.path.join(BASE, folder, "*.csv"))
    if not files:
        print("GAGAL – file tidak ditemukan")
        return pd.DataFrame()
    try:
        df = pd.read_csv(files[0], dtype=str)
        # Kolom: id, kode_provinsi, nama_provinsi, persentase_penduduk_miskin, satuan, tahun
        df["persentase_penduduk_miskin"] = df["persentase_penduduk_miskin"].apply(_to_float_id)
        df["tahun"] = df["tahun"].apply(lambda x: int(x) if str(x).strip().isdigit() else np.nan)
        df["nama_provinsi"] = df["nama_provinsi"].apply(_normalize_prov)
        # Filter nilai 0.0 yang mengindikasikan data tidak tersedia (provinsi baru)
        df = df[df["persentase_penduduk_miskin"] > 0]
        df_out = (df.dropna(subset=["persentase_penduduk_miskin", "tahun"])
                  .rename(columns={
                      "nama_provinsi": "Provinsi",
                      "tahun": "Tahun",
                      "persentase_penduduk_miskin": "Pct_Penduduk_Miskin"
                  })
                  [["Provinsi", "Tahun", "Pct_Penduduk_Miskin"]]
                  .drop_duplicates(["Provinsi", "Tahun"])
                  .sort_values(["Tahun", "Provinsi"])
                  .reset_index(drop=True))
        df_out["Tahun"] = df_out["Tahun"].astype(int)
        print(f"{len(df_out)} baris, {df_out['Tahun'].nunique()} tahun "
              f"({int(df_out['Tahun'].min())}–{int(df_out['Tahun'].max())})")
        return df_out
    except Exception as e:
        print(f"GAGAL – {e}")
        return pd.DataFrame()


# ===========================================================================
# BUILD OUTPUT 1: clean_inflasi_ts.csv
# Time-series bulanan untuk Model 1 (LSTM Forecasting)
# ===========================================================================

def build_inflasi_ts(inflasi, ihk, bi_rate, usd_idr,
                     inflasi_komp, harga_minyak,
                     usd_idr_2026=None,
                     brent=None, dxy=None, fed_rate=None,
                     gold=None, cpo=None, gpr=None,
                     fao_fpi=None, cmo_all=None) -> pd.DataFrame:
    """
    Gabungkan semua fitur time-series bulanan.

    Fitur domestik:
    - Inflasi MoM (backbone, target)
    - IHK (2005–2023 lengkap; 2024–2026 diimputasi dari Inflasi MoM)
    - BI Rate, USD/IDR
    - Inflasi Komponen (Inti, Harga Diatur, Bergejolak)
    - Harga Minyak Mentah (WTI)

    Fitur internasional:
    - Brent Oil, DXY, Fed Funds Rate, Gold, CPO
    - Geopolitical Risk Index (GPR)
    - FAO Food Price Index
    - 24 komoditas World Bank CMO (Palm Oil, Coal, Coffee, Wheat, dll)
    """
    print("\n▶ Membangun clean_inflasi_ts.csv ...")

    ts = inflasi.copy()

    # Merge IHK
    if not ihk.empty:
        ts = ts.join(ihk, how="left")

    # Merge BI Rate
    if not bi_rate.empty:
        ts = ts.join(bi_rate, how="left")

    # Merge USD/IDR (dari Investing.com, sampai 2025-12)
    if not usd_idr.empty:
        ts = ts.join(usd_idr, how="left")

    # Merge USD/IDR 2026 (dari Yahoo, Jan–Mei 2026) — append/update baris 2026
    if usd_idr_2026 is not None and not usd_idr_2026.empty:
        for idx, row in usd_idr_2026.iterrows():
            if idx in ts.index and "USD_IDR" in ts.columns:
                ts.loc[idx, "USD_IDR"] = row["USD_IDR"]
            elif "USD_IDR" in ts.columns:
                ts.loc[idx, "USD_IDR"] = row["USD_IDR"]

    # Merge Inflasi Komponen
    if not inflasi_komp.empty:
        ts = ts.join(inflasi_komp, how="left")

    # Merge Harga Minyak Mentah (WTI dari IndexMundi)
    if not harga_minyak.empty:
        ts = ts.join(harga_minyak, how="left")

    # Merge fitur internasional
    for new_df, name in [
        (brent, "Brent_USD"),
        (dxy, "DXY"),
        (fed_rate, "FedRate_Pct"),
        (gold, "Gold_USD"),
        (cpo, "CPO_USD"),
        (gpr, "GPR_Index"),
        (fao_fpi, "FAO_FPI"),
    ]:
        if new_df is not None and not new_df.empty and name in new_df.columns:
            ts = ts.join(new_df[[name]], how="left")

    # Merge semua komoditas CMO (24 kolom)
    if cmo_all is not None and not cmo_all.empty:
        ts = ts.join(cmo_all, how="left")

    # Tambahkan fitur waktu (aman dari leakage)
    ts["Bulan"] = ts.index.month
    ts["Tahun"] = ts.index.year

    # --- Imputasi nilai null untuk data 2026 (Maret–Mei) ---
    # 1. Inflasi_MoM: ambil dari Inflasi_Umum_MoM (mereka identik untuk MoM)
    if "Inflasi_Umum_MoM" in ts.columns and "Inflasi_MoM" in ts.columns:
        mask_null = ts["Inflasi_MoM"].isna() & ts["Inflasi_Umum_MoM"].notna()
        if mask_null.any():
            ts.loc[mask_null, "Inflasi_MoM"] = ts.loc[mask_null, "Inflasi_Umum_MoM"]
            print(f"   [IMPUTASI] Inflasi_MoM diisi dari Inflasi_Umum_MoM: {mask_null.sum()} baris")

    # 2. Inflasi Komponen: forward fill (sama dengan nilai bulan lalu)
    for col in ["Inflasi_Umum_MoM", "Inflasi_Inti_MoM",
                "Inflasi_HargaDiatur_MoM", "Inflasi_Bergejolak_MoM"]:
        if col in ts.columns:
            null_before = ts[col].isna().sum()
            ts[col] = ts[col].ffill().bfill()
            null_after = ts[col].isna().sum()
            if null_after < null_before:
                print(f"   [IMPUTASI] {col} ffilled: {null_before - null_after} baris")

    # 3. BI Rate: forward fill (BI Rate jarang berubah drastis)
    if "BI_Rate" in ts.columns:
        null_before = ts["BI_Rate"].isna().sum()
        ts["BI_Rate"] = ts["BI_Rate"].ffill().bfill()
        null_after = ts["BI_Rate"].isna().sum()
        if null_after < null_before:
            print(f"   [IMPUTASI] BI_Rate ffilled: {null_before - null_after} baris")

    # 4. IHK: imputasi 2024–2026 pakai rumus IHK_prev × (1 + Inflasi/100)
    if "IHK" in ts.columns and "Inflasi_MoM" in ts.columns:
        ihk_null_mask = ts["IHK"].isna()
        if ihk_null_mask.any():
            count = 0
            for idx in ts[ihk_null_mask].index:
                pos = ts.index.get_loc(idx)
                if pos > 0 and not np.isnan(ts.iloc[pos - 1]["IHK"]):
                    inflasi = ts.loc[idx, "Inflasi_MoM"]
                    if not np.isnan(inflasi):
                        ts.loc[idx, "IHK"] = ts.iloc[pos - 1]["IHK"] * (1 + inflasi / 100)
                        count += 1
            ts["IHK"] = ts["IHK"].ffill().bfill()
            print(f"   [IMPUTASI] IHK diestimasi dari IHK_prev × (1 + Inflasi): {count} baris")

    # 5. GPR Index: forward fill
    if "GPR_Index" in ts.columns:
        null_before = ts["GPR_Index"].isna().sum()
        ts["GPR_Index"] = ts["GPR_Index"].ffill().bfill()
        null_after = ts["GPR_Index"].isna().sum()
        if null_after < null_before:
            print(f"   [IMPUTASI] GPR_Index ffilled: {null_before - null_after} baris")

    # 6. Brent, CPO, dll: forward fill (mulai data setelah 2005)
    for col in ["Brent_USD", "DXY", "FedRate_Pct", "Gold_USD", "CPO_USD", "FAO_FPI"]:
        if col in ts.columns:
            null_before = ts[col].isna().sum()
            ts[col] = ts[col].ffill().bfill()
            null_after = ts[col].isna().sum()
            if null_after < null_before:
                print(f"   [IMPUTASI] {col} ffilled: {null_before - null_after} baris")

    # 6b. Semua kolom CMO: forward fill
    cmo_cols = [c for c in ts.columns if c.startswith("CMO_")]
    for col in cmo_cols:
        null_before = ts[col].isna().sum()
        ts[col] = ts[col].ffill().bfill()
        null_after = ts[col].isna().sum()
        if null_after < null_before:
            print(f"   [IMPUTASI] {col} ffilled: {null_before - null_after} baris")

    # 7. Drop baris dengan Inflasi_MoM null (tidak bisa diimputasi untuk target)
    if "Inflasi_MoM" in ts.columns:
        before = len(ts)
        ts = ts.dropna(subset=["Inflasi_MoM"])
        dropped = before - len(ts)
        if dropped > 0:
            print(f"   [DROP] Baris tanpa Inflasi_MoM: {dropped} (Mei 2026 dst yang belum rilis)")

    # Reset index agar Tanggal menjadi kolom biasa
    ts = ts.reset_index()

    out_path = os.path.join(OUT_DIR, "clean_inflasi_ts.csv")
    ts.to_csv(out_path, index=False)

    print(f"   ✓ {len(ts)} baris × {len(ts.columns)} kolom")
    print(f"   ✓ Rentang: {ts['Tanggal'].min().strftime('%b %Y')} – {ts['Tanggal'].max().strftime('%b %Y')}")
    print(f"   ✓ Kolom: {list(ts.columns)}")
    print(f"   ✓ Disimpan → {out_path}")
    return ts


# ===========================================================================
# BUILD OUTPUT 2: clean_daya_beli.csv
# Panel data provinsi untuk Model 2 (Regresi Daya Beli)
# ===========================================================================

def build_daya_beli_panel(inflasi, ump, pengeluaran,
                          pengangguran_sem, tpt_tpak,
                          pdrb, penduduk_miskin) -> pd.DataFrame:
    """
    Panel data provinsi × tahun:
    - Pengeluaran per Kapita (target Y)
    - UMP
    - TPT (dari TPT-BPS, fallback ke Semester)
    - TPAK
    - PDRB per Kapita
    - Persentase Penduduk Miskin
    - Inflasi rata-rata tahunan (dari Inflasi MoM)
    """
    print("\n▶ Membangun clean_daya_beli.csv ...")

    # --- Inflasi → rata-rata tahunan ---
    inflasi_tahunan = (inflasi.reset_index()
                       .assign(Tahun=lambda x: x["Tanggal"].dt.year)
                       .groupby("Tahun")["Inflasi_MoM"]
                       .mean()
                       .reset_index()
                       .rename(columns={"Inflasi_MoM": "Inflasi_Rata_Tahunan"}))

    # --- Normalisasi nama provinsi di semua dataset ---
    def norm_prov(df, col="Provinsi"):
        df = df.copy()
        df[col] = df[col].apply(_normalize_prov)
        return df

    pen_c = norm_prov(pengeluaran)
    ump_c = norm_prov(ump)
    tpt_sem_c = norm_prov(pengangguran_sem) if not pengangguran_sem.empty else pd.DataFrame()
    tpt_bps_c = norm_prov(tpt_tpak) if not tpt_tpak.empty else pd.DataFrame()
    pdrb_c = norm_prov(pdrb) if not pdrb.empty else pd.DataFrame()
    miskin_c = norm_prov(penduduk_miskin) if not penduduk_miskin.empty else pd.DataFrame()

    # --- Merge panel ---
    panel = pen_c.merge(ump_c, on=["Provinsi", "Tahun"], how="left")

    # TPT: gabungkan dari BPS TPT-TPAK (lebih detail) & fallback ke Semester
    if not tpt_bps_c.empty:
        panel = panel.merge(tpt_bps_c[["Provinsi", "Tahun", "TPT_BPS", "TPAK_BPS"]],
                            on=["Provinsi", "Tahun"], how="left")
    if not tpt_sem_c.empty:
        panel = panel.merge(tpt_sem_c[["Provinsi", "Tahun", "TPT"]],
                            on=["Provinsi", "Tahun"], how="left")

    # Buat kolom TPT_Final: gunakan TPT_BPS jika tersedia, fallback ke TPT semester
    if "TPT_BPS" in panel.columns and "TPT" in panel.columns:
        panel["TPT_Final"] = panel["TPT_BPS"].fillna(panel["TPT"])
        panel.drop(columns=["TPT_BPS", "TPT"], inplace=True)
        panel.rename(columns={"TPT_Final": "TPT"}, inplace=True)
    elif "TPT_BPS" in panel.columns:
        panel.rename(columns={"TPT_BPS": "TPT"}, inplace=True)

    # TPAK
    if "TPAK_BPS" in panel.columns:
        panel.rename(columns={"TPAK_BPS": "TPAK"}, inplace=True)

    # PDRB
    if not pdrb_c.empty:
        panel = panel.merge(pdrb_c[["Provinsi", "Tahun",
                                    "PDRB_HargaBerlaku", "PDRB_HargaKonstan"]],
                            on=["Provinsi", "Tahun"], how="left")

    # Persentase Penduduk Miskin
    if not miskin_c.empty:
        panel = panel.merge(miskin_c[["Provinsi", "Tahun", "Pct_Penduduk_Miskin"]],
                            on=["Provinsi", "Tahun"], how="left")

    # Inflasi rata-rata tahunan
    panel = panel.merge(inflasi_tahunan, on="Tahun", how="left")

    # --- Filter tahun yang memiliki data inti lengkap ---
    # Overlap: Pengeluaran (2017–2025) + UMP (2021–2025) = 2021–2025
    panel = panel[panel["Tahun"].between(2021, 2025)]

    # --- Drop baris dengan kolom kunci kosong ---
    key_cols = ["Total_Pengeluaran", "UMP", "Inflasi_Rata_Tahunan"]
    panel = panel.dropna(subset=key_cols)

    # --- Urutan kolom final ---
    col_order = [
        "Provinsi", "Tahun",
        "Pengeluaran_Makanan", "Pengeluaran_Bukan_Makanan", "Total_Pengeluaran",
        "UMP", "TPT", "TPAK",
        "PDRB_HargaBerlaku", "PDRB_HargaKonstan",
        "Pct_Penduduk_Miskin",
        "Inflasi_Rata_Tahunan",
    ]
    col_order = [c for c in col_order if c in panel.columns]
    panel = panel[col_order].sort_values(["Tahun", "Provinsi"]).reset_index(drop=True)

    out_path = os.path.join(OUT_DIR, "clean_daya_beli.csv")
    panel.to_csv(out_path, index=False)

    print(f"   ✓ {len(panel)} baris × {len(panel.columns)} kolom")
    print(f"   ✓ Provinsi: {panel['Provinsi'].nunique()}, "
          f"Tahun: {sorted(panel['Tahun'].unique())}")
    print(f"   ✓ Kolom: {list(panel.columns)}")
    print(f"   ✓ Disimpan → {out_path}")
    return panel


# ===========================================================================
# SUMMARY HELPER
# ===========================================================================

def print_summary(df: pd.DataFrame, name: str):
    print(f"\n{'─'*65}")
    print(f"  Preview: {name}")
    print(f"{'─'*65}")
    print(f"  Shape   : {df.shape}")
    print(f"  Kolom   : {list(df.columns)}")
    null_dict = df.isnull().sum().to_dict()
    null_str = {k: v for k, v in null_dict.items() if v > 0}
    print(f"  Null (non-zero): {null_str if null_str else 'tidak ada'}")
    print(f"\n  5 baris pertama:")
    print(df.head().to_string(index=False))


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 65)
    print("  PREPROCESSING PIPELINE v3 – Kelompok E ML UNAIR")
    print("=" * 65)
    print("\n>> Memuat semua dataset raw...\n")

    # --- Lokal (existing) ---
    print("[LOKAL]")
    inflasi       = load_inflasi_mom()        # [1]
    ihk           = load_ihk()                # [2]
    # [3] Inflasi Y-to-D: referensi saja, tidak dimasukkan ke model
    bi_rate       = load_bi_rate()            # [4]
    ump           = load_ump()                # [5]
    pengeluaran   = load_pengeluaran()        # [6]
    usd_idr       = load_usd_idr()            # [7]
    pengangguran_sem = load_pengangguran_semester()  # [8]
    tpt_tpak      = load_tpt_tpak()           # [9]
    pdrb          = load_pdrb()               # [10]
    penduduk_miskin = load_penduduk_miskin()  # [11]
    inflasi_komp  = load_inflasi_komponen()   # [12]
    harga_minyak  = load_harga_minyak()       # [13]

    # --- Lokal (BARU v3) ---
    print("\n[LOKAL - BARU]")
    usd_idr_2026  = load_usd_idr_2026()       # [14] USD/IDR Jan–Mei 2026

    # --- Internasional (BARU v3) ---
    print("\n[INTERNASIONAL - BARU]")
    brent         = load_brent_oil()          # [15]
    dxy           = load_dxy()                # [16]
    fed_rate      = load_fed_rate()           # [17]
    gold          = load_gold()               # [18]
    cpo           = load_cpo()                # [19]
    gpr           = load_gpr()                # [20] Geopolitical Risk Index
    fao_fpi       = load_fao_fpi()            # [21] FAO Food Price Index
    cmo_all       = load_cmo_commodities()    # [22] Semua komoditas World Bank CMO

    # Build output files
    print("\n[BUILD]")
    ts    = build_inflasi_ts(inflasi, ihk, bi_rate, usd_idr,
                              inflasi_komp, harga_minyak,
                              usd_idr_2026=usd_idr_2026,
                              brent=brent, dxy=dxy, fed_rate=fed_rate,
                              gold=gold, cpo=cpo, gpr=gpr,
                              fao_fpi=fao_fpi, cmo_all=cmo_all)
    panel = build_daya_beli_panel(inflasi, ump, pengeluaran,
                                   pengangguran_sem, tpt_tpak,
                                   pdrb, penduduk_miskin)

    # Ringkasan
    print_summary(ts, "clean_inflasi_ts.csv")
    print_summary(panel, "clean_daya_beli.csv")

    print(f"\n{'='*65}")
    print("  ✅ Preprocessing selesai!")
    print(f"  Output disimpan di: datasets/processed/")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
