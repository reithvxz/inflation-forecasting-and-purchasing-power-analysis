"""
============================================================================
  DOWNLOAD INTERNATIONAL DATASETS
  Proyek: Prediksi Inflasi dan Dampaknya terhadap Daya Beli
============================================================================
Script ini mendownload dataset internasional yang relevan untuk prediksi
inflasi Indonesia dari Yahoo Finance, FRED, dan World Bank.

Output: datasets/international/
  1. crude_oil_brent.csv
  2. dxy_dollar_index.csv
  3. fed_funds_rate.csv
  4. gold_price.csv
  5. fao_food_price_index.csv
  6. rice_price_thailand.csv
  7. cpo_price.csv
============================================================================
"""

import os
import time
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "datasets", "international")
os.makedirs(OUT_DIR, exist_ok=True)

START_DATE = "2003-01-01"
END_DATE = "2026-05-31"


def download_yahoo(symbol, name, col_name, start=START_DATE, end=END_DATE):
    """Download data dari Yahoo Finance, simpan sebagai CSV bulanan."""
    out_path = os.path.join(OUT_DIR, f"{name}.csv")
    try:
        import yfinance as yf
        print(f"  Downloading {name} ({symbol})...")
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
        if df.empty:
            print(f"  [GAGAL] {name}: data kosong")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Resample ke bulanan (ambil rata-rata)
        df_monthly = df["Close"].resample("MS").mean().reset_index()
        df_monthly.columns = ["Tanggal", col_name]
        df_monthly["Tanggal"] = pd.to_datetime(df_monthly["Tanggal"])
        df_monthly[col_name] = df_monthly[col_name].round(4)

        df_monthly.to_csv(out_path, index=False)
        print(f"  [OK] {name}: {len(df_monthly)} baris ({df_monthly['Tanggal'].min().date()} s.d {df_monthly['Tanggal'].max().date()})")
        return df_monthly
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")
        return None


def download_fred(series_id, name, col_name, start=START_DATE, end=END_DATE):
    """Download data dari FRED (Federal Reserve Economic Data) tanpa API key."""
    out_path = os.path.join(OUT_DIR, f"{name}.csv")
    try:
        import datetime
        import pandas_datareader.data as web
        start_dt = datetime.datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.datetime.strptime(end, "%Y-%m-%d")
        df = web.DataReader(series_id, "fred", start_dt, end_dt)
        if df is None or df.empty:
            print(f"  [GAGAL] {name}: FRED tidak merespons")
            return None

        df = df.reset_index()
        df.columns = ["Tanggal", col_name]
        df["Tanggal"] = pd.to_datetime(df["Tanggal"])
        df[col_name] = df[col_name].round(4)
        df = df.sort_values("Tanggal").reset_index(drop=True)

        df.to_csv(out_path, index=False)
        print(f"  [OK] {name}: {len(df)} baris ({df['Tanggal'].min().date()} s.d {df['Tanggal'].max().date()})")
        return df
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")
        return None


def main():
    print("=" * 60)
    print("  Download International Datasets")
    print("=" * 60)

    # 1. Crude Oil Brent (Yahoo Finance: BZ=F)
    download_yahoo("BZ=F", "crude_oil_brent", "Brent_USD", start=START_DATE, end=END_DATE)
    time.sleep(1)

    # 2. Dollar Index DXY (Yahoo Finance: DX-Y.NYB)
    download_yahoo("DX-Y.NYB", "dxy_dollar_index", "DXY", start=START_DATE, end=END_DATE)
    time.sleep(1)

    # 3. Fed Funds Rate (FRED: DFF atau FEDFUNDS)
    #    Coba dulu FRED; jika gagal, fallback ke Yahoo
    result = download_fred("FEDFUNDS", "fed_funds_rate", "FedRate_Pct")
    if result is None:
        # Fallback: manual CSV (dari FRED website) atau estimasi
        print("  [INFO] Fed Rate perlu API key FRED atau download manual")

    # 4. Gold Price (Yahoo Finance: GC=F)
    download_yahoo("GC=F", "gold_price", "Gold_USD", start=START_DATE, end=END_DATE)
    time.sleep(1)

    # 5. FAO Food Price Index (download dari FAO website atau World Bank)
    #    FAO URL: https://www.fao.org/worldfoodsituation/foodpricesindex/en/
    fao_path = os.path.join(OUT_DIR, "fao_food_price_index.csv")
    if not os.path.exists(fao_path):
        # Buat template kosong; user bisa isi manual dari FAO
        template = pd.DataFrame(columns=["Tanggal", "FAO_Food_Price_Index"])
        template.to_csv(fao_path, index=False)
        print(f"  [TEMPLATE] {fao_path} dibuat. Isi manual dari FAO website.")

    # 6. Rice Price Thailand 5% (World Bank Pink Sheet / FAO)
    rice_path = os.path.join(OUT_DIR, "rice_price_thailand.csv")
    if not os.path.exists(rice_path):
        template = pd.DataFrame(columns=["Tanggal", "Rice_USD_per_ton"])
        template.to_csv(rice_path, index=False)
        print(f"  [TEMPLATE] {rice_path} dibuat. Isi manual dari World Bank Pink Sheet.")

    # 7. CPO Price (Yahoo Finance: CPO=F)
    download_yahoo("CPO=F", "cpo_price", "CPO_USD", start=START_DATE, end=END_DATE)
    time.sleep(1)

    print("=" * 60)
    print("  Download selesai!")
    print(f"  Output: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
