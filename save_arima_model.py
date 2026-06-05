# -*- coding: utf-8 -*-
"""
============================================================================
  ARIMA Model Training Script
  Prediksi Inflasi Menggunakan ARIMA (AutoRegressive Integrated Moving Average)
  Dengan grid search untuk parameter optimal (p, d, q)
  
  Output:
    - models/arima_inflasi.pkl          → Model ARIMA terbaik
    - models/arima_metrics.pkl          → Metrik evaluasi
    - models/arima_forecast.pkl         → Forecast results
============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import pickle
from itertools import product
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

DATA_PATH = "datasets/processed/clean_inflasi_ts.csv"
MODEL_PATH = "models/arima_inflasi.pkl"
METRICS_PATH = "models/arima_metrics.pkl"
FORECAST_PATH = "models/arima_forecast.pkl"
PLOTS_DIR = "models/arima_plots"

os.makedirs("models", exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------
def load_data() -> pd.Series:
    """Load and prepare inflation time series data."""
    print("=" * 60)
    print("ARIMA INFLATION FORECASTING")
    print("=" * 60)
    
    print("\n[1/6] Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df = df.sort_values('Tanggal').reset_index(drop=True)
    df.set_index('Tanggal', inplace=True)
    
    # Use Inflasi_MoM as target
    ts = df['Inflasi_MoM'].dropna()
    print(f"  Data loaded: {len(ts)} observations")
    print(f"  Date range: {ts.index.min()} to {ts.index.max()}")
    print(f"  Mean: {ts.mean():.4f}%")
    print(f"  Std: {ts.std():.4f}%")
    
    return ts


# ---------------------------------------------------------------------------
# 2. Stationarity Test
# ---------------------------------------------------------------------------
def check_stationarity(ts: pd.Series, max_diff: int = 2) -> int:
    """Augmented Dickey-Fuller test. Returns recommended differencing order."""
    print("\n[2/6] Checking stationarity (ADF test)...")
    
    for d in range(max_diff + 1):
        if d == 0:
            data = ts
            label = "Original"
        else:
            data = ts.diff(d).dropna()
            label = f"Differenced (d={d})"
        
        result = adfuller(data.dropna())
        p_value = result[1]
        is_stationary = p_value <= 0.05
        
        status = "✓ STATIONER" if is_stationary else "✗ Belum stationer"
        print(f"  {label}: ADF={result[0]:.4f}, p={p_value:.4f} → {status}")
        
        if is_stationary:
            print(f"  → Rekomendasi: d = {d}")
            return d
    
    print(f"  → Data belum stationer setelah {max_diff} differencing, gunakan d={max_diff}")
    return max_diff


# ---------------------------------------------------------------------------
# 3. Grid Search ARIMA(p,d,q)
# ---------------------------------------------------------------------------
def grid_search_arima(ts: pd.Series, d: int,
                      p_range=range(0, 5), q_range=range(0, 5)) -> dict:
    """Grid search for optimal ARIMA(p,d,q) parameters."""
    print(f"\n[3/6] Grid Search ARIMA(p,{d},q)...")
    
    # Train/test split (80/20)
    split_idx = int(len(ts) * 0.8)
    train = ts.iloc[:split_idx]
    test = ts.iloc[split_idx:]
    
    print(f"  Train: {len(train)} obs ({train.index.min().strftime('%Y-%m')} to {train.index.max().strftime('%Y-%m')})")
    print(f"  Test:  {len(test)} obs ({test.index.min().strftime('%Y-%m')} to {test.index.max().strftime('%Y-%m')})")
    
    best_aic = float('inf')
    best_order = (1, d, 1)
    best_model = None
    all_results = []
    
    total = len(list(product(p_range, q_range)))
    count = 0
    
    for p, q in product(p_range, q_range):
        count += 1
        try:
            model = ARIMA(train, order=(p, d, q))
            fitted = model.fit()
            aic = fitted.aic
            
            # Forecast on test set
            pred = fitted.forecast(steps=len(test))
            mae = mean_absolute_error(test, pred)
            rmse = np.sqrt(mean_squared_error(test, pred))
            
            all_results.append({
                'order': (p, d, q),
                'aic': aic,
                'mae': mae,
                'rmse': rmse
            })
            
            if aic < best_aic:
                best_aic = aic
                best_order = (p, d, q)
                best_model = fitted
            
            if count % 10 == 0:
                print(f"  Progress: {count}/{total} combinations tested...")
                
        except Exception:
            continue
    
    # Sort by AIC
    all_results.sort(key=lambda x: x['aic'])
    
    print(f"\n  === Top 5 Models ===")
    print(f"  {'Order':<15} {'AIC':>10} {'MAE':>10} {'RMSE':>10}")
    print(f"  {'-'*45}")
    for r in all_results[:5]:
        print(f"  {str(r['order']):<15} {r['aic']:>10.2f} {r['mae']:>10.4f} {r['rmse']:>10.4f}")
    
    print(f"\n  ✓ Best: ARIMA{best_order} (AIC={best_aic:.2f})")
    
    return {
        'best_model': best_model,
        'best_order': best_order,
        'best_aic': best_aic,
        'train': train,
        'test': test,
        'all_results': all_results
    }


# ---------------------------------------------------------------------------
# 4. Final Model Training
# ---------------------------------------------------------------------------
def train_final_model(ts: pd.Series, order: tuple) -> dict:
    """Train final ARIMA model on full dataset."""
    print(f"\n[4/6] Training final ARIMA{order} on full dataset...")
    
    model = ARIMA(ts, order=order)
    fitted = model.fit()
    
    print(f"  AIC: {fitted.aic:.2f}")
    print(f"  BIC: {fitted.bic:.2f}")
    print(f"  Log-Likelihood: {fitted.llf:.2f}")
    
    # In-sample metrics
    pred_insample = fitted.fittedvalues
    mae = mean_absolute_error(ts.iloc[1:], pred_insample.iloc[1:])
    rmse = np.sqrt(mean_squared_error(ts.iloc[1:], pred_insample.iloc[1:]))
    mape = mean_absolute_percentage_error(ts.iloc[1:], pred_insample.iloc[1:]) * 100
    
    print(f"  In-sample MAE: {mae:.4f}")
    print(f"  In-sample RMSE: {rmse:.4f}")
    print(f"  In-sample MAPE: {mape:.2f}%")
    
    return {
        'model': fitted,
        'order': order,
        'aic': fitted.aic,
        'bic': fitted.bic,
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }


# ---------------------------------------------------------------------------
# 5. Forecast
# ---------------------------------------------------------------------------
def generate_forecast(fitted_model, steps: int = 3) -> dict:
    """Generate multi-step forecast."""
    print(f"\n[5/6] Generating {steps}-step forecast...")
    
    forecast = fitted_model.get_forecast(steps=steps)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int(alpha=0.05)
    
    print(f"  Forecast results:")
    for i in range(steps):
        mean = forecast_mean.iloc[i]
        lower = forecast_ci.iloc[i, 0]
        upper = forecast_ci.iloc[i, 1]
        print(f"  Step {i+1}: {mean:.4f}% (95% CI: [{lower:.4f}%, {upper:.4f}%])")
    
    return {
        'forecast_mean': forecast_mean.tolist(),
        'forecast_lower': forecast_ci.iloc[:, 0].tolist(),
        'forecast_upper': forecast_ci.iloc[:, 1].tolist(),
        'steps': steps
    }


# ---------------------------------------------------------------------------
# 6. Plot
# ---------------------------------------------------------------------------
def create_plots(ts, fitted_model, forecast_result, order, test=None):
    """Create diagnostic and forecast plots."""
    print(f"\n[6/6] Creating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'ARIMA{order} — Inflasi MoM Forecast', fontsize=16, fontweight='bold')
    
    # Plot 1: Actual vs Fitted
    ax1 = axes[0, 0]
    ax1.plot(ts.index, ts.values, color='#0EA5E9', linewidth=1.5, label='Actual')
    ax1.plot(ts.index, fitted_model.fittedvalues.values, color='#EF4444', 
             linewidth=1.5, linestyle='--', label='Fitted')
    ax1.set_title('Actual vs Fitted Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Forecast
    ax2 = axes[0, 1]
    last_date = ts.index[-1]
    forecast_dates = pd.date_range(start=last_date, periods=len(forecast_result['forecast_mean'])+1, freq='MS')[1:]
    
    ax2.plot(ts.index[-24:], ts.values[-24:], color='#0EA5E9', linewidth=1.5, label='Historis')
    ax2.plot(forecast_dates, forecast_result['forecast_mean'], color='#10B981', 
             linewidth=2, marker='o', label='Forecast')
    ax2.fill_between(forecast_dates, 
                     forecast_result['forecast_lower'],
                     forecast_result['forecast_upper'],
                     color='#10B981', alpha=0.15, label='95% CI')
    ax2.set_title('Forecast')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Residuals
    ax3 = axes[1, 0]
    residuals = ts - fitted_model.fittedvalues
    ax3.plot(ts.index, residuals.values, color='#F59E0B', linewidth=1)
    ax3.axhline(y=0, color='#EF4444', linestyle='--', alpha=0.5)
    ax3.set_title('Residuals')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Residual Distribution
    ax4 = axes[1, 1]
    ax4.hist(residuals.dropna().values, bins=30, color='#0EA5E9', 
             edgecolor='white', alpha=0.7)
    ax4.set_title('Residual Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'arima_diagnostics.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plots saved to: {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Load data
    ts = load_data()
    
    # 2. Check stationarity
    d = check_stationarity(ts)
    
    # 3. Grid search
    grid_result = grid_search_arima(ts, d=d)
    
    # 4. Train final model
    final = train_final_model(ts, grid_result['best_order'])
    
    # 5. Forecast
    forecast = generate_forecast(final['model'], steps=3)
    
    # 6. Plots
    create_plots(ts, final['model'], forecast, grid_result['best_order'],
                 test=grid_result['test'])
    
    # Save model
    print("\n" + "=" * 60)
    print("SAVING MODEL & METRICS")
    print("=" * 60)
    
    # Save ARIMA model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(final['model'], f)
    print(f"  ✓ Model saved: {MODEL_PATH}")
    
    # Save metrics
    metrics = {
        'order': grid_result['best_order'],
        'aic': final['aic'],
        'bic': final['bic'],
        'mae': final['mae'],
        'rmse': final['rmse'],
        'mape': final['mape'],
        'grid_search_top5': grid_result['all_results'][:5]
    }
    with open(METRICS_PATH, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"  ✓ Metrics saved: {METRICS_PATH}")
    
    # Save forecast
    forecast_data = {
        'forecast': forecast,
        'order': grid_result['best_order'],
        'last_date': ts.index[-1].strftime('%Y-%m-%d'),
        'last_value': float(ts.iloc[-1])
    }
    with open(FORECAST_PATH, 'wb') as f:
        pickle.dump(forecast_data, f)
    print(f"  ✓ Forecast saved: {FORECAST_PATH}")
    
    print("\n" + "=" * 60)
    print("SELESAI!")
    print("=" * 60)
    print(f"  Best ARIMA: {grid_result['best_order']}")
    print(f"  AIC: {final['aic']:.2f}")
    print(f"  MAE: {final['mae']:.4f}")
    print(f"  RMSE: {final['rmse']:.4f}")
    print(f"  MAPE: {final['mape']:.2f}%")
    print(f"\n  Forecast:")
    for i, val in enumerate(forecast['forecast_mean']):
        print(f"    Step {i+1}: {val:.4f}%")
