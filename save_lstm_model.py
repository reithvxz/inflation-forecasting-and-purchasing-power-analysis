import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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

def create_lstm_sequences(X, y, seq_length):
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:i+seq_length])
        ys.append(y[i+seq_length])
    return np.array(xs), np.array(ys)

def main():
    data_path = os.path.join('datasets', 'processed', 'clean_inflasi_ts.csv')
    df = pd.read_csv(data_path)
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df = df.sort_values('Tanggal').reset_index(drop=True)
    df.set_index('Tanggal', inplace=True)

    # Imputasi (pastikan semua data bersih)
    df = df.ffill().bfill()

    # Feature Engineering (Mirip notebook)
    df['Bulan_Sin'] = np.sin(2 * np.pi * df['Bulan']/12)
    df['Bulan_Cos'] = np.cos(2 * np.pi * df['Bulan']/12)
    if 'Harga_Minyak_USD' in df.columns and 'USD_IDR' in df.columns:
        df['Oil_x_USDIDR'] = df['Harga_Minyak_USD'] * df['USD_IDR']

    # Kolom fitur (semua kecuali Bulan, Tahun asli)
    exclude_cols = ['Bulan', 'Tahun']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Pastikan Inflasi_MoM di posisi pertama (penting untuk inverse nanti)
    if 'Inflasi_MoM' in feature_cols:
        feature_cols.remove('Inflasi_MoM')
    feature_cols = ['Inflasi_MoM'] + feature_cols
    
    df_lstm = df[feature_cols].copy()
    n_features = len(feature_cols)

    # Pisahkan target untuk scaling terpisah (lebih stabil)
    X_all = df_lstm.drop('Inflasi_MoM', axis=1).values
    y_all = df_lstm['Inflasi_MoM'].values.reshape(-1, 1)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Split data untuk validasi (80% train, 20% val)
    train_size = int(len(X_all) * 0.8)
    X_train_raw, X_val_raw = X_all[:train_size], X_all[train_size:]
    y_train_raw, y_val_raw = y_all[:train_size], y_all[train_size:]

    # Fit scaler pada train saja
    scaler_X.fit(X_train_raw)
    scaler_y.fit(y_train_raw)

    # Transform semua data
    X_scaled = scaler_X.transform(X_all)
    y_scaled = scaler_y.transform(y_all)

    # Buat sequences
    lag_steps = 12
    X_seq, y_seq = create_lstm_sequences(X_scaled, y_scaled, lag_steps)

    # Adjust train/val indices karena sequence shifting
    val_idx_start = train_size - lag_steps
    
    X_train_seq = X_seq[:val_idx_start]
    y_train_seq = y_seq[:val_idx_start]
    
    X_val_seq = X_seq[val_idx_start:]
    y_val_seq = y_seq[val_idx_start:]

    # Convert to tensors
    X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_seq, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_seq, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_seq, dtype=torch.float32)

    # Model setup
    input_size = X_all.shape[1] # Number of features excluding target
    hidden_size = 128
    num_layers = 2
    model = LSTMModel(input_size, hidden_size, num_layers, 1)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    print(f"Training LSTM with {input_size} features...")

    # Training loop dengan Early Stopping
    epochs = 200
    patience = 30
    best_val_loss = float('inf')
    counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t)
            
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch+1) % 20 == 0:
            print(f'Epoch {epoch+1}: Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}')

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Simpan model dan scalers
    os.makedirs('models', exist_ok=True)
    
    # Simpan model (retrain pada semua data untuk deployment final)
    # Untuk deployment, kita retrain pada SEMUA data (train+val) untuk memaksimalkan info
    print("Retraining on full dataset for deployment...")
    X_full_t = torch.tensor(X_seq, dtype=torch.float32)
    y_full_t = torch.tensor(y_seq, dtype=torch.float32)
    
    final_model = LSTMModel(input_size, hidden_size, num_layers, 1)
    final_optimizer = optim.AdamW(final_model.parameters(), lr=0.001)
    
    # Retrain singkat (50 epoch saja karena sudah converge)
    for epoch in range(50):
        final_model.train()
        final_optimizer.zero_grad()
        outputs = final_model(X_full_t)
        loss = criterion(outputs, y_full_t)
        loss.backward()
        final_optimizer.step()

    # Simpan checkpoint
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'input_size': input_size,
        'seq_length': lag_steps,
        'feature_columns': feature_cols
    }, 'models/lstm_model.pt')
    
    # Simpan scalers
    with open('models/lstm_scaler_x.pkl', 'wb') as f:
        pickle.dump(scaler_X, f)
        
    with open('models/lstm_scaler_y.pkl', 'wb') as f:
        pickle.dump(scaler_y, f)
        
    print(f"Model saved with {input_size} features.")
    print(f"Scalers saved (lstm_scaler_x.pkl, lstm_scaler_y.pkl)")

if __name__ == '__main__':
    main()
