import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def create_lstm_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i + seq_length)].values
        y = data.iloc[i + seq_length]['Inflasi_MoM']
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def main():
    data_path = os.path.join('datasets', 'processed', 'clean_inflasi_ts.csv')
    df = pd.read_csv(data_path)
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df = df.sort_values('Tanggal').reset_index(drop=True)
    df.set_index('Tanggal', inplace=True)

    # Imputasi
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
    df_lstm = df[features].copy()

    scaler = MinMaxScaler()
    df_lstm_scaled = pd.DataFrame(scaler.fit_transform(df_lstm), columns=features, index=df_lstm.index)

    lag_steps = 12
    X_seq, y_seq = create_lstm_sequences(df_lstm_scaled, lag_steps)
    
    # Train on all data for deployment
    X_train_lstm = torch.tensor(X_seq, dtype=torch.float32)
    y_train_lstm = torch.tensor(y_seq, dtype=torch.float32).view(-1, 1)

    lstm_model = LSTMModel(input_size=4, hidden_size=64, num_layers=2, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

    print("Melatih model LSTM pada seluruh dataset...")
    epochs = 100
    for epoch in range(epochs):
        lstm_model.train()
        optimizer.zero_grad()
        outputs = lstm_model(X_train_lstm)
        loss = criterion(outputs, y_train_lstm)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    os.makedirs('models', exist_ok=True)
    
    # Save model state dict
    torch.save(lstm_model.state_dict(), 'models/lstm_model.pt')
    
    # Save scaler
    with open('models/lstm_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    print("Berhasil mengekspor model LSTM (lstm_model.pt) dan scaler (lstm_scaler.pkl)!")

if __name__ == '__main__':
    main()
