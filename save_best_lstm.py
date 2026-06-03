"""
============================================================================
  SAVE BEST LSTM MODEL
  Proyek: Prediksi Inflasi dan Dampaknya terhadap Daya Beli
============================================================================
Script ini melatih model LSTM terbaik menggunakan data pipeline v2
dan menyimpannya ke file .pt (PyTorch) serta scaler ke .pkl.
============================================================================
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from data_pipeline import get_lstm_pipeline_data

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")


class LSTMModel(nn.Module):
    """LSTM untuk forecasting inflasi bulanan."""

    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load data dari pipeline v2
    print("Memuat data dari pipeline v2...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, df = \
        get_lstm_pipeline_data(seq_length=12)

    input_size = X_train.shape[2]  # 8 fitur

    # 2. Konversi ke tensor PyTorch
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # 3. DataLoader untuk training
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 4. Inisialisasi model
    model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # 5. Training loop dengan early stopping
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    max_patience = 30
    epochs = 200

    print(f"\nMelatih LSTM (input={input_size}, hidden=64, layers=2)...")
    print(f"  Epochs: {epochs}, Batch Size: 32, LR: 0.001")
    print("-" * 50)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()

        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if patience_counter >= max_patience:
            print(f"  Early stopping di epoch {epoch+1}")
            break

    # 6. Load best model dan evaluasi di test set
    model.load_state_dict(best_model_state)
    model.eval()

    with torch.no_grad():
        y_pred_scaled = model(X_test_t).numpy()

    # Denormalisasi: prediksi dan aktual
    dummy_pred = np.zeros((len(y_pred_scaled), input_size))
    dummy_pred[:, 0] = y_pred_scaled[:, 0]
    y_pred = scaler.inverse_transform(dummy_pred)[:, 0]

    dummy_actual = np.zeros((len(y_test), input_size))
    dummy_actual[:, 0] = y_test
    y_actual = scaler.inverse_transform(dummy_actual)[:, 0]

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))

    print("-" * 50)
    print(f"  Test MAE  : {mae:.4f}")
    print(f"  Test RMSE : {rmse:.4f}")

    # 7. Simpan model dan scaler
    model_path = os.path.join(MODELS_DIR, "best_lstm_inflasi.pt")
    scaler_path = os.path.join(MODELS_DIR, "lstm_scaler.pkl")

    torch.save({
        "model_state_dict": best_model_state,
        "input_size": input_size,
        "hidden_size": 64,
        "num_layers": 2,
        "seq_length": 12,
        "feature_cols": [
            "Inflasi_MoM", "BI_Rate", "USD_IDR",
            "Inflasi_Umum_MoM", "Inflasi_Inti_MoM",
            "Inflasi_HargaDiatur_MoM", "Inflasi_Bergejolak_MoM",
            "Harga_Minyak_USD"
        ],
        "test_mae": mae,
        "test_rmse": rmse,
    }, model_path)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n  ✓ Model disimpan  → {model_path}")
    print(f"  ✓ Scaler disimpan → {scaler_path}")

    # 8. Test loading
    checkpoint = torch.load(model_path, weights_only=False)
    loaded_model = LSTMModel(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"]
    )
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  ✓ Test load model: SUCCESS! (MAE: {checkpoint['test_mae']:.4f})")


if __name__ == "__main__":
    main()
