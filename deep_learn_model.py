import torch
from data import prepare_data_for_ai_model
from data_colecting import download_parts_data
from datetime import datetime
import torch.nn as nn
import os


first_train = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X, y_sequences, y_target, scaler, features = prepare_data_for_ai_model(download_parts_data(
    ticker='AAPL',
    start_date=datetime(2000, 1, 1),
    end_date=datetime(2009, 12, 30)
))

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y_target, dtype=torch.float32).unsqueeze(1)

# Dataset i DataLoader
dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

class LSTMModel(nn.Module):

    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)


    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # ostatnia wartość sekwencji
        out = self.fc(out)
        return out

model_path = "lstm_model.pth"
model = LSTMModel(input_size=X.shape[2]).to(device)

if first_train == True:
    # Trenowanie
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 200

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.6f}")

    # Zapis modelu
    torch.save(model.state_dict(), model_path)
    print("Model został zapisany.")

    # Predykcje
    model.eval()
    with torch.no_grad():
        X_tensor = X_tensor.to(device)
        predictions = model(X_tensor).cpu().numpy()

else:
    model = LSTMModel(input_size=X.shape[2])  # X.shape[2] = liczba cech
    model.load_state_dict(torch.load("lstm_model.pth", weights_only=True))
    model.to(device)
    model.train()  # ustawiamy tryb treningowy

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    epochs_fine = 200  # liczba epok do fine-tuningu

    for epoch in range(epochs_fine):
        epoch_loss = 0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Fine-tuning Epoch {epoch + 1}/{epochs_fine}, Loss: {epoch_loss / len(dataloader):.6f}")

    torch.save(model.state_dict(), "lstm_model_finetuned.pth")
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor.to(device)).cpu().numpy()


