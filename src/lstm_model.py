# lstm_model.py

import torch
import torch.nn as nn
from torch.optim import Adam

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 50).cuda()  # Initialize hidden state
        c0 = torch.zeros(2, x.size(0), 50).cuda()  # Initialize cell state
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Use the last output from LSTM
        return out

def train_lstm(X_train, y_train, epochs=50, lr=0.001):
    """
    Trains an LSTM model on the provided data.
    """
    model = LSTMModel(input_size=1, hidden_size=50, output_size=1, num_layers=2).cuda()
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        
        # Converting to tensors
        X_train_torch = torch.from_numpy(X_train).float().cuda()
        y_train_torch = torch.from_numpy(y_train).float().cuda()

        optimizer.zero_grad()
        outputs = model(X_train_torch)
        loss = criterion(outputs.squeeze(), y_train_torch)
        
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return model

def predict_lstm(model, X_test):
    """
    Predicts the output using the trained LSTM model.
    """
    model.eval()
    with torch.no_grad():
        X_test_torch = torch.from_numpy(X_test).float().cuda()
        predictions = model(X_test_torch)
    return predictions.cpu().numpy()  # Move back to CPU
