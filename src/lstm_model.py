#lstm_model.py
import joblib  # For saving the scaler
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers  # Store num_layers as a class attribute
        self.hidden_size = hidden_size  # Store hidden_size as a class attribute
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)  # Batch size is the first dimension of input

        # Initialize hidden and cell states with the stored `num_layers` and `hidden_size`
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()

        # Forward pass through LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Use the last output from LSTM
        out = self.fc(out[:, -1, :])
        return out


def load_lstm_model():
    """
    Load the saved LSTM model from disk.
    """
    model_path = "models/lstm_model.pth"
    
    # Create a new model instance with input_size=1
    model = LSTMModel(input_size=1, hidden_size=50, output_size=1, num_layers=2)
    
    # Load the model weights
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cuda')), strict=False)
    model.cuda()  # Move the model to GPU if available
    model.eval()  # Set the model to evaluation mode
    print(f"Loaded saved LSTM model weights from {model_path}")
    
    return model

def train_lstm(X_train, y_train, scaler, epochs=50, lr=0.001):
    """
    Trains an LSTM model on the provided data and saves the model and scaler in 'models/' directory.
    """
    model = LSTMModel(input_size=1, hidden_size=50, output_size=1, num_layers=2).cuda()

    # Directory where models and scaler will be saved
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'lstm_model.pth')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')  # Save scaler as .pkl file

    # Load saved model weights if available
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print("Loaded saved model weights from models/lstm_model.pth")

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
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # Save the model's state and the scaler
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)  # Save the scaler object
    print(f'Model saved to {model_path} and scaler saved to {scaler_path}')
    
    return model

def load_scaler():
    scaler_path = "models/scaler.pkl"  # Path to where you saved the scaler
    try:
        scaler = joblib.load(scaler_path)
        return scaler
    except FileNotFoundError:
        raise ValueError("Scaler not found. Train the models first.")

def predict_lstm(model, X_test, scaler_model, days=10):
    """
    Predicts stock prices using the trained LSTM model for the next 'days' number of days.
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        input_seq = torch.from_numpy(X_test).float().cuda()  # Convert to tensor and move to GPU
        
        # Dynamically calculate the sequence length based on input size
        sequence_length = input_seq.size(0)  # Get the actual sequence length
        input_seq = input_seq.view(1, sequence_length, 1)  # Reshape based on actual size

        for _ in range(days):
            # Perform a forward pass and make a prediction for the last time step
            pred = model(input_seq)  # Output shape should be (1, sequence_length, 1)

            # Extract the last prediction from the model's output
            pred_value = pred[:, -1, :].squeeze().item()  # Get the scalar value
            
            # Append the predicted value
            predictions.append(pred_value)
            
            # Prepare input for the next prediction
            pred = torch.tensor(pred_value).view(1, 1, 1).cuda()  # Reshape to (1, 1, 1) for the next time step
            input_seq = torch.cat([input_seq[:, 1:, :], pred], dim=1)  # Remove oldest, append new prediction

    # Inverse transform the predictions back to the original scale
    predictions = scaler_model.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    return predictions







