# preprocess.py

import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# preprocess.py
def load_data(stock_symbol):
    """
    Load stock data from SQLite database based on the stock symbol.
    Tables are named as the stock symbols in uppercase.
    """
    if not stock_symbol:
        raise ValueError("Stock symbol is empty or None")

    conn = sqlite3.connect('stocks.db')
    query = f"SELECT * FROM {stock_symbol.upper()}"  # Ensure symbol is not empty
    try:
        data = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error executing query for symbol {stock_symbol}: {e}")
        raise e
    conn.close()
    return data


def preprocess_data(stock_symbol, sequence_length=60):
    """
    Preprocesses data for LSTM and XGBoost models.
    Scales the 'Close' price for LSTM and prepares sequence data.
    Adds time-related features for XGBoost.
    """
    data = load_data(stock_symbol)
    
    # Convert 'timestamp' column to datetime format
    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
    
    # Drop rows where 'timestamp' conversion failed (if any)
    data = data.dropna(subset=['timestamp'])
    
    # Extracting time-related features
    data['day_of_week'] = data['timestamp'].dt.weekday
    data['hour'] = data['timestamp'].dt.hour
    data['day_of_month'] = data['timestamp'].dt.day
    data['month'] = data['timestamp'].dt.month
    data['quarter'] = data['timestamp'].dt.quarter
    
    # Use 'Close' price for LSTM and feature engineering for XGBoost
    closing_prices = data[['close']].values
    
    # Scaling data for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(closing_prices)

    # Preparing sequences for LSTM
    X_seq, y_seq = [], []
    for i in range(sequence_length, len(scaled_data)):
        X_seq.append(scaled_data[i-sequence_length:i, 0])
        y_seq.append(scaled_data[i, 0])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    X_seq = np.reshape(X_seq, (X_seq.shape[0], X_seq.shape[1], 1))  # Reshape for LSTM
    
    # For XGBoost, add time-related features along with other stock data
    stock_data = data.drop(columns=['timestamp'])  # Keep useful columns for XGBoost

    return X_seq, y_seq, stock_data, scaler

