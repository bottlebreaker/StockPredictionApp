# preprocess.py

import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(stock_symbol):
    """
    Load stock data from SQLite database based on the stock symbol.
    Tables are named as the stock symbols in uppercase.
    """
    conn = sqlite3.connect('stocks.db')
    query = f"SELECT * FROM {stock_symbol.upper()}"
    data = pd.read_sql_query(query, conn)
    conn.close()
    return data

def preprocess_data(stock_symbol, sequence_length=60):
    """
    Preprocesses data for LSTM and XGBoost models.
    Scales the 'Close' price for LSTM and prepares sequence data.
    """
    data = load_data(stock_symbol)
    
    # Use 'Close' price for LSTM and feature engineering for XGBoost
    closing_prices = data[['Close']].values
    
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

    return X_seq, y_seq, data, scaler
