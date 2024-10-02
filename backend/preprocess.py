import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import talib

def process_and_return_stock_data(historical_data):
   
    # Convert raw historical stock data to DataFrame
    df = pd.DataFrame(historical_data)
    if list(df.columns) == [0, 1, 2, 3, 4, 5]:
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    # Convert column names to lowercase and remove spaces
    df.columns = df.columns.map(str).str.lower().str.strip()

    # Continue as before
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    if all(col in df.columns for col in required_columns):
        df = df[required_columns]
    else:
        raise KeyError(f"Some required columns are missing. Available columns: {df.columns}")

    # Proceed with data normalization and adding technical indicators
    df = add_technical_indicators(df)
    return df.to_dict(orient='records')

# Convert historical data into a pandas DataFrame
def convert_to_dataframe(historical_data):
    df = pd.DataFrame(historical_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # Convert UNIX timestamp to human-readable
    return df

# Normalize OHLC and volume data
def normalize_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[['open', 'high', 'low', 'close', 'volume']] = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume']])
    return df

# Add technical indicators (RSI, MACD, Moving Averages)
def add_technical_indicators(df):
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['SMA'] = talib.SMA(df['close'], timeperiod=20)
    df['EMA'] = talib.EMA(df['close'], timeperiod=20)
    return df
