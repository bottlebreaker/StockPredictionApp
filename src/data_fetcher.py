import requests
import hashlib
import os
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import talib
import pyodbc
from datetime import datetime, timedelta
import sqlite3
import re


API_KEY = '1jz9tqg56utzy853'
API_SECRET = '8s32h28kdapn2st9z3a1p3bmg11eljie'
ACCESS_TOKEN_FILE = 'access_token.json'  # File to save the access token
DB_PATH = r'D:\StockPrediction\db\stocks.mdb'  # Path to your MS Access database
conn_str = f'DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={DB_PATH};'

# Function to calculate the checksum
def calculate_checksum(api_key, request_token, api_secret):
    checksum_str = f"{api_key}{request_token}{api_secret}"
    return hashlib.sha256(checksum_str.encode('utf-8')).hexdigest()

# Save the access token to a file
def save_access_token(token_data):
    with open(ACCESS_TOKEN_FILE, 'w') as f:
        json.dump(token_data, f)

# Load the access token from a file
def load_access_token():
    if os.path.exists(ACCESS_TOKEN_FILE):
        with open(ACCESS_TOKEN_FILE, 'r') as f:
            token_data = json.load(f)
            return token_data.get('access_token')
    return None

def authenticate_and_get_access_token():
    access_token = load_access_token()
    
    # If we already have an access token, return it
    if access_token:
        print(f"Access Token Loaded: {access_token}")
        return access_token

    # If no access token, authenticate the user
    login_url = f"https://kite.trade/connect/login?api_key={API_KEY}"
    print(f"Login at this URL to get the request token: {login_url}")
    request_token = input("Enter the request token from URL: ")
    checksum = calculate_checksum(API_KEY, request_token, API_SECRET)

    response = requests.post('https://api.kite.trade/session/token', data={
        'api_key': API_KEY,
        'request_token': request_token,
        'checksum': checksum
    })

    if response.status_code == 200:
        data = response.json()
        access_token = data['data']['access_token']
        save_access_token({'access_token': access_token})
        print(f"Access Token: {access_token}")
        return access_token
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

# This function handles API requests and re-authenticates if necessary
def make_authenticated_request(url, params=None):
    access_token = load_access_token()

    if not access_token:
        # Get a new token if none exists
        access_token = authenticate_and_get_access_token()
    
    headers = {'Authorization': f'token {API_KEY}:{access_token}'}
    
    response = requests.get(url, headers=headers, params=params)

    # If we get a 403 error, re-authenticate
    if response.status_code == 403:
        print("Access token is invalid or expired. Re-authenticating...")
        access_token = authenticate_and_get_access_token()
        headers = {'Authorization': f'token {API_KEY}:{access_token}'}
        response = requests.get(url, headers=headers, params=params)
    
    return response

def fetch_and_store_instruments():
    # Make an authenticated request to fetch instruments
    url = 'https://api.kite.trade/instruments'
    response = make_authenticated_request(url)  # This function handles token validity and re-authentication

    if response.status_code == 200:
        data = response.text.splitlines()

        # Connect to the SQLite database
        conn = sqlite3.connect('stocks.db')  # Using SQLite database
        cursor = conn.cursor()

        # Create the Stocks table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Stocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE,
                name TEXT,
                stock_token TEXT,
                exchange TEXT
            )
        ''')

        # Process instrument data and filter out futures and options
        for line in data[1:]:  # Skip the header row
            values = line.split(',')
            stock_token = values[0]
            stock_symbol = values[1]
            stock_name = values[2]
            exchange = values[11]  # Assuming column 11 corresponds to the exchange (NSE or BSE)
            instrument_type = values[9]  # Assuming this is the instrument type (equity, FUT, OPT)

            # We only want to store equities (and differentiate NSE/BSE)
            if instrument_type == 'EQ' and exchange in ['NSE', 'BSE']:
                # Check if the stock symbol already exists for the specific exchange
                cursor.execute('SELECT 1 FROM Stocks WHERE symbol = ? AND exchange = ?', (stock_symbol, exchange))
                stock_exists = cursor.fetchone()

                if stock_exists is None:
                    # Insert into the SQLite database if the symbol does not exist
                    cursor.execute('''
                        INSERT INTO Stocks (symbol, name, stock_token, exchange)
                        VALUES (?, ?, ?, ?)
                    ''', (stock_symbol, stock_name, stock_token, exchange))
                else:
                    print(f"Symbol {stock_symbol} on {exchange} already exists, skipping insert.")

        conn.commit()
        conn.close()
        print("Equity stock tokens stored successfully in SQLite.")
    else:
        print(f"Error fetching instruments: {response.status_code}, {response.text}")

def search_stock_in_db(symbol, exchange):
    # Connect to the SQLite database
    conn = sqlite3.connect('stocks.db')
    cursor = conn.cursor()
    
    # Check if the table exists (if not, create it)
    cursor.execute('''CREATE TABLE IF NOT EXISTS Stocks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT,
                        name TEXT,
                        stock_token INTEGER,
                        exchange TEXT
                      )''')
    
    # Fetch the stock token for the given symbol and exchange (using COLLATE NOCASE)
    cursor.execute("SELECT stock_token FROM Stocks WHERE name = ? AND exchange = ? COLLATE NOCASE", (symbol, exchange))
    row = cursor.fetchone()
    
    conn.close()
    
    # Return the stock token or None if not found
    if row:
        return row[0]
    else:
        print(f"No stock token found for symbol: {symbol} on {exchange}")
        return None

# Get date range (from_date = 2000 days before today, to_date = today)
def get_date_range():
    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=2000)).strftime('%Y-%m-%d')
    return from_date, to_date

## Modified store_stock_data function to create dynamic table names
def store_stock_data(stock_symbol, stock_data):
    # Sanitize the stock symbol to ensure it is a valid SQL table name
    print("Entering function store_stock_data")
    table_name = sanitize_symbol(stock_symbol)
    print(f"Sanitized table name for {stock_symbol}: {table_name}")

    # Check if any data exists
    if not stock_data:
        print(f"No data to store for {stock_symbol}")
        return

    try:
        # Connect to the SQLite database
        conn = sqlite3.connect('stocks.db')
        cursor = conn.cursor()
        print("Connected to the database.")

        # Dynamically create a table based on the stock symbol (if it doesn't already exist)
        create_table_query = f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                timestamp TEXT PRIMARY KEY,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL
            )
        '''
        #print(f"Creating table {table_name} with query: {create_table_query}")
        cursor.execute(create_table_query)
        print(f"Table {table_name} created (or already exists).")

        # Insert stock data into the dynamically named table
        for entry in stock_data:
            #print(f"Inserting data: {entry}")
            insert_query = f'''
                INSERT OR IGNORE INTO {table_name} (timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?)
            '''
            cursor.execute(insert_query, (entry[0], entry[1], entry[2], entry[3], entry[4], entry[5]))

        # Commit the transaction
        conn.commit()
        print(f"Data for {stock_symbol} committed successfully.")

    except sqlite3.Error as e:
        print(f"Error while working with SQLite: {e}")
    finally:
        conn.close()
        print("Database connection closed.")

# Fetch historical stock data from Zerodha
def fetch_historical_data(stock_symbol, stock_token, from_date, to_date, interval='day'):
    print(f"Fetching data for stock: {stock_symbol}, token: {stock_token}")
    access_token = load_access_token()
    if not access_token:
        print("Error: No access token found.")
        return

    url = f'https://api.kite.trade/instruments/historical/{stock_token}/{interval}'
    headers = {
        'Authorization': f'token {API_KEY}:{access_token}'
    }
    params = {
        'from': from_date,
        'to': to_date,
        'interval': interval
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        print(f"Successfully fetched historical data for stock token {stock_token}")
        candles = data['data']['candles']  # OHLCV data
        
        # Store the fetched data in the database using the stock symbol as the table name
        store_stock_data(stock_symbol, candles)  # Use stock symbol as the table name
        
        return candles
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

# Convert historical data into a pandas DataFrame
def convert_to_dataframe(historical_data):
    df = pd.DataFrame(historical_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# Normalize the data
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

# Sanitize the stock symbol to create valid table names
def sanitize_symbol(symbol):
    return re.sub(r'\W|^(?=\d)', '_', symbol)  # Replace non-alphanumeric chars and avoid table name starting with a number



