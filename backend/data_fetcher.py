import requests
import json

API_KEY = '1jz9tqg56utzy853'
ACCESS_TOKEN_FILE = 'StockPredictionApp\\access_token.json'  # File where access token is saved


# Function to load access token from file
def load_access_token():
    with open(ACCESS_TOKEN_FILE, 'r') as f:
        token_data = json.load(f)
        return token_data.get('access_token')

# Function to fetch historical stock data from Zerodha API
def fetch_historical_data(stock_token, from_date, to_date, interval='day'):
    access_token = load_access_token()
    if not access_token:
        print("Error: No access token found.")
        return
    
    url = f'https://api.kite.trade/instruments/historical/{stock_token}/{interval}'
    headers = {
        'Authorization': f'token {API_KEY}:{access_token}'
    }
    params = {
        'from': from_date,  # Format: YYYY-MM-DD
        'to': to_date,      # Format: YYYY-MM-DD
        'interval': interval  # day, 5minute, 15minute, etc.
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        print(f"Successfully fetched historical data for stock token {stock_token}")
        return data['data']['candles']  # List of OHLCV data
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None
