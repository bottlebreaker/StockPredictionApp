import requests
import json

API_KEY = '1jz9tqg56utzy853'
ACCESS_TOKEN_FILE = 'StockPredictionApp\\access_token.json'  # File where access token is saved


# Function to load access token from file
def load_access_token():
    with open(ACCESS_TOKEN_FILE, 'r') as f:
        token_data = json.load(f)
        return token_data.get('access_token')

