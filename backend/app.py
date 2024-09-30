import sys
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3

# Adding the parent directory to the path to access modules properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import functions from the respective modules
from . import data_fetcher, preprocess
from StockPredictionApp.src.data_fetcher import fetch_and_store_instruments, search_stock_in_db, get_date_range

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# API route to handle POST requests for stock data
@app.route('/api/stocks', methods=['POST'])
def get_stock_data():
    data = request.get_json()
    if not data or 'stocks' not in data:
        return jsonify({"error": "Invalid request, no stocks provided"}), 400

    stock_symbols = data.get('stocks')  # List of stock symbols
    stock_data_results = {}

    # First, ensure that stock tokens are fetched and stored in the database
    #search_stock_in_db(symbol, exchange="NSE")  # Ensure the instruments are stored before processing

    for symbol in stock_symbols:
        stock_token = search_stock_in_db(symbol, exchange='NSE')  # Fetch the stock token from SQLite (now properly integrated)

        if stock_token:
            from_date, to_date = get_date_range()  # Get the date range for fetching historical data
            historical_data = data_fetcher.fetch_historical_data(stock_token, from_date, to_date)  # Fetch historical stock data

            if historical_data:
                processed_data = process_and_return_stock_data(historical_data)  # Process the fetched data
                stock_data_results[symbol] = processed_data
            else:
                stock_data_results[symbol] = {"error": "No data available for this stock"}
        else:
            stock_data_results[symbol] = {"error": "Stock token not found"}

    return jsonify({"success": True, "data": stock_data_results})

# Health check route for GET requests
@app.route('/api/stocks', methods=['GET'])
def stock_data_health_check():
    return jsonify({"status": "API is working", "method": "GET"}), 200

# Process stock data (Normalize and add technical indicators)
def process_and_return_stock_data(historical_data):
    df = preprocess.convert_to_dataframe(historical_data)  # Convert to DataFrame
    df = preprocess.normalize_data(df)  # Normalize the data
    df = preprocess.add_technical_indicators(df)  # Add technical indicators like RSI, MACD, etc.
    return df.to_dict(orient='records')  # Convert DataFrame back to dictionary format for JSON response

if __name__ == '__main__':
    app.run(debug=True)
