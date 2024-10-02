# app.py
import pandas as pd
import sys
import os
from StockPredictionApp.backend import preprocess
from StockPredictionApp.src.preprocess import preprocess_data
from StockPredictionApp.backend.preprocess import process_and_return_stock_data
from StockPredictionApp.src.data_fetcher import fetch_historical_data, get_date_range, search_stock_in_db
from StockPredictionApp.src.rl_env import StockTradingEnv
from StockPredictionApp.src.rl_ppo_train import load_all_stock_data
from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

# Importing model-related modules
from StockPredictionApp.src.lstm_model import load_lstm_model, load_scaler, train_lstm, predict_lstm
from StockPredictionApp.src.xgboost_model import load_xgboost_model, train_xgboost, predict_xgboost
from StockPredictionApp.src.model_integration import predict_combined

# Adding the parent directory to the path to access modules properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables to store trained models
lstm_model = None
xgb_model = None
rl_model = None

# API route to handle POST requests for stock data (unchanged)
@app.route('/api/stocks', methods=['POST'])
def get_stock_data():
    data = request.get_json()
    if not data or 'stocks' not in data:
        return jsonify({"error": "Invalid request, no stocks provided"}), 400

    stock_symbols = data.get('stocks')  # List of stock symbols
    stock_data_results = {}

    for symbol in stock_symbols:
        stock_token = search_stock_in_db(symbol, exchange='NSE')  # Fetch the stock token from SQLite (now properly integrated)

        if stock_token:
            from_date, to_date = get_date_range()  # Get the date range for fetching historical data
            historical_data = fetch_historical_data(symbol, stock_token, from_date, to_date)  # Fetch historical stock data

            if historical_data:
                processed_data = preprocess.process_and_return_stock_data(historical_data)  # Process the fetched data
                stock_data_results[symbol] = processed_data
            else:
                stock_data_results[symbol] = {"error": "No data available for this stock"}
        else:
            stock_data_results[symbol] = {"error": "Stock token not found"}

    return jsonify({"success": True, "data": stock_data_results})

@app.route('/api/train', methods=['POST'])
def train_models():
    data = request.json
    if not data or 'symbol' not in data:
        return jsonify({"error": "Invalid request, no stock symbol provided"}), 400
    
    stock_symbol = data['symbol']

    # Preprocess data (returns stock_data, including 'close' column)
    X_lstm, y_lstm, stock_data, scaler = preprocess_data(stock_symbol)
    # Extract target variable 'close' for XGBoost model
    y_xgb = stock_data['close'].values  # Target is 'close' prices

    # Drop 'close' from features for XGBoost
    X_xgb = stock_data.drop(columns=['close'], errors='ignore').values

    # Train LSTM model
    global lstm_model
    lstm_model = train_lstm(X_lstm, y_lstm, scaler)

    # Train XGBoost model
    global xgb_model
    xgb_model = train_xgboost(X_xgb, y_xgb)

    # Store the scaler for future predictions
    global scaler_model
    scaler_model = scaler

    return jsonify({"message": "Models trained successfully!"})

# New API route for predictions
@app.route('/api/predict', methods=['POST'])
def predict_stock():
    data = request.json
    if not data or 'symbol' not in data:
        return jsonify({"error": "Invalid request, no stock symbol provided"}), 400
    
    stock_symbol = data['symbol']

    # Preprocess the data for LSTM and XGBoost
    X_lstm, _, stock_data, scaler_model = preprocess_data(stock_symbol)  # Ensure scaler_model is returned here
    X_xgb = stock_data.drop(columns=['close'], errors='ignore').values

    # Load the saved models if they are not already loaded
    global lstm_model, xgb_model
    if lstm_model is None:
        lstm_model = load_lstm_model()
    if xgb_model is None:
        xgb_model = load_xgboost_model()
    if scaler_model is None:
        scaler_model = load_scaler()

    # Ensure scaler_model is defined
    if 'scaler_model' not in globals():
        return jsonify({"error": "Scaler not found. Train the models first."}), 500

    # Perform predictions
    lstm_preds = predict_lstm(lstm_model, X_lstm, scaler_model)  # Pass the scaler_model for inverse transformation
    xgb_preds = predict_xgboost(xgb_model, X_xgb)

    # Combine LSTM and XGBoost predictions if necessary
    combined_preds = predict_combined(lstm_model, xgb_model, X_lstm, X_xgb)

    return jsonify({"predictions": combined_preds.tolist()})

@app.route('/api/train_rl_agent', methods=['POST'])
def train_rl_agent():
    data = request.json
    if not data or 'symbol' not in data:
        return jsonify({"error": "Invalid request, no stock symbol provided"}), 400
    
    stock_symbol = data.get('symbol')

    if not stock_symbol:
        return jsonify({"error": "Stock symbol is missing"}), 400

    # Preprocess data (you can reuse your existing preprocess_data function or fetch data for the RL model)
    try:
        X_lstm, _, stock_data, _ = preprocess_data(stock_symbol)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400  # Catch and return errors
    except Exception as e:
        return jsonify({"error": f"Failed to load stock data for {stock_symbol}: {e}"}), 500

    # Prepare stock data (assuming stock_data is a DataFrame with OHLCV columns)
    df = pd.DataFrame(stock_data)

    # Create a custom trading environment
    env = DummyVecEnv([lambda: StockTradingEnv(df)])

    # Initialize the PPO model for training
    global rl_model
    rl_model = PPO("MlpPolicy", env, verbose=1)

    # Train the RL model
    rl_model.learn(total_timesteps=100000)  # Adjust timesteps based on the complexity

    # Save the RL model
    rl_model.save("ppo_stock_trading")

    return jsonify({"message": "RL Agent trained successfully!"})

@app.route('/api/predict_rl', methods=['POST'])
def predict_rl():
    data = request.json
    if not data or 'symbol' not in data:
        return jsonify({"error": "Invalid request, no stock symbol provided"}), 400

    stock_symbol = data['symbol']

    # Preprocess data (you can reuse your existing preprocess_data function or fetch data for the RL model)
    X_lstm, _, stock_data, _ = preprocess_data(stock_symbol)

    # Prepare stock data (assuming stock_data is a DataFrame with OHLCV columns)
    df = pd.DataFrame(stock_data)

    # Load the trained RL model
    global rl_model
    if rl_model is None:
        try:
            rl_model = PPO.load("ppo_stock_trading")
        except:
            return jsonify({"error": "RL model not trained. Train the model first."}), 500

    # Create a new environment for evaluation
    env = DummyVecEnv([lambda: StockTradingEnv(df)])
    obs = env.reset()

    # Perform predictions over a fixed number of steps (e.g., 1000)
    actions = []
    portfolio_values = []

    for i in range(10):
        action, _states = rl_model.predict(obs)
        obs, rewards, done, info = env.step(action)

        # Track actions and portfolio values
        actions.append(int(action[0]))  # Ensure action is converted to native Python int
        portfolio_value = env.envs[0].balance + env.envs[0].shares_held * float(df.iloc[env.envs[0].current_step]['close'])
        portfolio_values.append(portfolio_value)

        if done:
            break

    # Ensure all values are native Python types
    portfolio_values = [float(value) for value in portfolio_values]  # Convert to native Python float

    # Return the actions and portfolio value
    return jsonify({
        "actions": actions,  # Actions are converted to native Python int
        "portfolio_value": float(portfolio_values[-1]),  # Final portfolio value as Python float
        "predictions": portfolio_values  # List of portfolio values as native Python floats
    })


@app.route('/api/backtest_rl_agent', methods=['POST'])
def backtest_rl_agent():
    data = request.get_json()
    if not data or 'symbol' not in data:
        return jsonify({"error": "Invalid request, no stock symbol provided"}), 400
    
    stock_symbol = data['symbol']

    # Load the stock data and split it into training and backtest datasets
    df = load_all_stock_data(stock_symbol)
    train_size = int(len(df) * 0.8)
    backtest_df = df[train_size:]

    # Set up the environment for backtesting
    backtest_env = DummyVecEnv([lambda: StockTradingEnv(backtest_df)])

    # Load the RL model
    model = PPO.load("ppo_stock_trading")

    # Perform backtesting
    obs = backtest_env.reset()
    backtest_actions = []
    backtest_portfolio_values = []

    for i in range(10):  # Backtest for 10 days
        action, _states = model.predict(obs)
        obs, rewards, done, info = backtest_env.step(action)
        
        current_portfolio_value = backtest_env.envs[0].balance + backtest_env.envs[0].shares_held * backtest_df.iloc[backtest_env.envs[0].current_step]['Close']
        backtest_actions.append(int(action[0]))
        backtest_portfolio_values.append(current_portfolio_value)

        if done:
            break

    # Return the backtest results
    return jsonify({
        "actions": backtest_actions,
        "portfolio_value": backtest_portfolio_values[-1],
        "total_return": ((backtest_portfolio_values[-1] - 10000) / 10000) * 100
    })


# Health check route (unchanged)
@app.route('/api/stocks', methods=['GET'])
def stock_data_health_check():
    return jsonify({"status": "API is working", "method": "GET"}), 200


if __name__ == '__main__':
    app.run(debug=True)
