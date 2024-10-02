import pandas as pd
from StockPredictionApp.src.rl_env import StockTradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import sqlite3

# Function to load stock data from SQLite
def load_all_stock_data(stock_symbols, db_path='stocks.db'):
    """
    Load stock data from multiple stock symbols and combine into one DataFrame.
    """
    conn = sqlite3.connect(db_path)
    combined_df = pd.DataFrame()  # Initialize an empty DataFrame

    for symbol in stock_symbols:
        query = f"SELECT * FROM {symbol.upper()}"
        df = pd.read_sql_query(query, conn)
        df['stock'] = symbol  # Add a column to indicate the stock symbol
        combined_df = pd.concat([combined_df, df], ignore_index=True)  # Append each stock's data

    conn.close()
    return combined_df

# List of all stock symbols
stock_symbols = ['RELIANCE', 'MOTHERSON', 'RELIGARE', 'TCS', 'WIPRO']  # Add all your stock symbols here

# Load combined data for all stocks
combined_df = load_all_stock_data(stock_symbols)

# Split data into training and backtest sets (80% for training, 20% for backtesting)
train_size = int(len(combined_df) * 0.8)
train_df = combined_df[:train_size]
backtest_df = combined_df[train_size:]

# Initialize the custom environment with the training data
train_env = DummyVecEnv([lambda: StockTradingEnv(train_df)])

# Initialize PPO model
model = PPO("MlpPolicy", train_env, verbose=1)

# Train the model on the combined dataset
model.learn(total_timesteps=50000)  # Increase timesteps for larger dataset

# Save the trained model
model.save("ppo_combined_stocks_trading")

print("Trained model on combined stock data saved.")

# Load the trained model for backtesting
model = PPO.load("ppo_combined_stocks_trading")

# Initialize the custom environment with the backtesting data
backtest_env = DummyVecEnv([lambda: StockTradingEnv(backtest_df)])

# Reset the environment for evaluation
obs = backtest_env.reset()

# During evaluation, store actions and portfolio values for visualization
actions = []
portfolio_values = []

# Backtest for 10 days
for i in range(10):
    action, _states = model.predict(obs)
    obs, rewards, done, info = backtest_env.step(action)
    
    # Track agent's actions (buy = 1, sell = 2, hold = 0)
    actions.append(action[0])  # action is a list with one element
    
    # Calculate portfolio value (balance + held shares * current stock price)
    portfolio_value = backtest_env.envs[0].balance + backtest_env.envs[0].shares_held * backtest_df.iloc[backtest_env.envs[0].current_step]['close']
    portfolio_values.append(portfolio_value)
    
    # Render the environment at each step
    backtest_env.render()

    # If the episode is finished, break the loop
    if done:
        print(f"Final Profit: {backtest_env.envs[0].total_profit}")
        break

# Visualization function to display stock prices, agent's actions, and portfolio value
def visualize_results(df, actions, portfolio_values):
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot stock prices
    ax1.plot(df['close'], label='Stock Price', color='blue')
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Stock Price', color='blue')

    # Plot buy/sell actions
    buy_actions = [i for i, action in enumerate(actions) if action == 1]
    sell_actions = [i for i, action in enumerate(actions) if action == 2]
    ax1.scatter(buy_actions, df['close'].iloc[buy_actions], color='green', marker='^', label='Buy', alpha=1)
    ax1.scatter(sell_actions, df['close'].iloc[sell_actions], color='red', marker='v', label='Sell', alpha=1)

    # Plot portfolio value on a secondary axis
    ax2 = ax1.twinx()
    ax2.plot(portfolio_values, label='Portfolio Value', color='purple', linestyle='--')
    ax2.set_ylabel('Portfolio Value', color='purple')

    # Display legends
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.85))
    plt.show()

# Use actions and portfolio_values for visualization
visualize_results(backtest_df, actions, portfolio_values)
