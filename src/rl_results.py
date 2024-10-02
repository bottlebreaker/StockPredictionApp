import matplotlib.pyplot as plt

def visualize_results(df, actions, portfolio_values):
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot stock prices
    ax1.plot(df['Close'], label='Stock Price', color='blue')
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Stock Price', color='blue')

    # Plot actions (buy/sell)
    buy_actions = [i for i, action in enumerate(actions) if action == 1]
    sell_actions = [i for i, action in enumerate(actions) if action == 2]
    ax1.scatter(buy_actions, df['Close'].iloc[buy_actions], color='green', marker='^', label='Buy', alpha=1)
    ax1.scatter(sell_actions, df['Close'].iloc[sell_actions], color='red', marker='v', label='Sell', alpha=1)

    # Plot portfolio value on a secondary axis
    ax2 = ax1.twinx()
    ax2.plot(portfolio_values, label='Portfolio Value', color='purple', linestyle='--')
    ax2.set_ylabel('Portfolio Value', color='purple')

    # Display legends
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.85))
    plt.show()

# Example usage after training:
# df contains stock prices
# actions is the list of actions taken by the agent (1: Buy, 2: Sell, 0: Hold)
# portfolio_values contains portfolio value over time


