#rl_env.py
import gym
from gym import spaces
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.current_step = 0
        self.balance = 10000  # Starting balance
        self.shares_held = 0
        self.total_shares_bought = 0
        self.total_shares_sold = 0
        self.total_profit = 0
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)

    def reset(self):
        self.balance = 10000
        self.shares_held = 0
        self.total_shares_bought = 0
        self.total_shares_sold = 0
        self.total_profit = 0
        self.current_step = 0
        return self._next_observation()

    def _next_observation(self):
        frame = self.df.iloc[self.current_step]
        obs = np.array([
            frame['open'],
            frame['high'],
            frame['low'],
            frame['close'],
            frame['volume'],
            self.shares_held
        ])
        return obs / obs.max()  # Normalize observation

    def _buy_shares(self, price):
        commission = 0.01 * price  # 1% commission fee
        if self.balance > price + commission:
            self.balance -= (price + commission)
            self.shares_held += 1
            self.total_shares_bought += 1

    def _sell_shares(self, price):
        commission = 0.01 * price  # 1% commission fee
        if self.shares_held > 0:
            self.balance += (price - commission)
            self.shares_held -= 1
            self.total_shares_sold += 1
            self.total_profit += (price - commission)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        
        # Action: 0 = hold, 1 = buy, 2 = sell
        if action == 1:  # Buy
            self._buy_shares(current_price)
        elif action == 2:  # Sell
            self._sell_shares(current_price)

        self.current_step += 1

        # Calculate profit/loss reward
        done = self.current_step >= len(self.df) - 1
        reward = (self.balance + self.shares_held * current_price) - 10000  # Reward is the change in portfolio value

        obs = self._next_observation()
        return obs, reward, done, {}

    def render(self, mode='human', close=False):
        profit = self.balance + self.shares_held * self.df.iloc[self.current_step]['close'] - 10000
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held}')
        print(f'Total profit: {self.total_profit}')

