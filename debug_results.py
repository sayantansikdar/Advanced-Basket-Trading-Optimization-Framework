"""
Debug why results changed dramatically
"""
import numpy as np
import pandas as pd
from src.data_utils import fetch_data
from src.strategy import TradingStrategy

# Load data
prices = fetch_data(['AAPL', 'MSFT', 'GOOGL'], '2023-01-01', '2023-12-31')
print(f"Total data points: {len(prices)}")
print(f"Date range: {prices.index[0]} to {prices.index[-1]}")

# Test the known good weights from previous run
good_weights = np.array([2.25867554, 5.0, -5.0])

# Test on different splits
split_ratios = [0.6, 0.7, 0.8]
for ratio in split_ratios:
    train_size = int(len(prices) * ratio)
    train_data = prices.iloc[:train_size]
    test_data = prices.iloc[train_size:]
    
    strategy = TradingStrategy(test_data, good_weights)
    returns = strategy.backtest()
    metrics = strategy.get_metrics(returns)
    
    print(f"\nSplit ratio {ratio}:")
    print(f"  Train: {len(train_data)} days, Test: {len(test_data)} days")
    print(f"  Sharpe Ratio: {metrics['Sharpe Ratio']:.3f}")
    print(f"  Total Return: {metrics['Total Return']:.2%}")
