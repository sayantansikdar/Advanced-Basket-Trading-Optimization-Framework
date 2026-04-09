"""
Debug backtesting logic
"""
import numpy as np
import pandas as pd
from src.data_utils import fetch_data
from src.strategy import TradingStrategy

# Load data
prices = fetch_data(['AAPL', 'MSFT', 'GOOGL'], '2023-01-01', '2023-12-31')
print(f"Data shape: {prices.shape}")
print(f"First few rows:\n{prices.head()}")

# Test a simple equal-weight strategy
weights = np.array([1/3, 1/3, 1/3])

print("\n" + "="*60)
print("Testing Simple Equal-Weight Strategy")
print("="*60)

strategy = TradingStrategy(prices, weights)
returns = strategy.backtest()

print(f"\nReturns statistics:")
print(f"  Mean daily return: {returns.mean():.6f}")
print(f"  Std daily return: {returns.std():.6f}")
print(f"  Total return: {(1 + returns).prod() - 1:.2%}")
print(f"  Sharpe ratio: {np.sqrt(252) * returns.mean() / (returns.std() + 1e-9):.3f}")

# Test a simple long-only strategy (all weights positive)
weights_long = np.array([1.0, 1.0, 1.0])
strategy_long = TradingStrategy(prices, weights_long)
returns_long = strategy_long.backtest()

print(f"\nLong-only strategy:")
print(f"  Total return: {(1 + returns_long).prod() - 1:.2%}")
print(f"  Sharpe ratio: {np.sqrt(252) * returns_long.mean() / (returns_long.std() + 1e-9):.3f}")

# Check the spread calculation
log_prices = np.log(prices)
spread = np.dot(log_prices.values, weights)
print(f"\nSpread statistics:")
print(f"  Mean: {spread.mean():.4f}")
print(f"  Std: {spread.std():.4f}")
print(f"  Is stationary? {np.abs(spread.mean()) < spread.std()}")
