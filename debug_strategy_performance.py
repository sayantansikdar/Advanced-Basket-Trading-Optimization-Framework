"""
Debug why strategy is losing money
"""
import numpy as np
import pandas as pd
from src.data_utils import fetch_data
from src.strategy import TradingStrategy

# Load data
prices = fetch_data(['AAPL', 'MSFT', 'GOOGL'], '2023-01-01', '2024-12-31')

# Split data
train_size = int(len(prices) * 0.6)
train_data = prices.iloc[:train_size]
test_data = prices.iloc[train_size:]

# Use weights from successful run
weights = np.array([2.25867554, 5.0, -5.0])

print("="*60)
print("Testing Strategy Performance")
print("="*60)

# Test on training data
print("\nTraining Data Performance:")
strategy_train = TradingStrategy(train_data, weights)
returns_train = strategy_train.backtest()
metrics_train = strategy_train.get_metrics(returns_train)

print(f"  Sharpe Ratio: {metrics_train['Sharpe Ratio']:.3f}")
print(f"  Total Return: {metrics_train['Total Return']:.2%}")
print(f"  Mean Return: {returns_train.mean():.6f}")
print(f"  Std Return: {returns_train.std():.6f}")

# Test on test data
print("\nTest Data Performance:")
strategy_test = TradingStrategy(test_data, weights)
returns_test = strategy_test.backtest()
metrics_test = strategy_test.get_metrics(returns_test)

print(f"  Sharpe Ratio: {metrics_test['Sharpe Ratio']:.3f}")
print(f"  Total Return: {metrics_test['Total Return']:.2%}")
print(f"  Mean Return: {returns_test.mean():.6f}")
print(f"  Std Return: {returns_test.std():.6f}")

# Check spread stationarity
log_prices_train = np.log(train_data)
spread_train = np.dot(log_prices_train.values, weights)
log_prices_test = np.log(test_data)
spread_test = np.dot(log_prices_test.values, weights)

print(f"\nSpread Statistics:")
print(f"  Training Spread Mean: {spread_train.mean():.4f}")
print(f"  Training Spread Std: {spread_train.std():.4f}")
print(f"  Test Spread Mean: {spread_test.mean():.4f}")
print(f"  Test Spread Std: {spread_test.std():.4f}")
print(f"  Mean Shift: {abs(spread_test.mean() - spread_train.mean()):.4f}")
