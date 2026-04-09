"""
Detailed signal analysis
"""
import numpy as np
import pandas as pd
from src.data_utils import fetch_data
from src.strategy import TradingStrategy

# Load data
prices = fetch_data(['AAPL', 'MSFT', 'GOOGL'], '2023-01-01', '2024-12-31')

# Split
train_size = int(len(prices) * 0.6)
train_data = prices.iloc[:train_size]
test_data = prices.iloc[train_size:]

weights = np.array([2.25867554, 5.0, -5.0])

print("="*60)
print("Signal Analysis on Test Data")
print("="*60)

strategy = TradingStrategy(test_data, weights)
signals = strategy.generate_signals()
zscore = strategy._calculate_zscore()
returns = strategy.backtest()

print(f"\nSignal Distribution:")
print(f"  Long signals (1): {(signals == 1).sum()}")
print(f"  Short signals (-1): {(signals == -1).sum()}")
print(f"  Neutral (0): {(signals == 0).sum()}")

print(f"\nZ-score Statistics:")
print(f"  Mean: {zscore.mean():.4f}")
print(f"  Std: {zscore.std():.4f}")
print(f"  Max: {zscore.max():.4f}")
print(f"  Min: {zscore.min():.4f}")
print(f"  % > 2: {(abs(zscore) > 2).mean() * 100:.1f}%")

# Check if strategy is consistently losing
cumulative_returns = (1 + returns).cumprod()
print(f"\nCumulative Return: {cumulative_returns.iloc[-1] - 1:.2%}")

# Check position P&L by looking at spread changes when in position
position_changes = signals.diff().abs()
print(f"\nTrading Activity:")
print(f"  Number of trades: {position_changes.sum():.0f}")
print(f"  Avg days in position: {(signals != 0).sum() / (position_changes.sum() + 1):.1f}")
