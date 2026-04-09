"""
Debug signal generation
"""
import numpy as np
import pandas as pd
from src.data_utils import fetch_data
from src.strategy import TradingStrategy

# Load data
prices = fetch_data(['AAPL', 'MSFT', 'GOOGL'], '2023-01-01', '2023-12-31')

# Use the weights from your previous successful run
weights = np.array([2.25867554, 5.0, -5.0])

print("Testing with successful weights from previous run")
print("="*60)

strategy = TradingStrategy(prices, weights)
returns = strategy.backtest()
signals = strategy.generate_signals()
zscore = strategy._calculate_zscore()

print(f"\nReturns: {returns.mean():.6f} mean, {returns.std():.6f} std")
print(f"Sharpe: {np.sqrt(252) * returns.mean() / (returns.std() + 1e-9):.3f}")
print(f"Total return: {(1 + returns).prod() - 1:.2%}")

print(f"\nSignal distribution:")
print(f"  Long signals (1): {(signals == 1).sum()}")
print(f"  Short signals (-1): {(signals == -1).sum()}")
print(f"  Neutral (0): {(signals == 0).sum()}")

print(f"\nZ-score statistics:")
print(f"  Mean: {zscore.mean():.4f}")
print(f"  Std: {zscore.std():.4f}")
print(f"  Max: {zscore.max():.4f}")
print(f"  Min: {zscore.min():.4f}")

# Test with lower thresholds for more signals
print("\n" + "="*60)
print("Testing with lower thresholds (1.0 entry, 0.3 exit)")
print("="*60)

strategy2 = TradingStrategy(prices, weights, entry_threshold=1.0, exit_threshold=0.3)
returns2 = strategy2.backtest()
signals2 = strategy2.generate_signals()

print(f"Sharpe: {np.sqrt(252) * returns2.mean() / (returns2.std() + 1e-9):.3f}")
print(f"Total return: {(1 + returns2).prod() - 1:.2%}")
print(f"Signals - Long: {(signals2 == 1).sum()}, Short: {(signals2 == -1).sum()}")
