"""
Test different entry/exit thresholds
"""
import numpy as np
from src.data_utils import fetch_data
from src.strategy import TradingStrategy

prices = fetch_data(['AAPL', 'MSFT', 'GOOGL'], '2023-01-01', '2024-12-31')
train_size = int(len(prices) * 0.6)
train_data = prices.iloc[:train_size]
test_data = prices.iloc[train_size:]

weights = np.array([2.25867554, 5.0, -5.0])

print("Testing different thresholds on test data:")
print("="*60)

thresholds = [(1.0, 0.3), (1.5, 0.4), (2.0, 0.5), (2.5, 0.6)]

for entry, exit_t in thresholds:
    strategy = TradingStrategy(test_data, weights, entry_threshold=entry, exit_threshold=exit_t)
    returns = strategy.backtest()
    metrics = strategy.get_metrics(returns)
    
    print(f"\nEntry={entry}, Exit={exit_t}:")
    print(f"  Sharpe: {metrics['Sharpe Ratio']:.3f}")
    print(f"  Return: {metrics['Total Return']:.2%}")
    print(f"  Trades: {(returns != 0).sum()}")
