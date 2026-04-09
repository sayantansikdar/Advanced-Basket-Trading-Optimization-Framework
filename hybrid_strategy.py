"""
Hybrid strategy combining CVFS-CMA-ES and Bayesian Optimization
"""
import numpy as np
from src.data_utils import fetch_data
from src.strategy import TradingStrategy

# Load data
prices = fetch_data(['AAPL', 'MSFT', 'GOOGL'], '2023-01-01', '2023-12-31')

# Weights from both optimizers
cvfs_weights = np.array([-0.08376943, -1.44482409, 0.76662195])
bo_weights = np.array([2.25867554, 5.0, -5.0])

# Test individual strategies
print("Individual Strategy Performance:")
print("="*60)

cvfs_strategy = TradingStrategy(prices, cvfs_weights)
cvfs_returns = cvfs_strategy.backtest()
cvfs_metrics = cvfs_strategy.get_metrics(cvfs_returns)

bo_strategy = TradingStrategy(prices, bo_weights)
bo_returns = bo_strategy.backtest()
bo_metrics = bo_strategy.get_metrics(bo_returns)

print(f"\nCVFS-CMA-ES:")
print(f"  Sharpe: {cvfs_metrics['Sharpe Ratio']:.3f}")
print(f"  Return: {cvfs_metrics['Total Return']:.2%}")
print(f"  DD: {cvfs_metrics['Max Drawdown']:.2%}")

print(f"\nBayesian:")
print(f"  Sharpe: {bo_metrics['Sharpe Ratio']:.3f}")
print(f"  Return: {bo_metrics['Total Return']:.2%}")
print(f"  DD: {bo_metrics['Max Drawdown']:.2%}")

# Test hybrid (equal weight)
print("\n" + "="*60)
print("Hybrid Strategy (50% CVFS + 50% BO):")
print("="*60)

hybrid_returns = 0.5 * cvfs_returns + 0.5 * bo_returns
sharpe = np.sqrt(252) * hybrid_returns.mean() / (hybrid_returns.std() + 1e-9)
total_return = (1 + hybrid_returns).prod() - 1

print(f"  Sharpe Ratio: {sharpe:.3f}")
print(f"  Total Return: {total_return:.2%}")
