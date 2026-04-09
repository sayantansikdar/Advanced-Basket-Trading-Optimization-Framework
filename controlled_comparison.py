"""
Controlled comparison with fixed parameters
"""
import numpy as np
import pandas as pd
from src.data_utils import fetch_data
from src.strategy import TradingStrategy
from src.bayesian_opt import BasketOptimizer
from src.optimizers import CVFS_CMAESOptimizer

# Set random seed
np.random.seed(42)

# Load data
prices = fetch_data(['AAPL', 'MSFT', 'GOOGL'], '2023-01-01', '2023-12-31')

# Fixed split (use same as your working run)
train_size = 150  # Fixed training size
train_data = prices.iloc[:train_size]
test_data = prices.iloc[train_size:]

print(f"Train data: {len(train_data)} days")
print(f"Test data: {len(test_data)} days")

# Test known good weights
good_weights = np.array([2.25867554, 5.0, -5.0])
strategy = TradingStrategy(test_data, good_weights)
returns = strategy.backtest()
metrics = strategy.get_metrics(returns)

print(f"\nKnown good weights performance:")
print(f"  Sharpe Ratio: {metrics['Sharpe Ratio']:.3f}")
print(f"  Total Return: {metrics['Total Return']:.2%}")

# Now run optimization with fixed seed
config = {
    'n_trials': 30,
    'metric': 'Sharpe Ratio',
    'entry_threshold': 2.0,
    'exit_threshold': 0.5,
    'transaction_cost': 0.001
}

print("\n" + "="*60)
print("Running Bayesian Optimization (controlled)")
print("="*60)

bo_optimizer = BasketOptimizer(train_data, config)
bo_weights, _ = bo_optimizer.optimize(n_trials=30)

bo_strategy = TradingStrategy(test_data, bo_weights)
bo_returns = bo_strategy.backtest()
bo_metrics = bo_strategy.get_metrics(bo_returns)

print(f"\nBayesian Results:")
print(f"  Sharpe Ratio: {bo_metrics['Sharpe Ratio']:.3f}")
print(f"  Total Return: {bo_metrics['Total Return']:.2%}")
print(f"  Weights: {bo_weights}")

print("\n" + "="*60)
print("Running CVFS-CMA-ES (controlled)")
print("="*60)

bounds = [(-5, 5) for _ in range(3)]

def objective(weights):
    strategy = TradingStrategy(train_data, np.array(weights))
    returns = strategy.backtest()
    metrics = strategy.get_metrics(returns)
    return metrics.get('Sharpe Ratio', -999)

cvfs_optimizer = CVFS_CMAESOptimizer(
    objective, bounds, 3, n_trials=30,
    active_cma=True, mirrored_sampling=True
)
cvfs_weights = cvfs_optimizer.optimize()

cvfs_strategy = TradingStrategy(test_data, cvfs_weights)
cvfs_returns = cvfs_strategy.backtest()
cvfs_metrics = cvfs_strategy.get_metrics(cvfs_returns)

print(f"\nCVFS-CMA-ES Results:")
print(f"  Sharpe Ratio: {cvfs_metrics['Sharpe Ratio']:.3f}")
print(f"  Total Return: {cvfs_metrics['Total Return']:.2%}")
print(f"  Weights: {cvfs_weights}")

print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)
print(f"{'Optimizer':<15} {'Sharpe':<10} {'Return':<12} {'Weights'}")
print("-"*60)
print(f"{'Bayesian':<15} {bo_metrics['Sharpe Ratio']:<10.3f} {bo_metrics['Total Return']:<11.2%} {bo_weights}")
print(f"{'CVFS-CMA-ES':<15} {cvfs_metrics['Sharpe Ratio']:<10.3f} {cvfs_metrics['Total Return']:<11.2%} {cvfs_weights}")
