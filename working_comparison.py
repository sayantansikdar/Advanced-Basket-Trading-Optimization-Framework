"""
Comparison using the working configuration from your successful run
"""
import numpy as np
import pandas as pd
from src.data_utils import fetch_data
from src.strategy import TradingStrategy
from src.bayesian_opt import BasketOptimizer
from src.optimizers import CVFS_CMAESOptimizer

# Load data
prices = fetch_data(['AAPL', 'MSFT', 'GOOGL'], '2023-01-01', '2023-12-31')

# Use the exact split from your working run
train_size = 150
val_size = 50

train_data = prices.iloc[:train_size]
val_data = prices.iloc[train_size:train_size+val_size]
test_data = prices.iloc[train_size+val_size:]

print(f"Train: {len(train_data)} days")
print(f"Validation: {len(val_data)} days") 
print(f"Test: {len(test_data)} days")

config = {
    'n_trials': 50,
    'metric': 'Sharpe Ratio',
    'entry_threshold': 2.0,
    'exit_threshold': 0.5,
    'transaction_cost': 0.001
}

# Run Bayesian Optimization
print("\n" + "="*60)
print("Bayesian Optimization")
print("="*60)

bo_optimizer = BasketOptimizer(train_data, config)
bo_weights, _ = bo_optimizer.optimize(n_trials=50)

# Evaluate on test
bo_strategy = TradingStrategy(test_data, bo_weights)
bo_returns = bo_strategy.backtest()
bo_metrics = bo_strategy.get_metrics(bo_returns)

print(f"\nBayesian Test Results:")
print(f"  Sharpe Ratio: {bo_metrics['Sharpe Ratio']:.3f}")
print(f"  Total Return: {bo_metrics['Total Return']:.2%}")
print(f"  Max Drawdown: {bo_metrics['Max Drawdown']:.2%}")

# Run CVFS-CMA-ES
print("\n" + "="*60)
print("CVFS-CMA-ES Optimization")
print("="*60)

bounds = [(-5, 5) for _ in range(3)]

def objective(weights):
    strategy = TradingStrategy(train_data, np.array(weights))
    returns = strategy.backtest()
    metrics = strategy.get_metrics(returns)
    return metrics.get('Sharpe Ratio', -999)

cvfs_optimizer = CVFS_CMAESOptimizer(
    objective, bounds, 3, n_trials=50,
    active_cma=True, mirrored_sampling=True
)
cvfs_weights = cvfs_optimizer.optimize()

cvfs_strategy = TradingStrategy(test_data, cvfs_weights)
cvfs_returns = cvfs_strategy.backtest()
cvfs_metrics = cvfs_strategy.get_metrics(cvfs_returns)

print(f"\nCVFS-CMA-ES Test Results:")
print(f"  Sharpe Ratio: {cvfs_metrics['Sharpe Ratio']:.3f}")
print(f"  Total Return: {cvfs_metrics['Total Return']:.2%}")
print(f"  Max Drawdown: {cvfs_metrics['Max Drawdown']:.2%}")

print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)
print(f"\n{'Optimizer':<15} {'Sharpe':<10} {'Return':<12} {'Drawdown':<12}")
print("-"*60)
print(f"{'Bayesian':<15} {bo_metrics['Sharpe Ratio']:<10.3f} {bo_metrics['Total Return']:<11.2%} {bo_metrics['Max Drawdown']:<11.2%}")
print(f"{'CVFS-CMA-ES':<15} {cvfs_metrics['Sharpe Ratio']:<10.3f} {cvfs_metrics['Total Return']:<11.2%} {cvfs_metrics['Max Drawdown']:<11.2%}")
