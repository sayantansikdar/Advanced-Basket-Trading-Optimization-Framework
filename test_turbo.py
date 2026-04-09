"""
Test TuRBO optimizer independently
"""
import numpy as np
from src.optimizers.turbo_optimizer import TuRBOOptimizer

# Simple test function (maximize)
def test_function(x):
    # Simple quadratic with maximum at [0,0,0]
    return -np.sum(x**2) + 10

bounds = [(-5, 5) for _ in range(3)]
optimizer = TuRBOOptimizer(test_function, bounds, 3, n_trials=30, batch_size=2, n_restarts=2)
best = optimizer.optimize()
print(f"Test result - Best: {best}, Value: {optimizer.best_value}")