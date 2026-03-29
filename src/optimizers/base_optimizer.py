"""
Base class for all optimizers
"""
import numpy as np
from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    """Abstract base class for optimization methods"""
    
    def __init__(self, objective_func, bounds, n_assets, n_trials=100):
        """
        Parameters:
        -----------
        objective_func : callable
            Function that takes weights and returns performance metric
        bounds : list of tuples
            Lower and upper bounds for each weight
        n_assets : int
            Number of assets in the basket
        n_trials : int
            Maximum number of optimization iterations
        """
        self.objective_func = objective_func
        self.bounds = bounds
        self.n_assets = n_assets
        self.n_trials = n_trials
        self.best_weights = None
        self.best_value = -np.inf
        self.history = []  # Track best value over iterations
        
    @abstractmethod
    def optimize(self):
        """Run the optimization and return best weights"""
        pass
    
    def _update_best(self, weights, value):
        """Update best solution if improved"""
        if value > self.best_value:
            self.best_value = value
            self.best_weights = weights.copy()
        self.history.append(self.best_value)
        return self.best_value