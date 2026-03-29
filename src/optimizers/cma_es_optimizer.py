"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) Optimizer
"""
import numpy as np
import cma
from .base_optimizer import BaseOptimizer

class CMAESOptimizer(BaseOptimizer):
    """CMA-ES optimizer for basket trading weights"""
    
    def __init__(self, objective_func, bounds, n_assets, n_trials=100):
        super().__init__(objective_func, bounds, n_assets, n_trials)
        self.dim = n_assets  # Number of weights to optimize
        
    def optimize(self):
        """Run CMA-ES optimization"""
        print(f"Starting CMA-ES optimization with {self.n_trials} trials")
        print(f"Optimizing {self.dim} weights")
        
        # Initial guess - random weights around zero
        x0 = np.random.randn(self.dim) * 0.5
        sigma0 = 0.5  # Initial step size
        
        # Define the fitness function (CMA-ES minimizes, so we negate)
        def fitness(x):
            # x is already in the bounded space
            value = self.objective_func(x)
            # Update best tracking
            self._update_best(x, value)
            # CMA-ES minimizes, so we return negative value
            return -value
        
        # Configure CMA-ES
        options = {
            'popsize': 15,  # Population size
            'maxfevals': self.n_trials,
            'verbose': -1,  # Quiet mode
            'tolfun': 1e-5,  # Stop tolerance
        }
        
        # Run optimization
        es = cma.CMAEvolutionStrategy(x0, sigma0, options)
        
        try:
            iteration = 0
            while not es.stop():
                solutions = es.ask()  # Get candidate solutions
                fitnesses = [fitness(s) for s in solutions]  # Evaluate them
                es.tell(solutions, fitnesses)  # Update CMA-ES
                iteration += 1
                
                # Print progress every 10 iterations
                if iteration % 10 == 0:
                    print(f"  Iteration {iteration}: Best value = {self.best_value:.4f}")
                    
        except KeyboardInterrupt:
            print("\nOptimization interrupted by user")
        
        # Get the best result
        best_weights = self.best_weights if self.best_weights is not None else es.result.xbest
        
        print(f"CMA-ES completed. Best value: {self.best_value:.4f}")
        return best_weights