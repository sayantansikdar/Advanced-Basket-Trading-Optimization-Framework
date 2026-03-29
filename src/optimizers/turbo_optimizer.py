"""
TuRBO (Trust Region Bayesian Optimization) Optimizer
"""
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from .base_optimizer import BaseOptimizer

class TuRBOOptimizer(BaseOptimizer):
    """TuRBO optimizer for basket trading weights"""
    
    def __init__(self, objective_func, bounds, n_assets, n_trials=100):
        super().__init__(objective_func, bounds, n_assets, n_trials)
        self.dim = n_assets
        
        # TuRBO parameters
        self.length = 0.8  # Initial trust region length
        self.length_min = 0.05  # Minimum length
        self.length_max = 1.6  # Maximum length
        self.succ_tol = 3  # Successes needed to expand
        self.fail_tol = 5  # Failures needed to shrink
        self.succ_count = 0
        self.fail_count = 0
        
    def optimize(self):
        """Run TuRBO optimization"""
        print(f"Starting TuRBO optimization with {self.n_trials} trials")
        print(f"Optimizing {self.dim} weights")
        
        # Initial design points
        n_init = min(max(10, 2 * self.dim), self.n_trials // 2)
        X = self._initial_design(n_init)
        y = np.array([self.objective_func(x) for x in X])
        
        # Update best
        best_idx = np.argmax(y)
        self._update_best(X[best_idx], y[best_idx])
        
        # Main optimization loop
        n_evaluations = n_init
        iteration = 0
        
        while n_evaluations < self.n_trials:
            iteration += 1
            
            # Fit Gaussian Process
            gp = self._fit_gp(X, y)
            
            # Find best point in current trust region
            candidate = self._optimize_acquisition(gp, X)
            
            # Evaluate candidate
            y_candidate = self.objective_func(candidate)
            n_evaluations += 1
            
            # Update best
            improved = y_candidate > self.best_value
            self._update_best(candidate, y_candidate)
            
            # Update trust region based on improvement
            if improved:
                self.succ_count += 1
                self.fail_count = 0
                if self.succ_count >= self.succ_tol:
                    # Expand trust region
                    self.length = min(self.length_max, self.length * 2)
                    self.succ_count = 0
                    print(f"  Trust region expanded to {self.length:.3f}")
            else:
                self.succ_count = 0
                self.fail_count += 1
                if self.fail_count >= self.fail_tol:
                    # Shrink trust region and restart
                    self.length = max(self.length_min, self.length / 2)
                    self.fail_count = 0
                    print(f"  Trust region shrunk to {self.length:.3f}")
                    
                    # Restart with new initial points
                    X = self._initial_design(n_init)
                    y = np.array([self.objective_func(x) for x in X])
                    n_evaluations += n_init
                    continue
            
            # Add candidate to dataset
            X = np.vstack([X, candidate])
            y = np.append(y, y_candidate)
            
            # Print progress
            if iteration % 5 == 0:
                print(f"  Iteration {iteration}: Best value = {self.best_value:.4f}")
        
        print(f"TuRBO completed. Best value: {self.best_value:.4f}")
        return self.best_weights
    
    def _initial_design(self, n_points):
        """Generate initial points using Latin Hypercube Sampling"""
        X = np.random.rand(n_points, self.dim)
        for i in range(self.dim):
            lower, upper = self.bounds[i]
            X[:, i] = lower + X[:, i] * (upper - lower)
        return X
    
    def _fit_gp(self, X, y):
        """Fit Gaussian Process model"""
        # Matern kernel with white noise
        kernel = Matern(nu=2.5, length_scale=1.0) + WhiteKernel(noise_level=0.01)
        
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            normalize_y=True,
            alpha=1e-6
        )
        gp.fit(X, y)
        return gp
    
    def _optimize_acquisition(self, gp, X):
        """Optimize Expected Improvement within trust region"""
        # Find best point so far
        best_idx = np.argmax(gp.predict(X))
        center = X[best_idx]
        
        # Generate candidates within trust region
        n_candidates = 1000
        candidates = center + self.length * np.random.randn(n_candidates, self.dim)
        
        # Clip to bounds
        for i in range(self.dim):
            lower, upper = self.bounds[i]
            candidates[:, i] = np.clip(candidates[:, i], lower, upper)
        
        # Calculate Expected Improvement
        mu, sigma = gp.predict(candidates, return_std=True)
        best_y = np.max(gp.predict(X))
        
        # EI formula
        with np.errstate(divide='ignore', invalid='ignore'):
            Z = (mu - best_y) / (sigma + 1e-9)
            ei = (mu - best_y) * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma < 1e-6] = 0  # No improvement if no uncertainty
        
        # Return best candidate
        best_idx = np.argmax(ei)
        return candidates[best_idx]