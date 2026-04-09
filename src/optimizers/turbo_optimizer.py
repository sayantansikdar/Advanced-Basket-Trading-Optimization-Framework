"""
TuRBO (Trust Region Bayesian Optimization) Optimizer - Robust Implementation
"""
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from .base_optimizer import BaseOptimizer

class TuRBOOptimizer(BaseOptimizer):
    """TuRBO optimizer for basket trading weights"""
    
    def __init__(self, objective_func, bounds, n_assets, n_trials=100, 
                 batch_size=1, n_restarts=1):
        super().__init__(objective_func, bounds, n_assets, n_trials)
        self.dim = n_assets
        self.batch_size = batch_size
        self.n_restarts = n_restarts
        
        # Trust region parameters
        self.length = 0.8
        self.length_min = 0.05
        self.length_max = 1.6
        self.succ_tol = 3
        self.fail_tol = 5
        self.succ_count = 0
        self.fail_count = 0
        
    def optimize(self):
        """Run TuRBO optimization"""
        print(f"Starting TuRBO optimization with {self.n_trials} trials")
        print(f"Optimizing {self.dim} weights")
        
        best_overall = None
        best_value_overall = -np.inf
        
        for restart in range(self.n_restarts):
            print(f"\n--- Restart {restart + 1}/{self.n_restarts} ---")
            result = self._run_single()
            if result[1] > best_value_overall:
                best_value_overall = result[1]
                best_overall = result[0]
        
        self.best_weights = best_overall
        self.best_value = best_value_overall
        
        print(f"\n✓ TuRBO completed")
        print(f"  Best value: {self.best_value:.4f}")
        print(f"  Best weights: {self.best_weights}")
        return self.best_weights
    
    def _run_single(self):
        """Run single TuRBO optimization"""
        # Initial points
        n_init = max(10, 2 * self.dim)
        X = self._latin_hypercube(n_init)
        y = np.array([self.objective_func(x) for x in X])
        
        # Find best so far
        best_idx = np.argmax(y)
        best_x = X[best_idx].copy()
        best_y = y[best_idx]
        
        # Update global best
        if best_y > self.best_value:
            self.best_value = best_y
            self.best_weights = best_x.copy()
        
        # Reset trust region
        self.length = 0.8
        self.succ_count = 0
        self.fail_count = 0
        
        # Main loop
        n_evals = n_init
        iteration = 0
        
        while n_evals < self.n_trials:
            iteration += 1
            
            # Fit GP with error handling
            try:
                gp = self._fit_gp(X, y)
            except Exception as e:
                print(f"  GP fit error: {e}, using random sampling")
                random_candidate = np.array([np.random.uniform(b[0], b[1]) for b in self.bounds])
                y_candidate = self.objective_func(random_candidate)
                n_evals += 1
                
                if y_candidate > best_y:
                    best_y = y_candidate
                    best_x = random_candidate.copy()
                    if y_candidate > self.best_value:
                        self.best_value = y_candidate
                        self.best_weights = random_candidate.copy()
                
                X = np.vstack([X, [random_candidate]])
                y = np.append(y, y_candidate)
                continue
            
            # Generate candidates
            candidates = self._generate_candidates(best_x)
            
            # Calculate Expected Improvement
            ei = self._expected_improvement(candidates, gp, best_y)
            
            # Select best candidate
            best_candidate_idx = np.argmax(ei)
            best_candidate_idx = min(best_candidate_idx, len(candidates) - 1)
            best_candidate = candidates[best_candidate_idx]
            
            # Evaluate candidate
            y_candidate = self.objective_func(best_candidate)
            n_evals += 1
            
            # Update best
            if y_candidate > best_y:
                best_y = y_candidate
                best_x = best_candidate.copy()
                
                if y_candidate > self.best_value:
                    self.best_value = y_candidate
                    self.best_weights = best_candidate.copy()
                
                self.succ_count += 1
                self.fail_count = 0
                
                if self.succ_count >= self.succ_tol:
                    self.length = min(self.length_max, self.length * 2)
                    self.succ_count = 0
                    print(f"  Trust region expanded to {self.length:.3f}")
            else:
                self.succ_count = 0
                self.fail_count += 1
                
                if self.fail_count >= self.fail_tol:
                    self.length = max(self.length_min, self.length / 2)
                    self.fail_count = 0
                    print(f"  Trust region shrunk to {self.length:.3f}")
                    
                    if self.length <= self.length_min:
                        print(f"  Trust region too small, restarting...")
                        break
            
            # Add to history
            X = np.vstack([X, [best_candidate]])
            y = np.append(y, y_candidate)
            
            if iteration % 10 == 0:
                print(f"  Iteration {iteration}: Best={best_y:.4f}, Length={self.length:.3f}, Evals={n_evals}")
        
        return best_x, best_y
    
    def _latin_hypercube(self, n_points):
        """Latin Hypercube Sampling"""
        X = np.random.rand(n_points, self.dim)
        for i in range(self.dim):
            for j in range(n_points):
                X[j, i] = (X[j, i] + j) / n_points
            np.random.shuffle(X[:, i])
            lower, upper = self.bounds[i]
            X[:, i] = lower + X[:, i] * (upper - lower)
        return X
    
    def _fit_gp(self, X, y):
        """Fit Gaussian Process"""
        import warnings
        warnings.filterwarnings('ignore')
        
        kernel = Matern(nu=2.5, length_scale=1.0) + WhiteKernel(noise_level=0.01)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=True,
            alpha=1e-6,
            random_state=42
        )
        gp.fit(X, y)
        return gp
    
    def _generate_candidates(self, center, n_candidates=1000):
        """Generate candidates within trust region"""
        candidates = center + self.length * np.random.randn(n_candidates, self.dim)
        for i in range(self.dim):
            lower, upper = self.bounds[i]
            candidates[:, i] = np.clip(candidates[:, i], lower, upper)
        return candidates
    
    def _expected_improvement(self, X, gp, best_y):
        """Calculate Expected Improvement"""
        mu, sigma = gp.predict(X, return_std=True)
        
        if sigma.ndim > 1:
            sigma = sigma.flatten()
        
        sigma = np.maximum(sigma, 1e-9)
        improvement = mu - best_y
        
        with np.errstate(divide='ignore', invalid='ignore'):
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        ei[improvement <= 0] = 0
        ei = np.nan_to_num(ei, nan=0.0)
        
        return ei.flatten()


# Export the class
__all__ = ['TuRBOOptimizer']