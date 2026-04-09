"""
TuRBO Optimizer with Proper Hyperparameters
Based on Eriksson et al. 2019 and SMAC3 implementation
"""
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.stats import qmc
from .base_optimizer import BaseOptimizer

class TuRBOTunedOptimizer(BaseOptimizer):
    """
    TuRBO with optimal hyperparameters from the original paper
    """
    
    def __init__(self, objective_func, bounds, n_assets, n_trials=100, 
                 n_restarts=3, batch_size=4):
        super().__init__(objective_func, bounds, n_assets, n_trials)
        self.dim = n_assets
        self.n_restarts = n_restarts
        self.batch_size = batch_size
        
        # Critical TuRBO hyperparameters from the paper
        self.length_init = 0.8
        self.length_min = 0.0078125  # 2^-7 - much smaller for fine-tuning
        self.length_max = 1.6
        self.success_tol = 3
        self.failure_tol = max(4, n_assets)  # Dimension-dependent
        
        # Candidate generation
        self.n_candidates = 5000  # As recommended in the paper
        
        # Sobol sequence for better initial design
        self.sobol_engine = qmc.Sobol(d=n_assets, scramble=True, seed=42)
        
    def optimize(self):
        """Run TuRBO with proper hyperparameters"""
        print(f"Starting Enhanced TuRBO with {self.n_trials} trials")
        print(f"Dimension: {self.dim}, Failure tolerance: {self.failure_tol}")
        
        best_overall = None
        best_value_overall = -np.inf
        
        for restart in range(self.n_restarts):
            print(f"\n--- TuRBO Restart {restart + 1}/{self.n_restarts} ---")
            result = self._run_turbo_run()
            
            if result[1] > best_value_overall:
                best_value_overall = result[1]
                best_overall = result[0]
                print(f"  New global best: {best_value_overall:.4f}")
        
        self.best_weights = best_overall
        self.best_value = best_value_overall
        return self.best_weights
    
    def _run_turbo_run(self):
        """Single TuRBO run with adaptive trust region"""
        # Initial design with Sobol sequence
        n_init = max(2 * self.dim, 10)
        X_init = self._sobol_design(n_init)
        y_init = np.array([self.objective_func(x) for x in X_init])
        
        # Find best initial point
        best_idx = np.argmax(y_init)
        best_x = X_init[best_idx].copy()
        best_y = y_init[best_idx]
        
        # Initialize
        X = X_init.copy()
        y = y_init.copy()
        length = self.length_init
        succ_count = 0
        fail_count = 0
        
        iteration = 0
        
        while len(X) < self.n_trials:
            iteration += 1
            
            # Fit GP
            try:
                gp = self._fit_gp(X, y)
            except Exception as e:
                print(f"  GP fit error: {e}")
                # Random exploration fallback
                candidate = np.array([np.random.uniform(b[0], b[1]) for b in self.bounds])
                y_candidate = self.objective_func(candidate)
                X = np.vstack([X, [candidate]])
                y = np.append(y, y_candidate)
                
                if y_candidate > best_y:
                    best_y = y_candidate
                    best_x = candidate.copy()
                continue
            
            # Generate candidates within trust region
            candidates = self._generate_candidates(best_x, length)
            
            # Calculate Expected Improvement
            ei = self._expected_improvement(candidates, gp, best_y)
            
            # Select top batch_size candidates
            best_indices = np.argsort(ei)[-self.batch_size:]
            batch_candidates = candidates[best_indices]
            
            # Evaluate batch
            batch_y = []
            for candidate in batch_candidates:
                y_candidate = self.objective_func(candidate)
                batch_y.append(y_candidate)
                
                if y_candidate > best_y:
                    best_y = y_candidate
                    best_x = candidate.copy()
            
            # Add batch to history
            X = np.vstack([X, batch_candidates])
            y = np.append(y, batch_y)
            
            # Update trust region based on best candidate in batch
            best_batch_idx = np.argmax(batch_y)
            best_batch_value = batch_y[best_batch_idx]
            
            if best_batch_value > best_y - 1e-6:
                succ_count += 1
                fail_count = 0
                
                if succ_count >= self.success_tol:
                    length = min(self.length_max, length * 2)
                    succ_count = 0
                    print(f"  Length expanded to {length:.4f}")
            else:
                succ_count = 0
                fail_count += 1
                
                if fail_count >= self.failure_tol:
                    length = max(self.length_min, length / 2)
                    fail_count = 0
                    print(f"  Length shrunk to {length:.4f}")
                    
                    if length <= self.length_min:
                        print(f"  Trust region too small, restarting...")
                        break
            
            if iteration % 10 == 0:
                print(f"  Iter {iteration}: Best={best_y:.4f}, Len={length:.4f}")
        
        return best_x, best_y
    
    def _sobol_design(self, n_points):
        """Sobol sequence for better space coverage"""
        n_power = 2 ** int(np.ceil(np.log2(n_points)))
        X_sobol = self.sobol_engine.random(n=n_power)[:n_points]
        
        # Scale to bounds
        for i in range(self.dim):
            lower, upper = self.bounds[i]
            X_sobol[:, i] = lower + X_sobol[:, i] * (upper - lower)
        return X_sobol
    
    def _fit_gp(self, X, y):
        """Fit Gaussian Process"""
        import warnings
        warnings.filterwarnings('ignore')
        
        kernel = Matern(nu=2.5, length_scale=1.0) + WhiteKernel(noise_level=0.01)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            normalize_y=True,
            alpha=1e-6,
            random_state=42
        )
        gp.fit(X, y)
        return gp
    
    def _generate_candidates(self, center, length, n_candidates=None):
        """Generate candidates within trust region"""
        if n_candidates is None:
            n_candidates = self.n_candidates
            
        candidates = center + length * np.random.randn(n_candidates, self.dim)
        
        # Clip to bounds
        for i in range(self.dim):
            lower, upper = self.bounds[i]
            candidates[:, i] = np.clip(candidates[:, i], lower, upper)
        
        return candidates
    
    def _expected_improvement(self, X, gp, best_y):
        """Expected Improvement acquisition"""
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

# At the end of the file, ensure the class is exported
__all__ = ['TuRBOTunedOptimizer']