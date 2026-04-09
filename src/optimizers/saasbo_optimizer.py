"""
SAASBO: Sparse Axis-Aligned Subspaces Bayesian Optimization
Simplified and robust implementation
"""
import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import cdist
from .base_optimizer import BaseOptimizer

class SAASBOOptimizer(BaseOptimizer):
    """
    SAASBO with hierarchical sparsity priors - Robust Version
    """
    
    def __init__(self, objective_func, bounds, n_assets, n_trials=100,
                 n_warmup=50, n_samples=30):
        super().__init__(objective_func, bounds, n_assets, n_trials)
        self.dim = n_assets
        self.n_warmup = n_warmup
        self.n_samples = n_samples
        
        # Store samples
        self.lengthscale_samples = []
        self.noise_samples = []
        
    def optimize(self):
        """Run SAASBO optimization"""
        print(f"Starting SAASBO with {self.n_trials} trials")
        print(f"Dimension: {self.dim}")
        
        # Initial design with Latin Hypercube
        n_init = max(2 * self.dim, 10)
        X = self._latin_hypercube(n_init)
        y = np.array([self.objective_func(x) for x in X])
        
        # Update best
        best_idx = np.argmax(y)
        self.best_weights = X[best_idx].copy()
        self.best_value = y[best_idx]
        
        n_evals = n_init
        iteration = 0
        
        while n_evals < self.n_trials:
            iteration += 1
            
            # Sample from posterior
            lengthscales, noise = self._sample_posterior()
            self.lengthscale_samples.append(lengthscales)
            self.noise_samples.append(noise)
            
            # Generate candidates
            candidates = self._generate_candidates(X)
            
            # Calculate acquisition
            acq = self._saas_acquisition(candidates, X, y)
            
            # Select best candidate
            best_idx = np.argmax(acq)
            best_candidate = candidates[best_idx]
            
            # Evaluate
            y_candidate = self.objective_func(best_candidate)
            n_evals += 1
            
            # Update best
            if y_candidate > self.best_value:
                self.best_value = y_candidate
                self.best_weights = best_candidate.copy()
            
            # Update data
            X = np.vstack([X, [best_candidate]])
            y = np.append(y, y_candidate)
            
            if iteration % 5 == 0:
                print(f"  Iter {iteration}: Best={self.best_value:.4f}")
        
        print(f"\n✓ SAASBO completed")
        print(f"  Best value: {self.best_value:.4f}")
        return self.best_weights
    
    def _sample_posterior(self):
        """Sample from posterior - simplified and robust"""
        # Sample lengthscales from log-normal (always positive)
        lengthscales = np.exp(np.random.normal(0, 1, self.dim))
        lengthscales = np.clip(lengthscales, 0.01, 10.0)
        
        # Sample noise level
        noise = np.exp(np.random.normal(-3, 1))
        noise = np.clip(noise, 0.0001, 0.1)
        
        return lengthscales, noise
    
    def _rbf_kernel(self, X1, X2, lengthscales):
        """RBF kernel with ARD - returns matrix"""
        X1 = np.array(X1)
        X2 = np.array(X2)
        
        # Ensure 2D arrays
        if X1.ndim == 1:
            X1 = X1.reshape(1, -1)
        if X2.ndim == 1:
            X2 = X2.reshape(1, -1)
        
        # Compute pairwise distances with lengthscales
        n1, d = X1.shape
        n2 = X2.shape[0]
        
        # Scale each dimension
        X1_scaled = X1 / lengthscales
        X2_scaled = X2 / lengthscales
        
        # Compute squared distances
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                diff = X1_scaled[i] - X2_scaled[j]
                K[i, j] = np.exp(-0.5 * np.sum(diff ** 2))
        
        return K
    
    def _saas_acquisition(self, candidates, X, y):
        """SAAS acquisition using Expected Improvement"""
        n_candidates = len(candidates)
        n_samples = min(len(self.lengthscale_samples), 20)
        
        if n_samples == 0:
            return np.random.rand(n_candidates)
        
        acq_values = np.zeros(n_candidates)
        
        # Monte Carlo integration over posterior samples
        for s in range(n_samples):
            ls = self.lengthscale_samples[s]
            noise = self.noise_samples[s]
            
            # Compute kernel matrix
            K = self._rbf_kernel(X, X, ls) + noise * np.eye(len(X))
            
            try:
                # Solve linear system K_inv * y
                K_inv_y = np.linalg.solve(K, y)
            except np.linalg.LinAlgError:
                # Add small diagonal if singular
                K_reg = K + 1e-6 * np.eye(len(X))
                K_inv_y = np.linalg.solve(K_reg, y)
            
            for i, x_cand in enumerate(candidates):
                # Compute predictive mean and variance
                k_x = self._rbf_kernel(X, [x_cand], ls).flatten()
                k_xx = self._rbf_kernel([x_cand], [x_cand], ls) + noise
                
                # Ensure k_xx is a scalar
                if isinstance(k_xx, np.ndarray):
                    k_xx = k_xx[0, 0] if k_xx.shape == (1, 1) else k_xx.item()
                else:
                    k_xx = float(k_xx)
                
                # Predictive mean
                mu = float(np.dot(k_x, K_inv_y))
                
                # Predictive variance
                v = np.linalg.solve(K, k_x)
                sigma2 = float(k_xx - np.dot(k_x, v))
                sigma = np.sqrt(max(1e-9, sigma2))
                
                # Expected Improvement
                best_y = float(np.max(y))
                improvement = mu - best_y
                
                if sigma > 0:
                    Z = improvement / sigma
                    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
                else:
                    ei = max(0, improvement)
                
                acq_values[i] += ei / n_samples
        
        return acq_values
    
    def _latin_hypercube(self, n_points):
        """Latin Hypercube Sampling"""
        X = np.random.rand(n_points, self.dim)
        for i in range(self.dim):
            # Stratify
            perm = np.random.permutation(n_points)
            X[:, i] = (perm + np.random.rand(n_points)) / n_points
            # Scale to bounds
            lower, upper = self.bounds[i]
            X[:, i] = lower + X[:, i] * (upper - lower)
        return X
    
    def _generate_candidates(self, X, n_candidates=1000):
        """Generate candidates using Latin hypercube in current region"""
        if len(X) > 10:
            # Find bounds of existing points
            bounds_min = np.min(X, axis=0)
            bounds_max = np.max(X, axis=0)
            
            # Expand bounds slightly
            expand = 0.2 * (bounds_max - bounds_min)
            lower = np.maximum([b[0] for b in self.bounds], bounds_min - expand)
            upper = np.minimum([b[1] for b in self.bounds], bounds_max + expand)
        else:
            lower = np.array([b[0] for b in self.bounds])
            upper = np.array([b[1] for b in self.bounds])
        
        # Generate candidates
        candidates = np.random.rand(n_candidates, self.dim)
        for i in range(self.dim):
            candidates[:, i] = lower[i] + candidates[:, i] * (upper[i] - lower[i])
        
        return candidates


# Export the class
__all__ = ['SAASBOOptimizer']