"""
CVFS-CMA-ES: Competitive Variable-Fidelity Surrogate-Assisted CMA-ES
"""
import numpy as np
import cma
from .base_optimizer import BaseOptimizer

class CVFS_CMAESOptimizer(BaseOptimizer):
    """CVFS-CMA-ES optimizer for basket trading weights"""
    
    def __init__(self, objective_func, bounds, n_assets, n_trials=100, 
                 low_fidelity_func=None, active_cma=True, mirrored_sampling=True):
        super().__init__(objective_func, bounds, n_assets, n_trials)
        self.dim = n_assets
        self.low_fidelity_func = low_fidelity_func
        self.active_cma = active_cma
        self.mirrored_sampling = mirrored_sampling
        self.active_ratio = 0.25
        
        # Surrogate model
        self.surrogate = None
        self.high_fidelity_points = []
        self.high_fidelity_values = []
        
        # Boundary handling
        self.use_tanh_transform = True
        
    def _transform_to_bounds(self, x):
        """Transform unbounded to bounded space"""
        if not self.use_tanh_transform:
            return np.clip(x, [b[0] for b in self.bounds], [b[1] for b in self.bounds])
        
        transformed = []
        for i, val in enumerate(x):
            lower, upper = self.bounds[i]
            t = 1 / (1 + np.exp(-val))
            bounded = lower + t * (upper - lower)
            transformed.append(bounded)
        return np.array(transformed)
    
    def optimize(self):
        """Run CVFS-CMA-ES optimization"""
        print(f"Starting CVFS-CMA-ES optimization with {self.n_trials} trials")
        print(f"Optimizing {self.dim} weights")
        print(f"Active CMA: {self.active_cma}, Mirrored sampling: {self.mirrored_sampling}")
        
        # Initial guess in unbounded space
        x0 = np.zeros(self.dim)
        sigma0 = 0.5
        
        # CMA-ES options (only valid parameters)
        options = {
            'popsize': 15,
            'maxfevals': self.n_trials,
            'verbose': -1,
            'tolfun': 1e-12,
            'tolx': 1e-12,
        }
        
        # Add active CMA if enabled
        if self.active_cma:
            options['CMA_active'] = True
        
        # Initialize CMA-ES
        es = cma.CMAEvolutionStrategy(x0, sigma0, options)
        
        print("\nCVFS-CMA-ES Progress:")
        print("-" * 60)
        
        generation = 0
        evaluations = 0
        
        try:
            while evaluations < self.n_trials and not es.stop():
                generation += 1
                
                # Generate solutions
                solutions = es.ask()
                
                # Apply mirrored sampling for better exploration
                if self.mirrored_sampling and generation > 1:
                    n_mirror = max(1, len(solutions) // 4)
                    mirrored = []
                    for i in range(min(n_mirror, len(solutions))):
                        # Mirror around the mean
                        mirror = 2 * es.mean - solutions[i]
                        mirrored.append(mirror)
                    if mirrored:
                        solutions.extend(mirrored)
                
                # Evaluate solutions
                fitnesses = []
                for sol in solutions:
                    # Transform to bounded space
                    bounded_sol = self._transform_to_bounds(sol)
                    
                    # Evaluate objective
                    value = self.objective_func(bounded_sol)
                    evaluations += 1
                    
                    # Track best
                    if value > self.best_value:
                        self.best_value = value
                        self.best_weights = bounded_sol.copy()
                    
                    # CMA-ES minimizes, so return negative value
                    fitnesses.append(-value)
                    
                    # Store for surrogate (if needed later)
                    self.high_fidelity_points.append(bounded_sol)
                    self.high_fidelity_values.append(value)
                
                # Update CMA-ES
                es.tell(solutions, fitnesses)
                
                # Progress update
                if generation % 5 == 0:
                    print(f"  Gen {generation:3d}: Best={self.best_value:.4f}, "
                          f"Sigma={es.sigma:.4f}, Evals={evaluations}")
                    
        except KeyboardInterrupt:
            print("\nOptimization interrupted by user")
        
        print(f"\n✓ CVFS-CMA-ES completed")
        print(f"  Best value: {self.best_value:.4f}")
        print(f"  Total evaluations: {evaluations}")
        print(f"  Best weights: {self.best_weights}")
        
        return self.best_weights
    
    def set_training_data(self, train_data):
        """Set training data for low-fidelity approximation"""
        self.train_data = train_data
        if hasattr(train_data, 'iloc'):
            self.train_data_subset = train_data.iloc[:min(50, len(train_data))]


# Export the class
__all__ = ['CVFS_CMAESOptimizer']