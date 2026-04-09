"""
Bayesian Optimization for basket trading weights
"""
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from src.strategy import TradingStrategy
import warnings
warnings.filterwarnings('ignore')

class BasketOptimizer:
    """Bayesian Optimization for basket trading weights"""
    
    def __init__(self, train_data, config):
        """
        Initialize Bayesian optimizer
        
        Parameters:
        -----------
        train_data : DataFrame
            Training data for optimization
        config : dict
            Configuration parameters
        """
        self.train_data = train_data
        self.config = config
        self.n_assets = len(train_data.columns)
        self.best_value_history = []
        
    def optimize(self, n_trials=50):
        """
        Run Bayesian optimization
        
        Parameters:
        -----------
        n_trials : int
            Number of optimization trials
            
        Returns:
        --------
        best_weights : array
            Optimal weights found
        history : dict
            Optimization history
        """
        print(f"Starting Bayesian Optimization with {n_trials} trials")
        print(f"Optimizing {self.n_assets} weights")
        
        # Define search space for all weights
        dimensions = [Real(-5, 5, name=f'w{i}') for i in range(self.n_assets)]
        
        # Track best value
        self.best_value = -np.inf
        self.best_weights_track = None
        
        # Objective function to maximize
        def objective_function(weights):
            # Convert to numpy array
            weights = np.array(weights)
            
            try:
                # Run strategy on training data
                strategy = TradingStrategy(
                    self.train_data, 
                    weights,
                    entry_threshold=self.config.get('entry_threshold', 2.0),
                    exit_threshold=self.config.get('exit_threshold', 0.5),
                    transaction_cost=self.config.get('transaction_cost', 0.001)
                )
                returns = strategy.backtest()
                metrics = strategy.get_metrics(returns)
                
                # Get the metric we want to optimize
                metric_name = self.config.get('metric', 'Sharpe Ratio')
                value = metrics.get(metric_name, -999)
                
                # Track best value
                if value > self.best_value:
                    self.best_value = value
                    self.best_weights_track = weights.copy()
                
                self.best_value_history.append(self.best_value)
                
                # Handle invalid values
                if np.isnan(value) or np.isinf(value):
                    return 999
                    
                return -value  # Minimize negative metric
                
            except Exception as e:
                print(f"Error evaluating weights {weights}: {e}")
                return 999
        
        # Run optimization
        print("\nOptimization progress:")
        result = gp_minimize(
            objective_function,
            dimensions,
            n_calls=n_trials,
            n_initial_points=min(10, n_trials // 2),
            acq_func='EI',  # Expected Improvement
            random_state=42,
            verbose=False
        )
        
        # Extract best weights
        if self.best_weights_track is not None:
            best_weights = self.best_weights_track
        else:
            best_weights = np.array(result.x)
        
        # Get best value
        best_value = -result.fun if result.fun > -999 else self.best_value
        
        print(f"\n✓ Bayesian Optimization completed")
        print(f"  Best {self.config.get('metric', 'Sharpe Ratio')}: {best_value:.4f}")
        print(f"  Best weights: {[round(w, 4) for w in best_weights]}")
        
        # Create history tracking
        history = {
            'best_weights': best_weights,
            'best_value': best_value,
            'optimizer_result': result,
            'x_iters': result.x_iters,
            'func_vals': result.func_vals,
            'best_history': self.best_value_history
        }
        
        return best_weights, history