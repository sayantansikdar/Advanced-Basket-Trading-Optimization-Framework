"""
Unified runner for all optimization methods
"""
import numpy as np
import pandas as pd
from src.strategy import TradingStrategy

# Try to import Bayesian optimizer, but don't fail if not available
try:
    from src.bayesian_opt import BasketOptimizer
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("Warning: Bayesian optimizer not available")

# Import our new optimizers
from src.optimizers import CMAESOptimizer, TuRBOOptimizer

class OptimizationRunner:
    """Run and compare different optimization methods"""
    
    def __init__(self, train_data, test_data, config):
        """
        Parameters:
        -----------
        train_data : DataFrame
            Training data for optimization
        test_data : DataFrame
            Test data for evaluation
        config : dict
            Configuration parameters
        """
        self.train_data = train_data
        self.test_data = test_data
        self.config = config
        self.results = {}
        
    def run_all(self, optimizers=['bayesian', 'cmaes', 'turbo']):
        """Run selected optimizers"""
        
        for opt_name in optimizers:
            print(f"\n{'='*60}")
            print(f"Running {opt_name.upper()} Optimization")
            print(f"{'='*60}")
            
            try:
                if opt_name == 'bayesian':
                    if not BAYESIAN_AVAILABLE:
                        print("Bayesian optimizer not available, skipping")
                        continue
                    weights, metrics = self._run_bayesian()
                elif opt_name == 'cmaes':
                    weights, metrics = self._run_cmaes()
                elif opt_name == 'turbo':
                    weights, metrics = self._run_turbo()
                else:
                    print(f"Unknown optimizer: {opt_name}")
                    continue
                
                self.results[opt_name] = {
                    'weights': weights,
                    'metrics': metrics
                }
                
                # Print results
                print(f"\n{opt_name.upper()} Results:")
                print(f"  Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.3f}")
                print(f"  Total Return: {metrics.get('Total Return', 0):.2%}")
                print(f"  Profit Factor: {metrics.get('Profit Factor', 0):.2f}")
                print(f"  Win Rate: {metrics.get('Win Rate', 0):.2%}")
                
            except Exception as e:
                print(f"Error with {opt_name}: {e}")
                import traceback
                traceback.print_exc()
                self.results[opt_name] = None
        
        return self.results
    
    def _run_bayesian(self):
        """Run Bayesian Optimization (existing)"""
        optimizer = BasketOptimizer(
            self.train_data,
            self.config
        )
        
        best_weights, history = optimizer.optimize(
            n_trials=self.config.get('n_trials', 50)
        )
        
        # best_weights should already be the correct length (N assets)
        print(f"Bayesian weights shape: {best_weights.shape}")
        
        # Evaluate on test data
        metrics = self._evaluate_strategy(best_weights, self.test_data)
        
        return best_weights, metrics
    
    def _run_cmaes(self):
        """Run CMA-ES optimization"""
        n_assets = len(self.train_data.columns)
        # For N assets, we need N-1 weights to optimize (the last weight is determined by normalization)
        # But actually, we want all N weights - let's optimize all N
        n_params = n_assets  # Optimize all weights
        bounds = [(-5, 5) for _ in range(n_params)]
        
        # Create objective function
        def objective(weights):
            # weights should be all N weights
            weights = np.array(weights)
            
            # Normalize weights so they sum to something reasonable? Or just use as is
            # For cointegration, we want weights that create a stationary spread
            # We'll just use them directly
            
            # Debug print occasionally
            if np.random.random() < 0.1:
                print(f"Testing weights: {weights}")
            
            metrics = self._evaluate_strategy(weights, self.train_data)
            # Return the metric we want to optimize
            result = metrics.get(self.config.get('metric', 'Sharpe Ratio'), -999)
            return result
        
        # Run CMA-ES
        optimizer = CMAESOptimizer(
            objective, bounds, n_assets,
            self.config.get('n_trials', 50)
        )
        best_weights = optimizer.optimize()
        
        print(f"CMA-ES best weights: {best_weights}")
        
        # Evaluate on test data
        metrics = self._evaluate_strategy(best_weights, self.test_data)
        
        return best_weights, metrics
    
    def _run_turbo(self):
        """Run TuRBO optimization"""
        n_assets = len(self.train_data.columns)
        bounds = [(-5, 5) for _ in range(n_assets)]
        
        def objective(weights):
            weights = np.array(weights)
            metrics = self._evaluate_strategy(weights, self.train_data)
            return metrics.get(self.config.get('metric', 'Sharpe Ratio'), -999)
        
        optimizer = TuRBOOptimizer(
            objective, bounds, n_assets,
            self.config.get('n_trials', 50)
        )
        best_weights = optimizer.optimize()
        
        print(f"TuRBO best weights: {best_weights}")
        
        metrics = self._evaluate_strategy(best_weights, self.test_data)
        
        return best_weights, metrics
    
    def _evaluate_strategy(self, weights, data):
        """Evaluate trading strategy with given weights"""
        try:
            # Ensure weights is a numpy array
            weights = np.array(weights).flatten()
            
            # Debug info
            print(f"Evaluating weights shape: {weights.shape}, data shape: {data.shape}")
            
            # Check dimensions
            if len(weights) != len(data.columns):
                print(f"Warning: weights length ({len(weights)}) != assets ({len(data.columns)})")
                # If weights is longer, take first N
                if len(weights) > len(data.columns):
                    weights = weights[:len(data.columns)]
                # If shorter, pad with zeros
                elif len(weights) < len(data.columns):
                    weights = np.pad(weights, (0, len(data.columns) - len(weights)), 'constant')
            
            strategy = TradingStrategy(data, weights, 
                                      entry_threshold=self.config.get('entry_threshold', 2.0),
                                      exit_threshold=self.config.get('exit_threshold', 0.5),
                                      transaction_cost=self.config.get('transaction_cost', 0.001))
            returns = strategy.backtest()
            
            # Calculate metrics
            metrics = strategy.get_metrics(returns)
            return metrics
        except Exception as e:
            print(f"Error evaluating strategy: {e}")
            return {
                'Sharpe Ratio': -999, 
                'Total Return': -999,
                'Max Drawdown': -999,
                'Profit Factor': -999,
                'Win Rate': -999
            }