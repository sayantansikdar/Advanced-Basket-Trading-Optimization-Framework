"""
Unified runner for all optimization methods
"""
import random
import numpy as np
import pandas as pd
from src.strategy import TradingStrategy

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
set_seed(42)

# Try to import Bayesian optimizer
try:
    from src.bayesian_opt import BasketOptimizer
    BAYESIAN_AVAILABLE = True
except ImportError as e:
    BAYESIAN_AVAILABLE = False
    print(f"Warning: Bayesian optimizer not available: {e}")

# Import all optimizers
from src.optimizers import CMAESOptimizer, TuRBOOptimizer, CVFS_CMAESOptimizer
from src.optimizers.turbo_optimizer_tuned import TuRBOTunedOptimizer
from src.optimizers.saasbo_optimizer import SAASBOOptimizer

class OptimizationRunner:
    """Run and compare different optimization methods"""
    
    def __init__(self, train_data, test_data, config):
        self.train_data = train_data
        self.test_data = test_data
        self.config = config
        self.results = {}
        
    def run_all(self, optimizers=None):
        """Run selected optimizers"""
        if optimizers is None:
            optimizers = ['bayesian', 'cmaes', 'turbo', 'cvfs_cmaes', 'turbo_tuned', 'saasbo']
        
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
                elif opt_name == 'cvfs_cmaes':
                    weights, metrics = self._run_cvfs_cmaes()
                elif opt_name == 'turbo_tuned':
                    weights, metrics = self._run_turbo_tuned()
                elif opt_name == 'saasbo':
                    weights, metrics = self._run_saasbo()
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
                print(f"  Max Drawdown: {metrics.get('Max Drawdown', 0):.2%}")
                
            except Exception as e:
                print(f"Error with {opt_name}: {e}")
                import traceback
                traceback.print_exc()
                self.results[opt_name] = None
        
        return self.results
    
    def _run_bayesian(self):
        """Run Bayesian Optimization"""
        try:
            print("Initializing Bayesian Optimizer...")
            optimizer = BasketOptimizer(self.train_data, self.config)
            best_weights, history = optimizer.optimize(n_trials=self.config.get('n_trials', 50))
            print(f"BO best weights: {best_weights}")
            metrics = self._evaluate_strategy(best_weights, self.test_data)
            return best_weights, metrics
        except Exception as e:
            print(f"Error in Bayesian Optimization: {e}")
            n_assets = len(self.train_data.columns)
            dummy_weights = np.ones(n_assets) / n_assets
            return dummy_weights, {'Sharpe Ratio': -999, 'Total Return': -999, 'Max Drawdown': -999, 'Profit Factor': -999, 'Win Rate': -999}
    
    def _run_cmaes(self):
        """Run CMA-ES optimization"""
        n_assets = len(self.train_data.columns)
        bounds = [(-5, 5) for _ in range(n_assets)]
        
        def objective(weights):
            weights = np.array(weights)
            metrics = self._evaluate_strategy(weights, self.train_data)
            return metrics.get(self.config.get('metric', 'Sharpe Ratio'), -999)
        
        optimizer = CMAESOptimizer(objective, bounds, n_assets, self.config.get('n_trials', 50))
        best_weights = optimizer.optimize()
        print(f"CMA-ES best weights: {best_weights}")
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
        
        optimizer = TuRBOOptimizer(objective, bounds, n_assets, self.config.get('n_trials', 50), batch_size=2, n_restarts=2)
        best_weights = optimizer.optimize()
        print(f"TuRBO best weights: {best_weights}")
        metrics = self._evaluate_strategy(best_weights, self.test_data)
        return best_weights, metrics

    def _run_cvfs_cmaes(self):
        """Run CVFS-CMA-ES optimization"""
        n_assets = len(self.train_data.columns)
        bounds = [(-5, 5) for _ in range(n_assets)]
        
        train_subset = self.train_data.iloc[:min(50, len(self.train_data))]
        
        def low_fidelity_objective(weights):
            try:
                from src.strategy import TradingStrategy
                strategy = TradingStrategy(train_subset, np.array(weights))
                returns = strategy.backtest()
                metrics = strategy.get_metrics(returns)
                return metrics.get(self.config.get('metric', 'Sharpe Ratio'), -999)
            except:
                return -999
        
        def high_fidelity_objective(weights):
            weights = np.array(weights)
            metrics = self._evaluate_strategy(weights, self.train_data)
            return metrics.get(self.config.get('metric', 'Sharpe Ratio'), -999)
        
        optimizer = CVFS_CMAESOptimizer(high_fidelity_objective, bounds, n_assets,
            self.config.get('n_trials', 50), low_fidelity_func=low_fidelity_objective,
            active_cma=True, mirrored_sampling=True)
        
        if hasattr(optimizer, 'set_training_data'):
            optimizer.set_training_data(self.train_data)
        
        best_weights = optimizer.optimize()
        print(f"CVFS-CMA-ES best weights: {best_weights}")
        metrics = self._evaluate_strategy(best_weights, self.test_data)
        return best_weights, metrics
    
    def _run_turbo_tuned(self):
        """Run enhanced TuRBO with optimal hyperparameters"""
        n_assets = len(self.train_data.columns)
        bounds = [(-5, 5) for _ in range(n_assets)]
        
        def objective(weights):
            weights = np.array(weights)
            metrics = self._evaluate_strategy(weights, self.train_data)
            return metrics.get(self.config.get('metric', 'Sharpe Ratio'), -999)
        
        optimizer = TuRBOTunedOptimizer(objective, bounds, n_assets,
            self.config.get('n_trials', 50), n_restarts=3, batch_size=4)
        best_weights = optimizer.optimize()
        print(f"Tuned TuRBO best weights: {best_weights}")
        metrics = self._evaluate_strategy(best_weights, self.test_data)
        return best_weights, metrics
    
    def _run_saasbo(self):
        """Run SAASBO optimization"""
        n_assets = len(self.train_data.columns)
        bounds = [(-5, 5) for _ in range(n_assets)]
        
        def objective(weights):
            weights = np.array(weights)
            metrics = self._evaluate_strategy(weights, self.train_data)
            return metrics.get(self.config.get('metric', 'Sharpe Ratio'), -999)
        
        optimizer = SAASBOOptimizer(objective, bounds, n_assets,
            self.config.get('n_trials', 50), n_warmup=50, n_samples=30)
        best_weights = optimizer.optimize()
        print(f"SAASBO best weights: {best_weights}")
        metrics = self._evaluate_strategy(best_weights, self.test_data)
        return best_weights, metrics
        
    def _evaluate_strategy(self, weights, data):
        """Evaluate trading strategy with given weights"""
        try:
            weights = np.array(weights).flatten()
            
            if len(weights) != len(data.columns):
                if len(weights) > len(data.columns):
                    weights = weights[:len(data.columns)]
                elif len(weights) < len(data.columns):
                    weights = np.pad(weights, (0, len(data.columns) - len(weights)), 'constant')
            
            strategy = TradingStrategy(data, weights, 
                                      entry_threshold=self.config.get('entry_threshold', 2.0),
                                      exit_threshold=self.config.get('exit_threshold', 0.5),
                                      transaction_cost=self.config.get('transaction_cost', 0.001))
            returns = strategy.backtest()
            metrics = strategy.get_metrics(returns)
            return metrics
        except Exception as e:
            print(f"Error evaluating strategy: {e}")
            return {'Sharpe Ratio': -999, 'Total Return': -999, 'Max Drawdown': -999, 'Profit Factor': -999, 'Win Rate': -999}