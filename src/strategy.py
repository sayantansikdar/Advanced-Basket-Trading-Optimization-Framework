"""
Trading strategy implementation for mean-reversion trading
"""
import numpy as np
import pandas as pd
"""
Trading strategy implementation for mean-reversion trading
"""
import numpy as np
import pandas as pd

class TradingStrategy:
    """Mean-reversion trading strategy for basket spreads"""
    
    def __init__(self, prices, weights, entry_threshold=2.0, exit_threshold=0.5, 
                 transaction_cost=0.001):
        """
        Initialize trading strategy
        
        Parameters:
        -----------
        prices : DataFrame
            Price data
        weights : array-like
            Cointegrating weights (should have same length as number of assets)
        entry_threshold : float
            Number of standard deviations to enter position
        exit_threshold : float
            Number of standard deviations to exit position
        transaction_cost : float
            Transaction cost as percentage
        """
        self.prices = prices
        # Ensure weights is a numpy array and has correct shape
        self.weights = np.array(weights, dtype=float).flatten()
        
        # Check if dimensions match
        if len(self.weights) != len(prices.columns):
            raise ValueError(f"Weights length ({len(self.weights)}) must match number of assets ({len(prices.columns)})")
        
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.transaction_cost = transaction_cost
        
        # Calculate log prices and spread
        self.log_prices = np.log(prices)
        self.spread = self._calculate_spread()
        
    def _calculate_spread(self):
        """Calculate spread using weights"""
        spread = np.dot(self.log_prices.values, self.weights)
        return pd.Series(spread, index=self.prices.index)
    
    def _calculate_zscore(self, window=20):
        """Calculate rolling z-score of spread"""
        rolling_mean = self.spread.rolling(window=window).mean()
        rolling_std = self.spread.rolling(window=window).std()
        zscore = (self.spread - rolling_mean) / rolling_std
        return zscore
    
    def generate_signals(self):
        """Generate trading signals based on z-score with proper position holding"""
        zscore = self._calculate_zscore()
        
        signals = pd.Series(0, index=self.prices.index)
        
        # Entry conditions - enter when z-score crosses threshold
        signals[(zscore > self.entry_threshold)] = -1  # Short spread
        signals[(zscore < -self.entry_threshold)] = 1  # Long spread
        
        # Exit conditions - exit when z-score crosses back below exit threshold
        # We need to track positions properly
        position = 0
        final_signals = pd.Series(0, index=self.prices.index)
        
        for i in range(len(signals)):
            sig = signals.iloc[i]
            
            if sig != 0 and position == 0:
                # Enter new position
                position = sig
            elif position != 0:
                # Check if we should exit
                if abs(zscore.iloc[i]) < self.exit_threshold:
                    position = 0
            # else: keep current position
            
            final_signals.iloc[i] = position
        
        return final_signals
    
    def backtest(self):
        """
        Run backtest and calculate returns
        
        Returns:
        --------
        returns : Series
            Strategy returns
        """
        signals = self.generate_signals()
        
        # Calculate spread returns (change in spread)
        spread_returns = self.spread.diff()
        
        # Strategy returns = position * spread_return (shifted to avoid look-ahead bias)
        strategy_returns = signals.shift(1) * spread_returns
        
        # Apply transaction costs when positions change
        position_changes = signals.diff().abs()
        transaction_costs = position_changes * self.transaction_cost
        
        # Net returns
        net_returns = strategy_returns - transaction_costs
        
        # Fill NaN
        net_returns = net_returns.fillna(0)
        
        return net_returns
    
    def get_metrics(self, returns):
        """Calculate performance metrics"""
        if len(returns) == 0 or returns.std() == 0:
            return {
                'Sharpe Ratio': 0,
                'Total Return': 0,
                'Max Drawdown': 0,
                'Profit Factor': 1.0,
                'Win Rate': 0
            }
        
        # Sharpe Ratio (annualized)
        sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-9)
        
        # Total Return
        total_return = (1 + returns).prod() - 1
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Profit Factor
        profits = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = profits / (losses + 1e-9)
        
        # Win Rate
        win_rate = (returns > 0).sum() / (len(returns) + 1e-9)
        
        return {
            'Sharpe Ratio': sharpe,
            'Total Return': total_return,
            'Max Drawdown': max_drawdown,
            'Profit Factor': profit_factor,
            'Win Rate': win_rate
        }
class TradingStrategy:
    """Mean-reversion trading strategy for basket spreads"""
    
    def __init__(self, prices, weights, entry_threshold=2.0, exit_threshold=0.5, 
                 transaction_cost=0.001):
        """
        Initialize trading strategy
        
        Parameters:
        -----------
        prices : DataFrame
            Price data
        weights : array-like
            Cointegrating weights (should have same length as number of assets)
        entry_threshold : float
            Number of standard deviations to enter position
        exit_threshold : float
            Number of standard deviations to exit position
        transaction_cost : float
            Transaction cost as percentage
        """
        self.prices = prices
        # Ensure weights is a numpy array and has correct shape
        self.weights = np.array(weights, dtype=float).flatten()
        
        # Debug: print shapes
        print(f"Debug: prices shape = {prices.shape}, weights shape = {self.weights.shape}")
        
        # Check if dimensions match
        if len(self.weights) != len(prices.columns):
            raise ValueError(f"Weights length ({len(self.weights)}) must match number of assets ({len(prices.columns)})")
        
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.transaction_cost = transaction_cost
        
        # Calculate log prices and spread
        self.log_prices = np.log(prices)
        self.spread = self._calculate_spread()
        
    def _calculate_spread(self):
        """Calculate spread using weights"""
        # Make sure dimensions align
        spread = np.dot(self.log_prices.values, self.weights)
        return pd.Series(spread, index=self.prices.index)
    
    def _calculate_zscore(self, window=20):
        """Calculate rolling z-score of spread"""
        rolling_mean = self.spread.rolling(window=window).mean()
        rolling_std = self.spread.rolling(window=window).std()
        zscore = (self.spread - rolling_mean) / rolling_std
        return zscore
    
    def generate_signals(self):
        """Generate trading signals based on z-score"""
        zscore = self._calculate_zscore()
        
        signals = pd.Series(0, index=self.prices.index)
        
        # Entry conditions
        signals[zscore > self.entry_threshold] = -1  # Short spread
        signals[zscore < -self.entry_threshold] = 1  # Long spread
        
        # Exit conditions
        signals[abs(zscore) < self.exit_threshold] = 0
        
        # Ensure signals are held until exit (simple version - no position holding)
        position = 0
        final_signals = pd.Series(0, index=self.prices.index)
        
        for i, sig in signals.items():
            if sig != 0:
                position = sig
            final_signals[i] = position
        
        return final_signals
    
    def backtest(self):
        """
        Run backtest and calculate returns
        
        Returns:
        --------
        returns : Series
            Strategy returns
        """
        signals = self.generate_signals()
        
        # Calculate spread returns (change in spread)
        spread_returns = self.spread.diff().shift(-1)  # Forward-looking returns
        
        # Strategy returns = position * spread_return
        strategy_returns = signals * spread_returns
        
        # Apply transaction costs when positions change
        position_changes = signals.diff().abs()
        transaction_costs = position_changes * self.transaction_cost
        
        # Net returns
        net_returns = strategy_returns - transaction_costs
        
        # Fill NaN and drop first row
        net_returns = net_returns.fillna(0)
        
        return net_returns
    
    def get_metrics(self, returns):
        """Calculate performance metrics"""
        if len(returns) == 0 or returns.std() == 0:
            return {
                'Sharpe Ratio': 0,
                'Total Return': 0,
                'Max Drawdown': 0,
                'Profit Factor': 0,
                'Win Rate': 0
            }
        
        # Sharpe Ratio (annualized)
        sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-9)
        
        # Total Return
        total_return = (1 + returns).prod() - 1
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Profit Factor
        profits = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = profits / (losses + 1e-9)
        
        # Win Rate
        win_rate = (returns > 0).sum() / (len(returns) + 1e-9)
        
        return {
            'Sharpe Ratio': sharpe,
            'Total Return': total_return,
            'Max Drawdown': max_drawdown,
            'Profit Factor': profit_factor,
            'Win Rate': win_rate
        }
