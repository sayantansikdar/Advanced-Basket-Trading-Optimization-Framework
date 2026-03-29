"""
Cointegration analysis using Johansen test
"""
import numpy as np
import pandas as pd

# Try to import statsmodels, provide helpful error if missing
try:
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not installed. Cointegration analysis will be limited.")

class CointegrationAnalyzer:
    """Analyze cointegration relationships between assets"""
    
    def __init__(self, prices):
        """
        Initialize with price data
        
        Parameters:
        -----------
        prices : DataFrame
            Price data for multiple assets
        """
        self.prices = prices
        self.log_prices = np.log(prices)
        self.weights = None
        self.spread = None
        self.results = None
        
    def johansen_test(self, det_order=0, k_ar_diff=1):
        """
        Perform Johansen cointegration test
        
        Parameters:
        -----------
        det_order : int
            Deterministic order (-1 for no constant, 0 for constant term)
        k_ar_diff : int
            Number of lagged differences in the VAR
        """
        if not STATSMODELS_AVAILABLE:
            print("Error: statsmodels is required for Johansen test")
            print("Please install it with: pip install statsmodels")
            return None
            
        # Prepare data
        data = self.log_prices.values
        
        # Ensure data has no NaN
        if np.any(np.isnan(data)):
            print("Warning: NaN values found in data, dropping them")
            valid_idx = ~np.isnan(data).any(axis=1)
            data = data[valid_idx]
        
        # Perform Johansen test
        try:
            result = coint_johansen(data, det_order, k_ar_diff)
            self.results = result
            
            # Get cointegrating vectors (normalized)
            # The eigenvectors are in result.evec
            # We take the first eigenvector (most significant)
            eigenvector = result.evec[:, 0]
            
            # Normalize so last weight = -1
            if abs(eigenvector[-1]) > 1e-10:
                self.weights = eigenvector / eigenvector[-1]
            else:
                # If last weight is near zero, use first eigenvector directly
                self.weights = eigenvector / (eigenvector[-1] + 1e-10)
            
            return self.weights
            
        except Exception as e:
            print(f"Error in Johansen test: {e}")
            return None
    
    def get_weights(self):
        """Get cointegrating weights"""
        if self.weights is None:
            self.johansen_test()
        return self.weights
    
    def calculate_spread(self, weights=None):
        """
        Calculate spread series using given weights
        
        Parameters:
        -----------
        weights : array-like
            Cointegrating weights (if None, uses Johansen weights)
        """
        if weights is None:
            weights = self.get_weights()
        
        if weights is None:
            return None
        
        # Calculate spread = sum(weights * log_prices)
        self.spread = np.dot(self.log_prices.values, weights)
        self.spread = pd.Series(self.spread, index=self.log_prices.index)
        
        return self.spread
    
    def test_stationarity(self, spread=None):
        """
        Test stationarity of spread using ADF test
        
        Parameters:
        -----------
        spread : array-like
            Spread series (if None, uses calculated spread)
        """
        if not STATSMODELS_AVAILABLE:
            print("Warning: statsmodels not available for ADF test")
            return None
            
        if spread is None:
            spread = self.spread
        
        if spread is None:
            return None
        
        # Drop NaN values
        spread_clean = spread.dropna()
        
        # ADF test
        try:
            adf_result = adfuller(spread_clean, autolag='AIC')
            
            return {
                'adf_statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < 0.05
            }
        except Exception as e:
            print(f"Error in ADF test: {e}")
            return None
    
    def get_statistics(self):
        """Get test statistics"""
        if self.results is None:
            return None
        
        return {
            'trace_statistic': self.results.lr1[0],
            'trace_critical_90': self.results.cvt[0][0],
            'trace_critical_95': self.results.cvt[0][1],
            'trace_critical_99': self.results.cvt[0][2],
            'eigen_statistic': self.results.lr2[0],
            'eigen_critical_90': self.results.cvm[0][0],
            'eigen_critical_95': self.results.cvm[0][1],
            'eigen_critical_99': self.results.cvm[0][2]
        }

def get_cointegrating_weights(prices):
    """
    Convenience function to get cointegrating weights
    
    Parameters:
    -----------
    prices : DataFrame
        Price data
    
    Returns:
    --------
    weights : array
        Cointegrating weights
    """
    analyzer = CointegrationAnalyzer(prices)
    return analyzer.get_weights()