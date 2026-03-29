"""
Data utilities for fetching and preprocessing price data
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fetch_data(tickers, start_date, end_date):
    """
    Fetch price data for multiple assets from Yahoo Finance
    
    Parameters:
    -----------
    tickers : list
        List of asset symbols (e.g., ['AAPL', 'MSFT', 'GOOGL'])
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    
    Returns:
    --------
    prices : DataFrame
        Adjusted close prices for all tickers
    """
    try:
        # Download data
        print(f"Downloading data for {tickers} from {start_date} to {end_date}...")
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        # Extract adjusted close prices
        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
        elif 'Close' in data.columns:
            prices = data['Close']
        else:
            # Try to get the price data
            if isinstance(data.columns, pd.MultiIndex):
                prices = data.xs('Close', axis=1, level=1)
            else:
                prices = data
        
        # Ensure we have a DataFrame
        if isinstance(prices, pd.Series):
            prices = pd.DataFrame(prices)
        
        # Drop any columns with all NaN
        prices = prices.dropna(axis=1, how='all')
        
        # Forward fill any missing values (updated syntax for newer pandas)
        prices = prices.ffill()  # This replaces fillna(method='ffill')
        
        # Drop any remaining NaN at the beginning
        prices = prices.dropna()
        
        print(f"Successfully fetched data for {len(prices.columns)} assets, {len(prices)} days")
        return prices
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        raise

def get_log_prices(prices):
    """
    Convert prices to log prices
    
    Parameters:
    -----------
    prices : DataFrame
        Price data
    
    Returns:
    --------
    log_prices : DataFrame
        Log-transformed prices
    """
    return np.log(prices)

def get_returns(prices):
    """
    Calculate daily returns
    
    Parameters:
    -----------
    prices : DataFrame
        Price data
    
    Returns:
    --------
    returns : DataFrame
        Daily returns
    """
    return prices.pct_change().dropna()

def align_data(*dataframes):
    """
    Align multiple dataframes to common dates
    
    Parameters:
    -----------
    dataframes : list of DataFrames
        DataFrames to align
    
    Returns:
    --------
    aligned : list of DataFrames
        Aligned DataFrames with common index
    """
    common_index = None
    for df in dataframes:
        if common_index is None:
            common_index = df.index
        else:
            common_index = common_index.intersection(df.index)
    
    aligned = [df.loc[common_index] for df in dataframes]
    return aligned

def load_from_csv(filepath):
    """
    Load price data from CSV file
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
    
    Returns:
    --------
    prices : DataFrame
        Price data with dates as index
    """
    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise

def save_to_csv(prices, filepath):
    """
    Save price data to CSV file
    
    Parameters:
    -----------
    prices : DataFrame
        Price data
    filepath : str
        Path to save CSV file
    """
    try:
        prices.to_csv(filepath)
        print(f"Data saved to {filepath}")
    except Exception as e:
        print(f"Error saving CSV: {e}")
        raise