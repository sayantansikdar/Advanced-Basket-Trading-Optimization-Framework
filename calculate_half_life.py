"""
Calculate half-life of mean reversion
"""
import numpy as np
import pandas as pd
from src.data_utils import fetch_data
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

# Load data
prices = fetch_data(['AAPL', 'MSFT', 'GOOGL'], '2023-01-01', '2024-12-31')
weights = np.array([2.25867554, 5.0, -5.0])

log_prices = np.log(prices)
spread = np.dot(log_prices.values, weights)

# Calculate half-life using Ornstein-Uhlenbeck process
spread_lag = spread[:-1]
spread_diff = np.diff(spread)

# Regress spread_diff on spread_lag
model = sm.OLS(spread_diff, sm.add_constant(spread_lag))
results = model.fit()

# Half-life = -ln(2) / coefficient
theta = -results.params[1]
half_life = -np.log(2) / theta if theta > 0 else np.inf

print(f"Mean Reversion Half-Life: {half_life:.1f} days")
print(f"Test period length: {len(spread)} days")
print(f"Recommendation: {'Strategy may work' if half_life < len(spread)/4 else 'Strategy may fail - half-life too long'}")
