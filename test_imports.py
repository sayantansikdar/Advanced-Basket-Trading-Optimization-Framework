#!/usr/bin/env python
"""Test all imports for the basket trading project"""

print("Testing imports...")

try:
    import pandas as pd
    print("✓ pandas imported successfully")
except Exception as e:
    print(f"✗ pandas import failed: {e}")

try:
    import numpy as np
    print("✓ numpy imported successfully")
except Exception as e:
    print(f"✗ numpy import failed: {e}")

try:
    import yfinance as yf
    print("✓ yfinance imported successfully")
except Exception as e:
    print(f"✗ yfinance import failed: {e}")

try:
    import sklearn
    print("✓ sklearn imported successfully")
except Exception as e:
    print(f"✗ sklearn import failed: {e}")

try:
    import skopt
    print("✓ skopt imported successfully")
except Exception as e:
    print(f"✗ skopt import failed: {e}")

try:
    import cma
    print("✓ cma imported successfully")
except Exception as e:
    print(f"✗ cma import failed: {e}")

try:
    import matplotlib.pyplot as plt
    print("✓ matplotlib imported successfully")
except Exception as e:
    print(f"✗ matplotlib import failed: {e}")

try:
    import seaborn as sns
    print("✓ seaborn imported successfully")
except Exception as e:
    print(f"✗ seaborn import failed: {e}")

print("\nAll import tests completed!")
