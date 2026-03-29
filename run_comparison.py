"""
Main script to run comparison of different optimization methods
"""
import sys
import os

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Check imports first
print("Checking required packages...")
required_packages = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'yfinance': 'yfinance',
    'sklearn': 'scikit-learn',
    'skopt': 'scikit-optimize',
    'cma': 'cma',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn'
}

missing_packages = []
for package, install_name in required_packages.items():
    try:
        __import__(package)
        print(f"  ✓ {package}")
    except ImportError:
        print(f"  ✗ {package} - NOT FOUND")
        missing_packages.append(install_name)

if missing_packages:
    print(f"\nERROR: Missing required packages: {', '.join(missing_packages)}")
    print("Install them with:")
    print(f"pip install {' '.join(missing_packages)}")
    sys.exit(1)

print("\nAll required packages found!\n")

# Now import all required modules
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Import project modules
try:
    from src.data_utils import fetch_data
    print("✓ src.data_utils imported")
except Exception as e:
    print(f"✗ Failed to import src.data_utils: {e}")
    sys.exit(1)

try:
    from src.cointegration import CointegrationAnalyzer
    print("✓ src.cointegration imported")
except Exception as e:
    print(f"✗ Failed to import src.cointegration: {e}")
    sys.exit(1)

try:
    from src.optimizer_runner import OptimizationRunner
    print("✓ src.optimizer_runner imported")
except Exception as e:
    print(f"✗ Failed to import src.optimizer_runner: {e}")
    sys.exit(1)

try:
    from visualization import OptimizationVisualizer
    print("✓ visualization imported")
except Exception as e:
    print(f"✗ Failed to import visualization: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("BASKET TRADING OPTIMIZATION COMPARISON")
print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Compare optimization methods for basket trading')
    parser.add_argument('--assets', nargs='+', required=True,
                       help='List of asset symbols (e.g., AAPL MSFT GOOGL)')
    parser.add_argument('--start-date', required=True,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--n-trials', type=int, default=30,
                       help='Number of optimization trials')
    parser.add_argument('--metric', default='Sharpe Ratio',
                       choices=['Sharpe Ratio', 'Total Return', 'Profit Factor'],
                       help='Metric to optimize for')
    parser.add_argument('--optimizers', nargs='+', 
                       default=['bayesian', 'cmaes', 'turbo'],
                       choices=['bayesian', 'cmaes', 'turbo'],
                       help='Optimizers to run')
    
    args = parser.parse_args()
    
    print(f"\nConfiguration:")
    print(f"  Assets: {args.assets}")
    print(f"  Period: {args.start_date} to {args.end_date}")
    print(f"  Optimizers: {args.optimizers}")
    print(f"  Trials per optimizer: {args.n_trials}")
    print(f"  Optimization metric: {args.metric}")
    print("="*60 + "\n")
    
    # Step 1: Fetch data
    print("Step 1: Fetching price data...")
    try:
        prices = fetch_data(args.assets, args.start_date, args.end_date)
        print(f"  ✓ Retrieved {len(prices)} days of data for {len(args.assets)} assets")
        print(f"  Data range: {prices.index[0].date()} to {prices.index[-1].date()}")
    except Exception as e:
        print(f"  ✗ Error fetching data: {e}")
        print("\nTroubleshooting tips:")
        print("  1. Check your internet connection")
        print("  2. Verify the stock symbols are correct (e.g., AAPL, MSFT, GOOGL)")
        print("  3. Try with fewer assets first")
        return
    
    # Step 2: Prepare train/test split
    print("\nStep 2: Preparing train/test split...")
    train_size = int(len(prices) * 0.6)
    val_size = int(len(prices) * 0.2)
    
    train_data = prices.iloc[:train_size]
    val_data = prices.iloc[train_size:train_size+val_size]
    test_data = prices.iloc[train_size+val_size:]
    
    print(f"  Training data: {train_data.index[0].date()} to {train_data.index[-1].date()} ({len(train_data)} days)")
    print(f"  Validation data: {val_data.index[0].date()} to {val_data.index[-1].date()} ({len(val_data)} days)")
    print(f"  Test data: {test_data.index[0].date()} to {test_data.index[-1].date()} ({len(test_data)} days)")
    
    # Step 3: Setup configuration
    config = {
        'n_trials': args.n_trials,
        'metric': args.metric,
        'entry_threshold': 2.0,
        'exit_threshold': 0.5,
        'transaction_cost': 0.001
    }
    
    # Step 4: Run optimization comparison
    print("\nStep 3: Running optimization comparison...")
    runner = OptimizationRunner(train_data, test_data, config)
    results = runner.run_all(optimizers=args.optimizers)
    
    # Step 5: Generate visualizations
    if any(v is not None for v in results.values()):
        print("\nStep 4: Generating comparison visualizations...")
        try:
            visualizer = OptimizationVisualizer(results)
            visualizer.create_all_plots()
        except Exception as e:
            print(f"  Warning: Could not generate visualizations: {e}")
    else:
        print("\nWarning: No successful optimization results to visualize")
    
    # Step 6: Save results
    print("\nStep 5: Saving results...")
    try:
        os.makedirs('results', exist_ok=True)
        
        summary_data = []
        for name, result in results.items():
            if result is not None:
                row = {'Optimizer': name.upper()}
                row.update(result['metrics'])
                summary_data.append(row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv('results/comparison_summary.csv', index=False)
            print("  ✓ Results saved to results/comparison_summary.csv")
    except Exception as e:
        print(f"  Warning: Could not save results: {e}")
    
    # Print final comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON SUMMARY")
    print("="*60)
    
    for name, result in results.items():
        if result is not None:
            print(f"\n{name.upper()}:")
            metrics = result['metrics']
            print(f"  Sharpe Ratio:     {metrics.get('Sharpe Ratio', 0):.3f}")
            print(f"  Total Return:     {metrics.get('Total Return', 0):.2%}")
            print(f"  Profit Factor:    {metrics.get('Profit Factor', 0):.2f}")
            print(f"  Win Rate:         {metrics.get('Win Rate', 0):.2%}")
            print(f"  Max Drawdown:     {metrics.get('Max Drawdown', 0):.2%}")
    
    print("\n" + "="*60)
    print("✓ COMPARISON COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()