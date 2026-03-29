"""
Detailed performance analysis and visualizations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.data_utils import fetch_data
from src.strategy import TradingStrategy

def plot_equity_curves(weights_dict, prices, output_dir):
    """Plot equity curves for different optimizers"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'cmaes': '#2ecc71', 'turbo': '#3498db'}
    
    for idx, (name, weights) in enumerate(weights_dict.items()):
        ax = axes[idx]
        
        # Run backtest
        strategy = TradingStrategy(prices, weights)
        returns = strategy.backtest()
        
        # Calculate equity curve
        equity = (1 + returns).cumprod()
        
        # Plot equity curve
        ax.plot(equity.index, equity.values, label=f'{name.upper()} Strategy', 
                linewidth=2, color=colors.get(name, 'gray'))
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax.set_title(f'{name.upper()} - Equity Curve', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics text box
        metrics = strategy.get_metrics(returns)
        stats = f"Sharpe: {metrics['Sharpe Ratio']:.2f}\n"
        stats += f"Return: {metrics['Total Return']:.2%}\n"
        stats += f"Max DD: {metrics['Max Drawdown']:.2%}\n"
        stats += f"Win Rate: {metrics['Win Rate']:.2%}"
        ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'equity_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Equity curves saved")

def plot_drawdown_analysis(weights_dict, prices, output_dir):
    """Plot drawdown analysis"""
    n_optimizers = len(weights_dict)
    fig, axes = plt.subplots(n_optimizers, 2, figsize=(14, 5*n_optimizers))
    
    # If only one optimizer, make axes 2D for consistent indexing
    if n_optimizers == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (name, weights) in enumerate(weights_dict.items()):
        # Run backtest
        strategy = TradingStrategy(prices, weights)
        returns = strategy.backtest()
        
        # Calculate drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # Plot drawdown
        ax1 = axes[idx, 0]
        ax1.fill_between(drawdown.index, drawdown.values * 100, 0, 
                         color='red', alpha=0.3, label='Drawdown')
        ax1.plot(drawdown.index, drawdown.values * 100, color='red', linewidth=1)
        ax1.set_title(f'{name.upper()} - Drawdown Analysis', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Drawdown (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot returns distribution
        ax2 = axes[idx, 1]
        returns_nonzero = returns[returns != 0]
        if len(returns_nonzero) > 0:
            ax2.hist(returns_nonzero, bins=30, edgecolor='black', alpha=0.7, color='blue')
            ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax2.axvline(x=returns_nonzero.mean(), color='green', linestyle='-', alpha=0.7, 
                        label=f'Mean: {returns_nonzero.mean():.4f}')
        ax2.set_title(f'{name.upper()} - Returns Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Daily Return')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        if len(returns_nonzero) > 0:
            winning_returns = returns_nonzero[returns_nonzero > 0]
            losing_returns = returns_nonzero[returns_nonzero < 0]
            stats = f"Total Trades: {len(returns_nonzero)}\n"
            stats += f"Positive: {len(winning_returns)} ({len(winning_returns)/len(returns_nonzero)*100:.1f}%)\n"
            stats += f"Negative: {len(losing_returns)} ({len(losing_returns)/len(returns_nonzero)*100:.1f}%)\n"
            if len(winning_returns) > 0:
                stats += f"Avg Win: {winning_returns.mean():.4f}\n"
            if len(losing_returns) > 0:
                stats += f"Avg Loss: {losing_returns.mean():.4f}"
            ax2.text(0.95, 0.95, stats, transform=ax2.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'drawdown_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Drawdown analysis saved")
    
def plot_rolling_performance(weights_dict, prices, output_dir):
    """Plot rolling performance metrics"""
    window = 20
    metrics = ['Sharpe Ratio', 'Total Return', 'Profit Factor', 'Win Rate']
    colors = {'cmaes': '#2ecc71', 'turbo': '#3498db'}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    n_periods = len(prices)
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        for name, weights in weights_dict.items():
            rolling_metrics = []
            
            for i in range(window, n_periods):
                window_prices = prices.iloc[i-window:i]
                try:
                    strategy = TradingStrategy(window_prices, weights)
                    returns = strategy.backtest()
                    metrics_dict = strategy.get_metrics(returns)
                    rolling_metrics.append(metrics_dict.get(metric, 0))
                except Exception as e:
                    rolling_metrics.append(0)
            
            # Plot rolling metrics
            ax.plot(prices.index[window:], rolling_metrics, 
                   label=name.upper(), color=colors.get(name, 'gray'), linewidth=2)
        
        ax.set_title(f'Rolling {metric} ({window}-day window)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rolling_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Rolling performance saved")

def plot_cumulative_returns_comparison(weights_dict, prices, output_dir):
    """Plot cumulative returns comparison"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {'cmaes': '#2ecc71', 'turbo': '#3498db', 'buy_hold': '#95a5a6'}
    
    # Calculate buy and hold returns (equal weight)
    buy_hold_returns = prices.pct_change().mean(axis=1)
    buy_hold_cumulative = (1 + buy_hold_returns).cumprod()
    ax.plot(buy_hold_cumulative.index, buy_hold_cumulative.values, 
            label='Buy & Hold (Equal Weight)', color=colors['buy_hold'], linewidth=2, linestyle='--')
    
    # Calculate strategy cumulative returns
    for name, weights in weights_dict.items():
        strategy = TradingStrategy(prices, weights)
        returns = strategy.backtest()
        cumulative = (1 + returns).cumprod()
        ax.plot(cumulative.index, cumulative.values, 
                label=f'{name.upper()} Strategy', color=colors.get(name, 'gray'), linewidth=2)
    
    ax.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cumulative_returns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Cumulative returns comparison saved")

def plot_trade_analysis(weights_dict, prices, output_dir):
    """Analyze and plot trade statistics"""
    fig, axes = plt.subplots(len(weights_dict), 2, figsize=(14, 5*len(weights_dict)))
    if len(weights_dict) == 1:
        axes = [axes]
    
    for idx, (name, weights) in enumerate(weights_dict.items()):
        # Run strategy
        strategy = TradingStrategy(prices, weights)
        returns = strategy.backtest()
        
        # Generate signals
        signals = strategy.generate_signals()
        
        # Find trade entry and exit points
        trades = []
        in_position = False
        entry_idx = None
        entry_value = None
        
        for i in range(len(signals)):
            if signals.iloc[i] != 0 and not in_position:
                in_position = True
                entry_idx = i
                entry_value = (1 + returns.iloc[:i]).prod()
            elif signals.iloc[i] == 0 and in_position:
                in_position = False
                if entry_idx is not None:
                    exit_value = (1 + returns.iloc[:i]).prod()
                    trade_return = (exit_value / entry_value) - 1
                    trades.append(trade_return)
        
        # Plot trade returns distribution
        ax1 = axes[idx][0] if len(weights_dict) > 1 else axes[0]
        if trades:
            ax1.hist(trades, bins=20, edgecolor='black', alpha=0.7, color='blue')
            ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax1.axvline(x=np.mean(trades), color='green', linestyle='-', alpha=0.7, label=f'Mean: {np.mean(trades):.2%}')
            ax1.legend()
        ax1.set_title(f'{name.upper()} - Trade Returns Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Trade Return')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Plot trade sequence
        ax2 = axes[idx][1] if len(weights_dict) > 1 else axes[1]
        if trades:
            colors = ['green' if t > 0 else 'red' for t in trades]
            ax2.bar(range(len(trades)), trades, color=colors, alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_title(f'{name.upper()} - Trade Sequence', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('Trade Return')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        if trades:
            winning_trades = [t for t in trades if t > 0]
            losing_trades = [t for t in trades if t < 0]
            stats = f"Total Trades: {len(trades)}\n"
            stats += f"Winning Trades: {len(winning_trades)} ({len(winning_trades)/len(trades)*100:.1f}%)\n"
            stats += f"Losing Trades: {len(losing_trades)} ({len(losing_trades)/len(trades)*100:.1f}%)\n"
            if winning_trades:
                stats += f"Avg Win: {np.mean(winning_trades):.2%}\n"
            if losing_trades:
                stats += f"Avg Loss: {np.mean(losing_trades):.2%}\n"
            stats += f"Best Trade: {max(trades):.2%}\n"
            stats += f"Worst Trade: {min(trades):.2%}"
            ax2.text(0.98, 0.98, stats, transform=ax2.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'trade_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Trade analysis saved")

def plot_heatmap_correlation(weights_dict, prices, output_dir):
    """Plot correlation heatmap of returns"""
    fig, axes = plt.subplots(1, len(weights_dict), figsize=(6*len(weights_dict), 5))
    if len(weights_dict) == 1:
        axes = [axes]
    
    for idx, (name, weights) in enumerate(weights_dict.items()):
        # Get strategy returns
        strategy = TradingStrategy(prices, weights)
        strategy_returns = strategy.backtest()
        
        # Calculate asset returns
        asset_returns = prices.pct_change().dropna()
        
        # Align indices
        common_idx = strategy_returns.index.intersection(asset_returns.index)
        strategy_returns_aligned = strategy_returns.loc[common_idx]
        asset_returns_aligned = asset_returns.loc[common_idx]
        
        # Create correlation matrix
        corr_data = pd.DataFrame({
            'Strategy': strategy_returns_aligned,
            'AAPL': asset_returns_aligned['AAPL'],
            'MSFT': asset_returns_aligned['MSFT'],
            'GOOGL': asset_returns_aligned['GOOGL']
        })
        corr_matrix = corr_data.corr()
        
        # Plot heatmap
        im = axes[idx].imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
        axes[idx].set_xticks(range(len(corr_matrix.columns)))
        axes[idx].set_yticks(range(len(corr_matrix.columns)))
        axes[idx].set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        axes[idx].set_yticklabels(corr_matrix.columns)
        axes[idx].set_title(f'{name.upper()} - Correlation Heatmap', fontsize=12, fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[idx])
        
        # Add correlation values
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                text = axes[idx].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                     ha="center", va="center", color="black", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Correlation heatmap saved")

def main():
    """Main analysis function"""
    print("Loading results and data...")
    
    # Load results
    results_df = pd.read_csv('results/comparison_summary.csv')
    print("\nResults Summary:")
    print(results_df.to_string())
    
    # Fetch data for backtesting
    prices = fetch_data(['AAPL', 'MSFT', 'GOOGL'], '2023-01-01', '2023-12-31')
    
    # Extract weights from results
    weights_dict = {
        'cmaes': np.array([0.59949504, -0.17014191, -0.48113266]),
        'turbo': np.array([1.43935182, -5.0, 1.9032658])
    }
    
    print("\nGenerating detailed visualizations...")
    
    # Create plots directory
    output_dir = Path('results/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all visualizations
    plot_equity_curves(weights_dict, prices, output_dir)
    plot_drawdown_analysis(weights_dict, prices, output_dir)
    plot_rolling_performance(weights_dict, prices, output_dir)
    plot_cumulative_returns_comparison(weights_dict, prices, output_dir)
    plot_trade_analysis(weights_dict, prices, output_dir)
    plot_heatmap_correlation(weights_dict, prices, output_dir)
    
    print(f"\n✓ All visualizations saved to {output_dir}/")
    print("\nFiles created:")
    for file in sorted(output_dir.glob('*.png')):
        print(f"  - {file.name}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    for name, weights in weights_dict.items():
        print(f"\n{name.upper()}:")
        strategy = TradingStrategy(prices, weights)
        returns = strategy.backtest()
        metrics = strategy.get_metrics(returns)
        
        print(f"  Sharpe Ratio:     {metrics['Sharpe Ratio']:.3f}")
        print(f"  Total Return:     {metrics['Total Return']:.2%}")
        print(f"  Max Drawdown:     {metrics['Max Drawdown']:.2%}")
        print(f"  Profit Factor:    {metrics['Profit Factor']:.2f}")
        print(f"  Win Rate:         {metrics['Win Rate']:.2%}")
        
        # Additional statistics
        print(f"  Total Trades:     {(returns != 0).sum()}")
        print(f"  Avg Daily Return: {returns.mean():.4%}")
        print(f"  Std Daily Return: {returns.std():.4%}")

if __name__ == "__main__":
    main()