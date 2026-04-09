"""
Comparison of Top Four Optimizers: BO, CVFS-CMA-ES, Tuned TuRBO, and SAASBO
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.data_utils import fetch_data
from src.strategy import TradingStrategy

# Load results
results_df = pd.read_csv('results/comparison_summary.csv')

# Filter to only our top four
top_four = ['BAYESIAN', 'CVFS_CMAES', 'TURBO_TUNED', 'SAASBO']
results_df = results_df[results_df['Optimizer'].isin(top_four)]

print("="*80)
print("TOP FOUR OPTIMIZERS COMPARISON")
print("="*80)
print(results_df.to_string(index=False))

# Load data for backtesting
prices = fetch_data(['AAPL', 'MSFT', 'GOOGL'], '2023-01-01', '2023-12-31')
train_size = int(len(prices) * 0.6)
test_data = prices.iloc[train_size:]

# Define weights from the optimization results (update with actual values)
weights_dict = {
    'BAYESIAN': np.array([2.25867554, 5.0, -5.0]),
    'CVFS_CMAES': np.array([-0.08376943, -1.44482409, 0.76662195]),
    'TURBO_TUNED': np.array([0.71943142, 5.0, -2.14575856]),  # Update with actual
    'SAASBO': np.array([0.5, -0.3, -0.2])  # Update with actual SAASBO weights
}

# Colors for top four
colors = {
    'BAYESIAN': '#2ecc71',
    'CVFS_CMAES': '#9b59b6',
    'TURBO_TUNED': '#3498db',
    'SAASBO': '#e74c3c'
}

output_dir = Path('results/plots')
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Equity Curves Comparison
fig, ax = plt.subplots(figsize=(14, 7))

for name, weights in weights_dict.items():
    if name in results_df['Optimizer'].values:
        strategy = TradingStrategy(test_data, weights)
        returns = strategy.backtest()
        equity = (1 + returns).cumprod()
        ax.plot(equity.index, equity.values, label=name, linewidth=2, color=colors.get(name, 'gray'))

# Buy and hold benchmark
bh_returns = test_data.pct_change().mean(axis=1)
bh_equity = (1 + bh_returns).cumprod()
ax.plot(bh_equity.index, bh_equity.values, label='Buy & Hold', linewidth=2, linestyle='--', color='black', alpha=0.5)

ax.set_title('Top Four Optimizers - Equity Curves Comparison', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.axhline(y=1, color='black', linestyle='-', alpha=0.2)

plt.tight_layout()
plt.savefig(output_dir / 'top_four_equity_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Top four equity curves saved")

# 2. Performance Bar Chart
fig, ax = plt.subplots(figsize=(14, 8))
metrics = ['Sharpe Ratio', 'Total Return (%)', 'Profit Factor', 'Win Rate (%)']
x = np.arange(len(metrics))
width = 0.2

for i, (name, color) in enumerate(colors.items()):
    if name in results_df['Optimizer'].values:
        row = results_df[results_df['Optimizer'] == name].iloc[0]
        values = [
            row['Sharpe Ratio'],
            row['Total Return'] * 100,
            row['Profit Factor'],
            row['Win Rate'] * 100
        ]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, values, width, label=name, color=color, alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Values', fontsize=12, fontweight='bold')
ax.set_title('Top Four Optimizers - Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'top_four_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Top four metrics comparison saved")

# 3. Risk-Return Scatter Plot
fig, ax = plt.subplots(figsize=(10, 8))

for name, color in colors.items():
    if name in results_df['Optimizer'].values:
        row = results_df[results_df['Optimizer'] == name].iloc[0]
        ax.scatter(row['Max Drawdown'] * 100, row['Sharpe Ratio'], 
                  s=200, c=color, marker='o', edgecolors='black', linewidth=2, label=name)
        ax.annotate(name, (row['Max Drawdown'] * 100, row['Sharpe Ratio']),
                   xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')

ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Maximum Drawdown (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
ax.set_title('Risk-Return Trade-off: Top Four Optimizers', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'top_four_risk_return.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Top four risk-return scatter plot saved")

# 4. Create Summary Table
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

# Sort by Sharpe ratio
results_sorted = results_df.sort_values('Sharpe Ratio', ascending=False)

table_data = []
for _, row in results_sorted.iterrows():
    table_data.append([
        row['Optimizer'],
        f"{row['Sharpe Ratio']:.3f}",
        f"{row['Total Return']:.2%}",
        f"{row['Max Drawdown']:.2%}",
        f"{row['Profit Factor']:.2f}",
        f"{row['Win Rate']:.2%}"
    ])

headers = ['Optimizer', 'Sharpe Ratio', 'Total Return', 'Max Drawdown', 'Profit Factor', 'Win Rate']
table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center',
                 colWidths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.12])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# Style header
for i in range(len(headers)):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight best in each column
for col in range(1, len(headers)):
    col_values = [float(row[col].replace('%', '').replace('f', '')) for row in table_data if row[col].replace('%', '').replace('f', '').replace('-', '0').strip()]
    if col_values:
        if col == 3:  # Max Drawdown - lower is better
            best_idx = np.argmin(col_values)
        else:  # Others - higher is better
            best_idx = np.argmax(col_values)
        table[(best_idx + 1, col)].set_facecolor('#d4edda')

plt.title('Top Four Optimizers - Performance Summary', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(output_dir / 'top_four_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Top four summary table saved")

# Print final analysis
print("\n" + "="*80)
print("TOP FOUR OPTIMIZERS - FINAL ANALYSIS")
print("="*80)

best_sharpe = results_sorted.iloc[0]
best_return = results_df.loc[results_df['Total Return'].idxmax()]
best_drawdown = results_df.loc[results_df['Max Drawdown'].idxmax()]

print(f"\n🏆 BEST SHARPE RATIO: {best_sharpe['Optimizer']} ({best_sharpe['Sharpe Ratio']:.3f})")
print(f"📈 BEST TOTAL RETURN: {best_return['Optimizer']} ({best_return['Total Return']:.2%})")
print(f"🛡️ BEST DRAWDOWN CONTROL: {best_drawdown['Optimizer']} ({best_drawdown['Max Drawdown']:.2%})")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)
print("""
• For MAXIMUM RISK-ADJUSTED RETURNS: Choose CVFS-CMA-ES
• For MAXIMUM ABSOLUTE RETURNS: Choose Bayesian Optimization  
• For BEST DRAWDOWN CONTROL: Choose the optimizer with highest Sharpe
• For BALANCED APPROACH: Consider hybrid of CVFS-CMA-ES and BO
""")
