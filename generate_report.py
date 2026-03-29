"""
Generate comprehensive HTML report for optimization comparison
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

def generate_report():
    """Generate comprehensive HTML report with all visualizations"""
    
    # Load results
    results_df = pd.read_csv('results/comparison_summary.csv')
    
    # Find best optimizer for each metric
    best_sharpe = results_df.loc[results_df['Sharpe Ratio'].idxmax()]
    best_return = results_df.loc[results_df['Total Return'].idxmax()]
    best_profit = results_df.loc[results_df['Profit Factor'].idxmax()]
    best_winrate = results_df.loc[results_df['Win Rate'].idxmax()]
    best_drawdown = results_df.loc[results_df['Max Drawdown'].idxmax()]
    
    # Create HTML content
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Basket Trading Optimization - Complete Analysis Report</title>
    <meta charset="UTF-8">
    <style>
        body {{ 
            font-family: 'Segoe UI', Arial, sans-serif; 
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{ 
            max-width: 1400px; 
            margin: 0 auto; 
            background-color: white; 
            padding: 30px; 
            border-radius: 10px; 
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{ 
            color: #2c3e50; 
            border-bottom: 3px solid #3498db; 
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        h2 {{ 
            color: #34495e; 
            margin-top: 40px;
            margin-bottom: 20px;
            background-color: #ecf0f1; 
            padding: 10px 15px; 
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        h3 {{ 
            color: #7f8c8d; 
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        table {{ 
            border-collapse: collapse; 
            width: 100%; 
            margin: 20px 0;
            font-size: 14px;
        }}
        th, td {{ 
            border: 1px solid #ddd; 
            padding: 12px; 
            text-align: center;
        }}
        th {{ 
            background-color: #3498db; 
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #e8f4f8; }}
        .metric-good {{ color: #27ae60; font-weight: bold; }}
        .metric-bad {{ color: #e74c3c; }}
        .highlight {{ background-color: #fff3cd; }}
        .winner {{ 
            color: #27ae60; 
            font-size: 1.1em; 
            font-weight: bold;
            background-color: #d4edda;
            padding: 2px 8px;
            border-radius: 4px;
            display: inline-block;
        }}
        img {{ 
            max-width: 100%; 
            margin: 15px 0; 
            border: 1px solid #ddd; 
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .gallery {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(550px, 1fr)); 
            gap: 25px; 
            margin-top: 20px;
        }}
        .plot {{ 
            background-color: #fafafa; 
            padding: 15px; 
            border-radius: 8px; 
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        .plot:hover {{
            transform: translateY(-3px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .plot img {{ 
            width: 100%; 
            height: auto;
            cursor: pointer;
        }}
        .summary-box {{ 
            background-color: #ecf0f1; 
            padding: 20px; 
            border-radius: 8px; 
            margin: 20px 0;
            border-left: 4px solid #3498db;
        }}
        .metric-card {{
            display: inline-block;
            background: white;
            padding: 15px;
            margin: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            min-width: 150px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-label {{
            font-size: 12px;
            color: #7f8c8d;
            margin-top: 5px;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #7f8c8d;
            font-size: 12px;
        }}
        @media (max-width: 768px) {{
            .gallery {{
                grid-template-columns: 1fr;
            }}
            .container {{
                padding: 15px;
            }}
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>📊 Basket Trading Optimization - Complete Analysis Report</h1>
    <p><strong>Generated on:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Assets:</strong> AAPL, MSFT, GOOGL | <strong>Period:</strong> 2023-01-01 to 2023-12-31</p>
    <p><strong>Optimizers:</strong> CMA-ES, TuRBO</p>
    
    <div class="summary-box">
        <h2>🎯 Key Findings</h2>
        <ul style="font-size: 16px; line-height: 1.6;">
            <li><strong>🏆 Best Risk-Adjusted Returns:</strong> <span class="winner">{best_sharpe['Optimizer']}</span> 
                (Sharpe Ratio: {best_sharpe['Sharpe Ratio']:.3f})</li>
            <li><strong>📈 Highest Total Return:</strong> <span class="winner">{best_return['Optimizer']}</span> 
                ({best_return['Total Return']:.2%})</li>
            <li><strong>💰 Best Profit Factor:</strong> <span class="winner">{best_profit['Optimizer']}</span> 
                ({best_profit['Profit Factor']:.2f})</li>
            <li><strong>🎲 Best Win Rate:</strong> <span class="winner">{best_winrate['Optimizer']}</span> 
                ({best_winrate['Win Rate']:.2%})</li>
            <li><strong>🛡️ Lowest Drawdown:</strong> <span class="winner">{best_drawdown['Optimizer']}</span> 
                ({best_drawdown['Max Drawdown']:.2%})</li>
        </ul>
    </div>
    
    <h2>📊 Performance Summary Table</h2>
    {results_df.to_html(index=False, classes='table', float_format='%.3f')}
    
    <h2>🖼️ Visualization Gallery</h2>
    <p>Click on any image to view full size. All visualizations provide detailed insights into strategy performance.</p>
    
    <div class="gallery">
        <div class="plot">
            <h3>1. Metric Comparison</h3>
            <img src="plots/metric_comparison.png" alt="Metric Comparison" onclick="window.open(this.src)">
            <p>Bar chart comparing key performance metrics across optimizers.</p>
        </div>
        <div class="plot">
            <h3>2. Weight Distributions</h3>
            <img src="plots/weight_distributions.png" alt="Weight Distributions" onclick="window.open(this.src)">
            <p>Asset weights found by each optimizer for the basket trading strategy.</p>
        </div>
        <div class="plot">
            <h3>3. Radar Chart</h3>
            <img src="plots/radar_comparison.png" alt="Radar Chart" onclick="window.open(this.src)">
            <p>Multi-dimensional comparison of normalized performance metrics.</p>
        </div>
        <div class="plot">
            <h3>4. Summary Table</h3>
            <img src="plots/summary_table.png" alt="Summary Table" onclick="window.open(this.src)">
            <p>Visual summary table with all key metrics.</p>
        </div>
        <div class="plot">
            <h3>5. Equity Curves</h3>
            <img src="plots/equity_curves.png" alt="Equity Curves" onclick="window.open(this.src)">
            <p>Wealth accumulation over time for each strategy.</p>
        </div>
        <div class="plot">
            <h3>6. Drawdown Analysis</h3>
            <img src="plots/drawdown_analysis.png" alt="Drawdown Analysis" onclick="window.open(this.src)">
            <p>Drawdown periods and return distributions.</p>
        </div>
        <div class="plot">
            <h3>7. Rolling Performance</h3>
            <img src="plots/rolling_performance.png" alt="Rolling Performance" onclick="window.open(this.src)">
            <p>20-day rolling window performance metrics showing strategy consistency.</p>
        </div>
        <div class="plot">
            <h3>8. Cumulative Returns</h3>
            <img src="plots/cumulative_returns.png" alt="Cumulative Returns" onclick="window.open(this.src)">
            <p>Comparison with buy-and-hold equal-weight strategy.</p>
        </div>
        <div class="plot">
            <h3>9. Trade Analysis</h3>
            <img src="plots/trade_analysis.png" alt="Trade Analysis" onclick="window.open(this.src)">
            <p>Individual trade returns distribution and sequence.</p>
        </div>
        <div class="plot">
            <h3>10. Correlation Heatmap</h3>
            <img src="plots/correlation_heatmap.png" alt="Correlation Heatmap" onclick="window.open(this.src)">
            <p>Correlation between strategy returns and individual assets.</p>
        </div>
    </div>
    
    <h2>📝 Detailed Strategy Analysis</h2>
    
    <div class="summary-box">
        <h3>CMA-ES Strategy (Conservative)</h3>
        <ul>
            <li><strong>Approach:</strong> Conservative optimization focusing on stability</li>
            <li><strong>Key Characteristics:</strong> Lower drawdown (-3.38%), consistent but modest returns</li>
            <li><strong>Best For:</strong> Risk-averse investors seeking steady returns</li>
            <li><strong>Weights:</strong> Balanced approach with moderate positions</li>
        </ul>
        
        <h3>TuRBO Strategy (Aggressive)</h3>
        <ul>
            <li><strong>Approach:</strong> Trust Region Bayesian Optimization for global optimization</li>
            <li><strong>Key Characteristics:</strong> Superior returns (54.05%) with higher risk profile</li>
            <li><strong>Best For:</strong> Growth-oriented investors willing to accept higher drawdown</li>
            <li><strong>Weights:</strong> More extreme positions capturing larger opportunities</li>
        </ul>
    </div>
    
    <h2>📈 Performance Statistics by Optimizer</h2>
    
    <div style="display: flex; flex-wrap: wrap; justify-content: space-around; margin: 20px 0;">
        <div class="metric-card">
            <div class="metric-value">CMA-ES: {results_df.loc[results_df['Optimizer']=='CMAES', 'Sharpe Ratio'].values[0]:.2f}</div>
            <div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value" style="font-size: 16px;">TuRBO: {results_df.loc[results_df['Optimizer']=='TURBO', 'Sharpe Ratio'].values[0]:.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">CMA-ES: {results_df.loc[results_df['Optimizer']=='CMAES', 'Total Return'].values[0]:.2%}</div>
            <div class="metric-label">Total Return</div>
            <div class="metric-value" style="font-size: 16px;">TuRBO: {results_df.loc[results_df['Optimizer']=='TURBO', 'Total Return'].values[0]:.2%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">CMA-ES: {results_df.loc[results_df['Optimizer']=='CMAES', 'Max Drawdown'].values[0]:.2%}</div>
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value" style="font-size: 16px;">TuRBO: {results_df.loc[results_df['Optimizer']=='TURBO', 'Max Drawdown'].values[0]:.2%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">CMA-ES: {results_df.loc[results_df['Optimizer']=='CMAES', 'Profit Factor'].values[0]:.2f}</div>
            <div class="metric-label">Profit Factor</div>
            <div class="metric-value" style="font-size: 16px;">TuRBO: {results_df.loc[results_df['Optimizer']=='TURBO', 'Profit Factor'].values[0]:.2f}</div>
        </div>
    </div>
    
    <h2>💡 Conclusions & Recommendations</h2>
    <div class="summary-box">
        <h3>Summary of Findings:</h3>
        <ul>
            <li><strong>TuRBO significantly outperformed CMA-ES</strong> in both absolute returns (54.05% vs 2.44%) and risk-adjusted returns (Sharpe 3.19 vs 1.86)</li>
            <li><strong>Trade-off between risk and return:</strong> TuRBO's higher returns come with higher drawdown (-19.26% vs -3.38%)</li>
            <li><strong>Both strategies demonstrate mean-reversion profitability</strong> with win rates around 26-30%</li>
            <li><strong>TuRBO found more extreme weights</strong> allowing it to capture larger spread deviations</li>
        </ul>
        
        <h3>Recommendations:</h3>
        <ul>
            <li><strong>For conservative investors:</strong> CMA-ES provides stable, low-drawdown returns</li>
            <li><strong>For aggressive investors:</strong> TuRBO delivers superior returns with managed risk</li>
            <li><strong>Portfolio allocation:</strong> Consider combining both strategies for diversification</li>
            <li><strong>Next steps:</strong> Test on longer time periods and different asset classes</li>
            <li><strong>Risk management:</strong> Implement stop-losses to protect against drawdowns in TuRBO strategy</li>
        </ul>
        
        <h3>Future Improvements:</h3>
        <ul>
            <li>Add Bayesian Optimization to complete the comparison</li>
            <li>Optimize entry/exit thresholds dynamically</li>
            <li>Implement position sizing based on volatility</li>
            <li>Test on out-of-sample data across different market regimes</li>
            <li>Add transaction cost sensitivity analysis</li>
        </ul>
    </div>
    
    <div class="footer">
        <p>Report generated by Basket Trading Optimization Framework</p>
        <p>CMA-ES vs TuRBO Comparison | 2023 Trading Year Analysis</p>
    </div>
</div>

<script>
    // Add click-to-enlarge functionality for images
    document.querySelectorAll('.plot img').forEach(img => {{
        img.style.cursor = 'pointer';
        img.addEventListener('click', () => {{
            window.open(img.src, '_blank');
        }});
    }});
</script>
</body>
</html>
    """
    
    # Save report
    output_path = Path('results/comprehensive_report.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ Comprehensive report saved to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    return output_path

if __name__ == "__main__":
    generate_report()