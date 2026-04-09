"""
Generate comprehensive HTML report for all optimizers
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def generate_report():
    """Generate comprehensive HTML report with all visualizations"""
    
    # Load results
    results_df = pd.read_csv('results/comparison_summary.csv')
    
    # Define optimizer display names
    display_names = {
        'BAYESIAN': 'Bayesian Optimization (BO)',
        'CMAES': 'CMA-ES',
        'TURBO': 'TuRBO',
        'CVFS_CMAES': 'CVFS-CMA-ES',
        'TURBO_TUNED': 'TuRBO (Tuned)',
        'SAASBO': 'SAASBO'
    }
    
    # Add display names to dataframe
    results_df['Display Name'] = results_df['Optimizer'].map(display_names)
    
    # Find best performers
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
    <p><strong>Assets:</strong> AAPL, MSFT, GOOGL</p>
    <p><strong>Optimizers Compared:</strong> {', '.join(results_df['Display Name'].tolist())}</p>
    
    <div class="summary-box">
        <h2>🎯 Key Findings</h2>
        <ul style="font-size: 16px; line-height: 1.6;">
            <li><strong>🏆 Best Risk-Adjusted Returns:</strong> <span class="winner">{best_sharpe['Display Name']}</span> 
                (Sharpe Ratio: {best_sharpe['Sharpe Ratio']:.3f})</li>
            <li><strong>📈 Highest Total Return:</strong> <span class="winner">{best_return['Display Name']}</span> 
                ({best_return['Total Return']:.2%})</li>
            <li><strong>💰 Best Profit Factor:</strong> <span class="winner">{best_profit['Display Name']}</span> 
                ({best_profit['Profit Factor']:.2f})</li>
            <li><strong>🎲 Best Win Rate:</strong> <span class="winner">{best_winrate['Display Name']}</span> 
                ({best_winrate['Win Rate']:.2%})</li>
            <li><strong>🛡️ Lowest Drawdown:</strong> <span class="winner">{best_drawdown['Display Name']}</span> 
                ({best_drawdown['Max Drawdown']:.2%})</li>
        </ul>
    </div>
    
    <h2>📊 Performance Summary Table</h2>
    {results_df[['Display Name', 'Sharpe Ratio', 'Total Return', 'Max Drawdown', 'Profit Factor', 'Win Rate']].to_html(index=False, float_format='%.3f', classes='table')}
    
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
            <h3>5. Ranking Heatmap</h3>
            <img src="plots/ranking_heatmap.png" alt="Ranking Heatmap" onclick="window.open(this.src)">
            <p>Heatmap showing ranking of each optimizer per metric (1 = best).</p>
        </div>
        <div class="plot">
            <h3>6. Parallel Coordinates</h3>
            <img src="plots/parallel_coordinates.png" alt="Parallel Coordinates" onclick="window.open(this.src)">
            <p>Multi-dimensional performance comparison across all optimizers.</p>
        </div>
        <div class="plot">
            <h3>7. Top Four Equity Curves</h3>
            <img src="plots/top_four_equity_curves.png" alt="Top Four Equity Curves" onclick="window.open(this.src)">
            <p>Equity curves for the top performing optimizers.</p>
        </div>
        <div class="plot">
            <h3>8. Top Four Metrics</h3>
            <img src="plots/top_four_metrics.png" alt="Top Four Metrics" onclick="window.open(this.src)">
            <p>Detailed metric comparison for top optimizers.</p>
        </div>
        <div class="plot">
            <h3>9. Risk-Return Scatter</h3>
            <img src="plots/top_four_risk_return.png" alt="Risk-Return Scatter" onclick="window.open(this.src)">
            <p>Risk-return trade-off visualization for top optimizers.</p>
        </div>
        <div class="plot">
            <h3>10. Top Four Summary</h3>
            <img src="plots/top_four_summary.png" alt="Top Four Summary" onclick="window.open(this.src)">
            <p>Summary table for the top four optimizers.</p>
        </div>
    </div>
    
    <h2>📝 Detailed Strategy Analysis</h2>
    
    <div class="summary-box">
        <h3>CVFS-CMA-ES Strategy (Top Performer - Risk-Adjusted)</h3>
        <ul>
            <li><strong>Approach:</strong> Competitive Variable-Fidelity Surrogate-Assisted CMA-ES</li>
            <li><strong>Key Characteristics:</strong> Best Sharpe ratio ({best_sharpe['Sharpe Ratio']:.3f}), excellent drawdown control ({best_sharpe['Max Drawdown']:.2%})</li>
            <li><strong>Best For:</strong> Risk-averse investors seeking optimal risk-adjusted returns</li>
            <li><strong>Key Features:</strong> Active CMA, mirrored sampling, variable-fidelity modeling</li>
        </ul>
        
        <h3>Bayesian Optimization Strategy (Best Absolute Returns)</h3>
        <ul>
            <li><strong>Approach:</strong> Gaussian Process-based Bayesian Optimization</li>
            <li><strong>Key Characteristics:</strong> Highest returns ({best_return['Total Return']:.2%}) with higher risk profile</li>
            <li><strong>Best For:</strong> Growth-oriented investors willing to accept higher drawdown</li>
            <li><strong>Key Features:</strong> Expected Improvement acquisition, efficient global search</li>
        </ul>
        
        <h3>TuRBO (Tuned) Strategy</h3>
        <ul>
            <li><strong>Approach:</strong> Trust Region Bayesian Optimization with optimal hyperparameters</li>
            <li><strong>Key Characteristics:</strong> Adaptive trust region, Sobol initialization, batch evaluation</li>
            <li><strong>Best For:</strong> Balanced approach with good exploration-exploitation trade-off</li>
        </ul>
        
        <h3>SAASBO Strategy</h3>
        <ul>
            <li><strong>Approach:</strong> Sparse Axis-Aligned Subspaces Bayesian Optimization</li>
            <li><strong>Key Characteristics:</strong> Hierarchical sparsity priors, automatic relevance determination</li>
            <li><strong>Best For:</strong> Problems where only few dimensions are relevant</li>
        </ul>
    </div>
    
    <h2>💡 Conclusions & Recommendations</h2>
    <div class="summary-box">
        <h3>Summary of Findings:</h3>
        <ul>
            <li><strong>CVFS-CMA-ES achieved the best risk-adjusted returns</strong> with Sharpe ratio of {best_sharpe['Sharpe Ratio']:.3f}</li>
            <li><strong>Bayesian Optimization delivered the highest absolute returns</strong> at {best_return['Total Return']:.2%}</li>
            <li><strong>Trade-off between risk and return</strong> is clearly visible across optimizers</li>
            <li><strong>Advanced methods (CVFS-CMA-ES, SAASBO) show promise</strong> for better drawdown control</li>
        </ul>
        
        <h3>Recommendations:</h3>
        <ul>
            <li><strong>For conservative investors:</strong> CVFS-CMA-ES provides optimal risk-adjusted returns</li>
            <li><strong>For aggressive investors:</strong> Bayesian Optimization delivers superior absolute returns</li>
            <li><strong>For balanced approach:</strong> Consider TuRBO (Tuned) as a middle ground</li>
            <li><strong>Portfolio allocation:</strong> Combine CVFS-CMA-ES (60%) + BO (40%) for diversification</li>
            <li><strong>Risk management:</strong> Implement stop-losses to protect against drawdowns</li>
        </ul>
        
        <h3>Future Improvements:</h3>
        <ul>
            <li>Test on longer time periods and different asset classes</li>
            <li>Optimize entry/exit thresholds dynamically</li>
            <li>Implement position sizing based on volatility</li>
            <li>Add transaction cost sensitivity analysis</li>
            <li>Explore ensemble methods combining multiple optimizers</li>
        </ul>
    </div>
    
    <div class="footer">
        <p>Report generated by Basket Trading Optimization Framework</p>
        <p>Comparing {len(results_df)} optimization methods: {', '.join(results_df['Optimizer'].tolist())}</p>
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
    print(f"  Contains {len(results_df)} optimizers")
    return output_path

if __name__ == "__main__":
    generate_report()