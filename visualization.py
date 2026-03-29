"""
Visualization utilities for comparing optimization methods
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

class OptimizationVisualizer:
    """Create comparison visualizations for different optimizers"""
    
    def __init__(self, results, output_dir='results/plots'):
        """
        Parameters:
        -----------
        results : dict
            Dictionary with optimizer names as keys and results as values
        output_dir : str
            Directory to save plots
        """
        # Filter out None results
        self.results = {k: v for k, v in results.items() if v is not None}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_all_plots(self):
        """Generate all comparison plots"""
        if not self.results:
            print("No valid results to visualize")
            return
            
        print("\nGenerating visualizations...")
        
        try:
            self.plot_metric_comparison()
            self.plot_weight_distributions()
            self.plot_performance_radar()
            self.create_summary_table()
            print(f"Plots saved to {self.output_dir}")
        except Exception as e:
            print(f"Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()
        
    def plot_metric_comparison(self):
        """Bar chart comparing key metrics"""
        # Prepare data
        metrics = ['Sharpe Ratio', 'Total Return', 'Profit Factor', 'Win Rate']
        data = []
        optimizer_names = list(self.results.keys())
        
        for name in optimizer_names:
            result = self.results[name]
            row = []
            for metric in metrics:
                val = result['metrics'].get(metric, 0)
                if metric == 'Total Return':
                    val = val * 100  # Convert to percentage
                row.append(val)
            data.append(row)
        
        if not data:
            print("No data available for metric comparison")
            return
        
        # Create bar chart
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = [d[idx] for d in data]
            
            bars = ax.bar(optimizer_names, values, color=colors[:len(optimizer_names)])
            ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric if metric != 'Total Return' else 'Total Return (%)')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}' if metric != 'Total Return' else f'{val:.1f}%',
                       ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metric_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def plot_weight_distributions(self):
        """Compare weight distributions"""
        n_optimizers = len(self.results)
        if n_optimizers == 0:
            return
            
        fig, axes = plt.subplots(1, n_optimizers, figsize=(5*n_optimizers, 5))
        if n_optimizers == 1:
            axes = [axes]
            
        for idx, (name, result) in enumerate(self.results.items()):
            weights = result['weights']
            ax = axes[idx]
            
            # Create bar plot
            colors = ['green' if w > 0 else 'red' for w in weights]
            bars = ax.bar(range(len(weights)), weights, color=colors, alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_title(f'{name.upper()} - Asset Weights', fontsize=12, fontweight='bold')
            ax.set_xlabel('Asset Index')
            ax.set_ylabel('Weight')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, w in zip(bars, weights):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{w:.2f}', ha='center', va='bottom' if height > 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'weight_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def plot_performance_radar(self):
        """Radar chart comparing multiple metrics"""
        if len(self.results) < 2:
            print("Need at least 2 optimizers for radar chart")
            return
            
        # Prepare data
        metrics = ['Sharpe Ratio', 'Total Return', 'Profit Factor', 'Win Rate', '-Max Drawdown']
        
        # Normalize data
        all_values = []
        for result in self.results.values():
            values = [
                result['metrics'].get('Sharpe Ratio', 0),
                result['metrics'].get('Total Return', 0),
                result['metrics'].get('Profit Factor', 0),
                result['metrics'].get('Win Rate', 0),
                -result['metrics'].get('Max Drawdown', 0)
            ]
            all_values.append(values)
        
        all_values = np.array(all_values)
        
        # Normalize to [0, 1]
        min_vals = all_values.min(axis=0)
        max_vals = all_values.max(axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1  # Avoid division by zero
        
        data = {}
        for i, (name, result) in enumerate(self.results.items()):
            normalized = (all_values[i] - min_vals) / ranges
            data[name] = normalized
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
        
        for (name, values), color in zip(data.items(), colors[:len(data)]):
            values = values.tolist()
            values += values[:1]  # Close the loop
            ax.plot(angles, values, 'o-', linewidth=2, label=name.upper(), color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title('Performance Metrics Comparison (Normalized)', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'radar_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def create_summary_table(self):
        """Create formatted summary table"""
        fig, ax = plt.subplots(figsize=(12, max(3, len(self.results) * 0.5)))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        headers = ['Optimizer', 'Sharpe Ratio', 'Total Return', 'Profit Factor', 'Win Rate', 'Max Drawdown']
        table_data = []
        
        for name, result in self.results.items():
            metrics = result['metrics']
            row = [
                name.upper(),
                f"{metrics.get('Sharpe Ratio', 0):.3f}",
                f"{metrics.get('Total Return', 0):.2%}",
                f"{metrics.get('Profit Factor', 0):.2f}",
                f"{metrics.get('Win Rate', 0):.2%}",
                f"{metrics.get('Max Drawdown', 0):.2%}"
            ]
            table_data.append(row)
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        colWidths=[0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Optimization Methods Comparison Summary', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'summary_table.png', dpi=150, bbox_inches='tight')
        plt.close()