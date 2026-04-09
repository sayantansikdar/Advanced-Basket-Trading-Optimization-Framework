"""
Comprehensive visualization for any number of optimizers
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

class OptimizationVisualizer:
    """Create comparison visualizations for any number of optimizers"""
    
    def __init__(self, results, output_dir='results/plots'):
        # Filter out None results
        self.results = {k: v for k, v in results.items() if v is not None}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"VISUALIZER INITIALIZED")
        print(f"{'='*60}")
        print(f"Number of optimizers: {len(self.results)}")
        for name in self.results.keys():
            print(f"  - {name.upper()}: Sharpe={self.results[name]['metrics'].get('Sharpe Ratio', 0):.3f}")
        
        # Dynamic color palette
        self.color_palette = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12', 
                              '#1abc9c', '#e67e22', '#34495e', '#95a5a6', '#16a085']
        self.colors = {name: self.color_palette[i % len(self.color_palette)] 
                       for i, name in enumerate(self.results.keys())}
        
        # Optimizer name mapping for display
        self.name_mapping = {
            'bayesian': 'BO',
            'cmaes': 'CMA-ES',
            'turbo': 'TuRBO',
            'cvfs_cmaes': 'CVFS-CMA-ES',
            'turbo_tuned': 'TuRBO-Tuned',
            'saasbo': 'SAASBO'
        }
        
    def get_display_name(self, name):
        """Get display name for optimizer"""
        return self.name_mapping.get(name.lower(), name.upper())
    
    def create_all_plots(self):
        """Generate all comparison plots"""
        if not self.results:
            print("No valid results to visualize")
            return
        
        print(f"\n{'='*60}")
        print(f"GENERATING VISUALIZATIONS")
        print(f"{'='*60}")
        
        self.plot_metric_comparison()
        self.plot_weight_distributions()
        self.plot_performance_radar()
        self.create_summary_table()
        self.plot_ranking_heatmap()
        self.plot_parallel_coordinates()
        
        print(f"\n✓ All plots saved to {self.output_dir}")
    
    def plot_metric_comparison(self):
        """Bar chart comparing key metrics"""
        metrics = ['Sharpe Ratio', 'Total Return (%)', 'Profit Factor', 'Win Rate (%)']
        optimizer_names = list(self.results.keys())
        n_optimizers = len(optimizer_names)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = []
            
            for name in optimizer_names:
                metric_name = metric.replace(' (%)', '').replace(' (%)', '')
                val = self.results[name]['metrics'].get(metric_name, 0)
                if 'Return' in metric:
                    val = val * 100
                elif 'Win Rate' in metric:
                    val = val * 100
                values.append(val)
            
            # Create bars
            x_pos = np.arange(n_optimizers)
            bars = ax.bar(x_pos, values, color=[self.colors[n] for n in optimizer_names], alpha=0.7, edgecolor='black')
            
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.set_ylabel(metric)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([self.get_display_name(n) for n in optimizer_names], fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                offset = max(values) * 0.02 if max(values) > 0 else 0.5
                ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metric_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Metric comparison saved")
    
    def plot_weight_distributions(self):
        """Compare weight distributions across optimizers"""
        n_optimizers = len(self.results)
        fig, axes = plt.subplots(1, n_optimizers, figsize=(5*n_optimizers, 5))
        
        if n_optimizers == 1:
            axes = [axes]
        
        for idx, (name, result) in enumerate(self.results.items()):
            weights = result['weights']
            ax = axes[idx]
            
            colors = ['green' if w > 0 else 'red' for w in weights]
            bars = ax.bar(range(len(weights)), weights, color=colors, alpha=0.7, edgecolor='black')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_title(f'{self.get_display_name(name)} - Asset Weights', fontsize=12, fontweight='bold')
            ax.set_xlabel('Asset Index')
            ax.set_ylabel('Weight')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, w in zip(bars, weights):
                height = bar.get_height()
                offset = 0.1 if height >= 0 else -0.2
                ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                       f'{w:.2f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'weight_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Weight distributions saved")
    
    def plot_performance_radar(self):
        """Radar chart comparing multiple metrics"""
        if len(self.results) < 2:
            print("Need at least 2 optimizers for radar chart")
            return
        
        metrics = ['Sharpe Ratio', 'Total Return', 'Profit Factor', 'Win Rate', '-Max Drawdown']
        optimizer_names = list(self.results.keys())
        n_optimizers = len(optimizer_names)
        
        print(f"\n  Creating radar chart for {n_optimizers} optimizers:")
        
        # Collect all values
        all_values = []
        for name in optimizer_names:
            result = self.results[name]
            values = [
                result['metrics'].get('Sharpe Ratio', 0),
                result['metrics'].get('Total Return', 0),
                result['metrics'].get('Profit Factor', 0),
                result['metrics'].get('Win Rate', 0),
                -result['metrics'].get('Max Drawdown', 0)
            ]
            all_values.append(values)
            print(f"    {name.upper()}: Sharpe={values[0]:.3f}, Return={values[1]:.2%}")
        
        all_values = np.array(all_values)
        
        # Normalize to [0, 1]
        min_vals = all_values.min(axis=0)
        max_vals = all_values.max(axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for i, name in enumerate(optimizer_names):
            values = (all_values[i] - min_vals) / ranges
            values = values.tolist()
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=self.get_display_name(name), 
                    color=self.colors[name])
            ax.fill(angles, values, alpha=0.15, color=self.colors[name])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title(f'Performance Metrics Comparison - {n_optimizers} Optimizers', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'radar_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Radar chart saved")
    
    def create_summary_table(self):
        """Create formatted summary table with ranking"""
        fig_height = max(4, len(self.results) * 0.5 + 1)
        fig, ax = plt.subplots(figsize=(12, fig_height))
        ax.axis('tight')
        ax.axis('off')
        
        headers = ['Rank', 'Optimizer', 'Sharpe Ratio', 'Total Return', 'Profit Factor', 'Win Rate', 'Max Drawdown']
        table_data = []
        
        # Prepare data and sort by Sharpe ratio
        data_list = []
        for name, result in self.results.items():
            metrics = result['metrics']
            data_list.append({
                'name': self.get_display_name(name),
                'sharpe': metrics.get('Sharpe Ratio', 0),
                'return': metrics.get('Total Return', 0),
                'profit': metrics.get('Profit Factor', 0),
                'winrate': metrics.get('Win Rate', 0),
                'drawdown': metrics.get('Max Drawdown', 0)
            })
        
        # Sort by Sharpe ratio descending
        data_list.sort(key=lambda x: x['sharpe'], reverse=True)
        
        for rank, d in enumerate(data_list, 1):
            medal = {1: '🥇', 2: '🥈', 3: '🥉'}.get(rank, f'{rank}')
            row = [
                medal,
                d['name'],
                f"{d['sharpe']:.3f}",
                f"{d['return']:.2%}",
                f"{d['profit']:.2f}",
                f"{d['winrate']:.2%}",
                f"{d['drawdown']:.2%}"
            ]
            table_data.append(row)
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        colWidths=[0.06, 0.15, 0.12, 0.12, 0.12, 0.12, 0.12])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style header row
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Highlight top performer
        for i in range(1, len(table_data) + 1):
            if i == 1:
                for j in range(len(headers)):
                    table[(i, j)].set_facecolor('#d4edda')
        
        plt.title(f'Optimization Methods Comparison Summary', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'summary_table.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Summary table saved")
    
    def plot_ranking_heatmap(self):
        """Create ranking heatmap for all metrics"""
        optimizer_names = list(self.results.keys())
        metrics = ['Sharpe Ratio', 'Total Return', 'Profit Factor', 'Win Rate', 'Max Drawdown']
        
        # Create ranking matrix
        ranking_matrix = []
        for metric in metrics:
            metric_values = []
            for name in optimizer_names:
                val = self.results[name]['metrics'].get(metric, 0)
                if metric == 'Max Drawdown':
                    val = -val
                metric_values.append(val)
            
            ranks = len(optimizer_names) - np.argsort(np.argsort(metric_values)) + 1
            ranking_matrix.append(ranks)
        
        ranking_matrix = np.array(ranking_matrix)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(ranking_matrix, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=len(optimizer_names))
        
        ax.set_xticks(range(len(optimizer_names)))
        ax.set_yticks(range(len(metrics)))
        ax.set_xticklabels([self.get_display_name(n) for n in optimizer_names], fontsize=10)
        ax.set_yticklabels(metrics)
        
        for i in range(len(metrics)):
            for j in range(len(optimizer_names)):
                ax.text(j, i, f'{int(ranking_matrix[i, j])}', ha="center", va="center", 
                       color="black", fontsize=11, fontweight='bold')
        
        ax.set_title(f'Ranking Heatmap (1 = Best)', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, ticks=range(1, len(optimizer_names)+1))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ranking_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Ranking heatmap saved")
    
    def plot_parallel_coordinates(self):
        """Parallel coordinates plot"""
        if len(self.results) < 2:
            return
        
        optimizer_names = list(self.results.keys())
        metrics = ['Sharpe Ratio', 'Total Return', 'Profit Factor', 'Win Rate']
        
        normalized_data = []
        for name in optimizer_names:
            values = [
                self.results[name]['metrics'].get('Sharpe Ratio', 0),
                self.results[name]['metrics'].get('Total Return', 0),
                self.results[name]['metrics'].get('Profit Factor', 0),
                self.results[name]['metrics'].get('Win Rate', 0)
            ]
            normalized_data.append(values)
        
        normalized_data = np.array(normalized_data)
        min_vals = normalized_data.min(axis=0)
        max_vals = normalized_data.max(axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1
        normalized_data = (normalized_data - min_vals) / ranges
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = range(len(metrics))
        for i, name in enumerate(optimizer_names):
            ax.plot(x_pos, normalized_data[i], 'o-', linewidth=2, 
                   label=self.get_display_name(name), color=self.colors[name], markersize=8)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics)
        ax.set_ylabel('Normalized Score (0-1)')
        ax.set_title('Parallel Coordinates - Performance Comparison', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parallel_coordinates.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Parallel coordinates saved")


# For direct execution
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    
    # Load results from CSV
    results_df = pd.read_csv('results/comparison_summary.csv')
    
    # Create results dictionary
    results = {}
    for _, row in results_df.iterrows():
        name = row['Optimizer'].lower()
        results[name] = {
            'weights': np.array([0, 0, 0]),  # Placeholder
            'metrics': {
                'Sharpe Ratio': row['Sharpe Ratio'],
                'Total Return': row['Total Return'],
                'Max Drawdown': row['Max Drawdown'],
                'Profit Factor': row['Profit Factor'],
                'Win Rate': row['Win Rate']
            }
        }
    
    # Create visualizer
    visualizer = OptimizationVisualizer(results)
    visualizer.create_all_plots()
    
    print("\n✓ All visualizations regenerated successfully!")