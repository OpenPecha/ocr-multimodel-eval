import os
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns


class ModelPerformanceLeaderboard:
    """Generate and display OCR model performance leaderboards."""
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize the leaderboard generator.
        
        Args:
            results_dir (str): Directory containing CSV result files
        """
        self.results_dir = Path(results_dir)
        self.model_data = {}
        self.leaderboard_df = None
        
    def load_model_results(self) -> Dict[str, Dict[str, float]]:
        """
        Load CER results from all CSV files in the results directory.
        
        Returns:
            Dict[str, Dict[str, float]]: Model name -> {image: CER, combined: CER}
        """
        print("🔍 Loading model results from CSV files...")
        
        csv_files = list(self.results_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.results_dir}")
        
        for csv_file in csv_files:
            model_name = csv_file.stem  # Get filename without extension
            print(f"  📊 Loading {model_name}...")
            
            try:
                df = pd.read_csv(csv_file)
                model_results = {}
                
                # Extract CER values for each image
                for _, row in df.iterrows():
                    image_key = row['Image']
                    cer_value = float(row['CER (decimal)'])
                    model_results[image_key] = cer_value
                
                self.model_data[model_name] = model_results
                print(f"    ✅ Loaded {len(model_results)} entries for {model_name}")
                
            except Exception as e:
                print(f"    ❌ Error loading {csv_file}: {e}")
                continue
        
        print(f"📈 Successfully loaded data for {len(self.model_data)} models\n")
        return self.model_data
    
    def create_leaderboard_dataframe(self) -> pd.DataFrame:
        """
        Create a comprehensive leaderboard DataFrame.
        
        Returns:
            pd.DataFrame: Leaderboard with models as columns and images as rows
        """
        if not self.model_data:
            self.load_model_results()
        
        # Get all unique images across all models
        all_images = set()
        for model_results in self.model_data.values():
            all_images.update(model_results.keys())
        
        # Sort images (separate regular images from combined)
        regular_images = sorted([img for img in all_images if img.startswith('Image')])
        combined_images = sorted([img for img in all_images if img.startswith('Combined')])
        sorted_images = regular_images + combined_images
        
        # Create DataFrame
        leaderboard_data = {}
        for model_name, results in self.model_data.items():
            leaderboard_data[model_name] = [
                results.get(image, np.nan) for image in sorted_images
            ]
        
        self.leaderboard_df = pd.DataFrame(
            leaderboard_data, 
            index=sorted_images
        )
        
        return self.leaderboard_df
    
    def display_performance_table(self) -> None:
        """Display a formatted performance comparison table."""
        if self.leaderboard_df is None:
            self.create_leaderboard_dataframe()
        
        print("🏆 MODEL PERFORMANCE LEADERBOARD")
        print("=" * 80)
        print("📊 Character Error Rate (CER) Comparison - Lower is Better")
        print("=" * 80)
        
        # Format the DataFrame for display
        display_df = self.leaderboard_df.round(4)
        
        # Create table with tabulate
        table = tabulate(
            display_df, 
            headers=display_df.columns,
            tablefmt="grid",
            floatfmt=".4f",
            showindex=True
        )
        
        print(table)
        print()
    
    def display_model_rankings(self) -> None:
        """Display overall model rankings based on average performance."""
        if self.leaderboard_df is None:
            self.create_leaderboard_dataframe()
        
        print("🥇 OVERALL MODEL RANKINGS")
        print("=" * 50)
        
        # Calculate average CER for each model (excluding Combined CER)
        regular_images_mask = self.leaderboard_df.index.str.startswith('Image')
        regular_images_df = self.leaderboard_df[regular_images_mask]
        
        model_averages = regular_images_df.mean().sort_values()
        
        print("📈 Average CER across all images (lower is better):")
        print("-" * 50)
        
        for rank, (model, avg_cer) in enumerate(model_averages.items(), 1):
            emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "📊"
            print(f"{emoji} #{rank:2d} {model:25s} | Avg CER: {avg_cer:.4f}")
        
        print()
    
    def display_per_image_winners(self) -> None:
        """Display the best performing model for each image."""
        if self.leaderboard_df is None:
            self.create_leaderboard_dataframe()
        
        print("🎯 BEST PERFORMER PER IMAGE")
        print("=" * 50)
        
        regular_images_mask = self.leaderboard_df.index.str.startswith('Image')
        regular_images_df = self.leaderboard_df[regular_images_mask]
        
        for image in regular_images_df.index:
            image_results = regular_images_df.loc[image].dropna()
            if not image_results.empty:
                best_model = image_results.idxmin()
                best_cer = image_results.min()
                worst_model = image_results.idxmax()
                worst_cer = image_results.max()
                
                print(f"🖼️  {image}:")
                print(f"   🏆 Best:  {best_model:20s} (CER: {best_cer:.4f})")
                print(f"   📉 Worst: {worst_model:20s} (CER: {worst_cer:.4f})")
                print()
    
    def display_statistics_summary(self) -> None:
        """Display statistical summary of model performance."""
        if self.leaderboard_df is None:
            self.create_leaderboard_dataframe()
        
        print("📊 STATISTICAL SUMMARY")
        print("=" * 50)
        
        regular_images_mask = self.leaderboard_df.index.str.startswith('Image')
        regular_images_df = self.leaderboard_df[regular_images_mask]
        
        stats_df = regular_images_df.describe().round(4)
        
        print("📈 Performance Statistics (CER values):")
        print("-" * 50)
        print(tabulate(stats_df, headers=stats_df.columns, tablefmt="grid", floatfmt=".4f"))
        print()
    
    def save_leaderboard_report(self, output_file: str = "model_performance_report.txt") -> None:
        """
        Save the complete leaderboard report to a text file.
        
        Args:
            output_file (str): Output filename for the report
        """
        import sys
        from io import StringIO
        
        # Capture all print output
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            self.display_performance_table()
            self.display_model_rankings()
            self.display_per_image_winners()
            self.display_statistics_summary()
        finally:
            sys.stdout = old_stdout
        
        # Write to file
        report_content = captured_output.getvalue()
        output_path = Path(output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("MODEL PERFORMANCE LEADERBOARD REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated by: {__file__}\n")
            f.write(f"Data source: {self.results_dir}\n")
            f.write("=" * 80 + "\n\n")
            f.write(report_content)
        
        print(f"📄 Report saved to: {output_path.absolute()}")
    
    def create_visualization(self, save_plot: bool = True) -> None:
        """
        Create comprehensive visualizations of model performance.
        
        Args:
            save_plot (bool): Whether to save the plots as images
        """
        if self.leaderboard_df is None:
            self.create_leaderboard_dataframe()
        
        # Create output directory for visualizations
        viz_dir = Path("visualizations")
        viz_dir.mkdir(exist_ok=True)
        
        # Filter to regular images only for visualization
        regular_images_mask = self.leaderboard_df.index.str.startswith('Image')
        plot_df = self.leaderboard_df[regular_images_mask]
        
        # 1. Create overall heatmap
        self._create_heatmap(plot_df, viz_dir, save_plot)
        
        # 2. Create per-image bar charts
        self._create_per_image_bar_charts(plot_df, viz_dir, save_plot)
        
        # 3. Create overall comparison bar chart
        self._create_overall_comparison_chart(plot_df, viz_dir, save_plot)
    
    def _create_heatmap(self, plot_df: pd.DataFrame, viz_dir: Path, save_plot: bool) -> None:
        """Create and save heatmap visualization."""
        plt.figure(figsize=(14, 8))
        
        # Create heatmap
        sns.heatmap(
            plot_df.T,  # Transpose so models are on Y-axis
            annot=True,
            fmt='.3f',
            cmap='RdYlGn_r',  # Red-Yellow-Green reversed (lower is better)
            cbar_kws={'label': 'Character Error Rate (CER)'},
            linewidths=0.5
        )
        
        plt.title('OCR Model Performance Heatmap\n(Lower CER values are better)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Images', fontsize=12, fontweight='bold')
        plt.ylabel('Models', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_plot:
            plot_path = viz_dir / "model_performance_heatmap.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Heatmap saved to: {plot_path}")
        
        plt.show()
        plt.close()
    
    def _create_per_image_bar_charts(self, plot_df: pd.DataFrame, viz_dir: Path, save_plot: bool) -> None:
        """Create individual bar charts for each image."""
        print("🎯 Creating per-image bar charts...")
        
        # Create subdirectory for per-image charts
        per_image_dir = viz_dir / "per_image_charts"
        per_image_dir.mkdir(exist_ok=True)
        
        # Calculate number of rows and columns for subplots
        n_images = len(plot_df.index)
        n_cols = 3  # 3 charts per row
        n_rows = (n_images + n_cols - 1) // n_cols
        
        # Create a large figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
        fig.suptitle('Model Performance by Image\n(Lower CER is Better)', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # Flatten axes array for easier indexing
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Create bar chart for each image
        for idx, image in enumerate(plot_df.index):
            ax = axes[idx]
            image_data = plot_df.loc[image].dropna().sort_values()
            
            # Create color map (green for best, red for worst)
            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(image_data)))
            
            bars = ax.bar(range(len(image_data)), image_data.values, color=colors)
            
            # Customize the chart
            ax.set_title(f'{image}', fontsize=14, fontweight='bold', pad=10)
            ax.set_ylabel('CER (Lower is Better)', fontsize=10)
            ax.set_xticks(range(len(image_data)))
            ax.set_xticklabels(image_data.index, rotation=45, ha='right', fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, image_data.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Highlight best performer
            best_idx = 0  # Already sorted, so first is best
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(3)
        
        # Hide empty subplots
        for idx in range(len(plot_df.index), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = per_image_dir / "all_images_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Per-image charts saved to: {plot_path}")
        
        plt.show()
        plt.close()
        
        # Also create individual charts for each image
        for image in plot_df.index:
            self._create_single_image_chart(image, plot_df.loc[image], per_image_dir, save_plot)
    
    def _create_single_image_chart(self, image: str, image_data: pd.Series, 
                                  output_dir: Path, save_plot: bool) -> None:
        """Create individual bar chart for a single image."""
        image_data_clean = image_data.dropna().sort_values()
        
        plt.figure(figsize=(10, 6))
        
        # Create color gradient
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(image_data_clean)))
        
        bars = plt.bar(range(len(image_data_clean)), image_data_clean.values, color=colors)
        
        # Customize the chart
        plt.title(f'Model Performance - {image}\n(Lower CER is Better)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Character Error Rate (CER)', fontsize=12, fontweight='bold')
        plt.xlabel('Models', fontsize=12, fontweight='bold')
        plt.xticks(range(len(image_data_clean)), image_data_clean.index, 
                  rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, image_data_clean.values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Highlight best performer
        bars[0].set_edgecolor('gold')
        bars[0].set_linewidth(4)
        
        # Add best performer annotation
        best_model = image_data_clean.index[0]
        best_cer = image_data_clean.iloc[0]
        plt.annotate(f'🏆 Best: {best_model}\nCER: {best_cer:.3f}', 
                    xy=(0, best_cer), xytext=(0.5, best_cer + 0.1),
                    ha='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='gold', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='gold', lw=2))
        
        plt.tight_layout()
        
        if save_plot:
            safe_image_name = image.replace(' ', '_').lower()
            plot_path = output_dir / f"{safe_image_name}_performance.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        plt.close()  # Don't show individual charts, just save them
    
    def _create_overall_comparison_chart(self, plot_df: pd.DataFrame, viz_dir: Path, save_plot: bool) -> None:
        """Create overall model comparison bar chart."""
        # Calculate average CER for each model
        model_averages = plot_df.mean().sort_values()
        
        plt.figure(figsize=(12, 8))
        
        # Create color gradient
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(model_averages)))
        
        bars = plt.bar(range(len(model_averages)), model_averages.values, color=colors)
        
        # Customize the chart
        plt.title('Overall Model Performance Ranking\n(Average CER across all images)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Average Character Error Rate (CER)', fontsize=12, fontweight='bold')
        plt.xlabel('Models (Ranked by Performance)', fontsize=12, fontweight='bold')
        plt.xticks(range(len(model_averages)), model_averages.index, 
                  rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, model_averages.values)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Add ranking labels
            rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📊"
            plt.text(bar.get_x() + bar.get_width()/2., -0.05,
                    f'{rank_emoji} #{i+1}', ha='center', va='top', 
                    fontsize=12, fontweight='bold')
        
        # Highlight top 3 performers
        for i in range(min(3, len(bars))):
            bars[i].set_edgecolor(['gold', 'silver', '#CD7F32'][i])  # Gold, Silver, Bronze
            bars[i].set_linewidth(3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = viz_dir / "overall_model_ranking.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Overall ranking chart saved to: {plot_path}")
        
        plt.show()
        plt.close()
    
    def run_complete_analysis(self) -> None:
        """Run the complete leaderboard analysis and display all results."""
        print("🚀 Starting Model Performance Analysis...")
        print("=" * 80)
        
        try:
            # Load data and create leaderboard
            self.load_model_results()
            self.create_leaderboard_dataframe()
            
            # Display all analysis components
            self.display_performance_table()
            self.display_model_rankings()
            self.display_per_image_winners()
            self.display_statistics_summary()
            
            # Save report
            self.save_leaderboard_report()
            
            # Create visualization
            print("🎨 Creating performance visualization...")
            self.create_visualization()
            
            print("✅ Analysis complete!")
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            raise


def main():
    """Main function to run the leaderboard analysis."""
    # Initialize leaderboard generator
    leaderboard = ModelPerformanceLeaderboard(results_dir="results")
    
    # Run complete analysis
    leaderboard.run_complete_analysis()


if __name__ == "__main__":
    main()
