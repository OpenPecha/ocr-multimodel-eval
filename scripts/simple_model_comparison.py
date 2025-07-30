import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def create_model_comparison_chart():
    """Create bar charts comparing all models' CER performance for each image separately."""
    
    # Load all CSV files from results directory
    results_dir = Path("../results")
    csv_files = list(results_dir.glob("*.csv"))
    
    if not csv_files:
        print("❌ No CSV files found in results directory!")
        return
    
    # Dictionary to store all model data: {image: {model: cer}}
    image_data = {}
    
    # Extract CER for each image and model
    for csv_file in csv_files:
        model_name = csv_file.stem
        try:
            df = pd.read_csv(csv_file)
            # Get only regular images (not Combined)
            regular_images = df[df['Image'].str.startswith('Image')]
            
            for _, row in regular_images.iterrows():
                image = row['Image']
                cer = row['CER']
                
                if image not in image_data:
                    image_data[image] = {}
                image_data[image][model_name] = cer
            
            print(f"✅ Loaded {model_name}: {len(regular_images)} images")
        except Exception as e:
            print(f"❌ Error loading {csv_file}: {e}")
    
    if not image_data:
        print("❌ No valid data found!")
        return
    
    # Create a bar chart for each image
    images = sorted(image_data.keys(), key=lambda x: int(x.split()[-1]))  # Sort by image number
    
    for image in images:
        create_single_image_chart(image, image_data[image])

def create_single_image_chart(image_name, model_cer_data):
    """Create a single bar chart for one image comparing all models."""
    
    # Sort models by performance (lower CER is better)
    sorted_models = dict(sorted(model_cer_data.items(), key=lambda x: x[1]))
    
    # Create the bar chart
    plt.figure(figsize=(12, 8))
    
    models = list(sorted_models.keys())
    cer_values = list(sorted_models.values())
    
    # Create color gradient (green for best, red for worst)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(models)))
    
    bars = plt.bar(models, cer_values, color=colors, edgecolor='black', linewidth=1)
    
    # Customize the chart
    plt.title(f'Model Performance for {image_name}\n(Lower CER is Better)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Character Error Rate (CER)', fontsize=12, fontweight='bold')
    plt.xlabel('Models', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, cer_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Highlight best performer
    bars[0].set_edgecolor('gold')
    bars[0].set_linewidth(4)
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = Path("../data/bar_chart")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the chart
    safe_image_name = image_name.replace(' ', '_').lower()
    output_path = output_dir / f"{safe_image_name}_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 {image_name} chart saved to: {output_path}")
    
    # Show the chart
    plt.show()
    
    # Print ranking for this image
    print(f"\n🏆 {image_name.upper()} RANKING:")
    print("=" * 50)
    for i, (model, cer) in enumerate(sorted_models.items(), 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
        print(f"{emoji} #{i:2d} {model:25s} | CER: {cer:.4f}")
    print()


if __name__ == "__main__":
    create_model_comparison_chart()
