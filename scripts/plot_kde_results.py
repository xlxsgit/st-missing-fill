import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_kde():
    csv_path = Path("logs/xlx/summary.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Model name mapping
    name_map = {
        'mymodel': 'STAT-Net',
        'itransformer': 'iTransformer'
    }
    
    def map_name(name):
        n_lower = name.lower()
        if n_lower in name_map:
            return name_map[n_lower]
        return name.upper()

    df['Model'] = df['模型'].apply(map_name)
    df['RMSE'] = df['test rmse']

    # Plot Style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })

    plt.figure(figsize=(12, 7))
    
    # 1. Generate palette based on alphabetical order (matches baseline_compare.py mapping)
    all_models = sorted(df['Model'].unique())
    palette = dict(zip(all_models, sns.color_palette("Set2", n_colors=max(len(all_models), 1))))
    
    # 2. Calculate mean RMSE to sort models by performance (matches test.png legend order)
    model_order = df.groupby('Model')['RMSE'].mean().sort_values().index.tolist()
    
    ax = sns.kdeplot(
        data=df, 
        x="RMSE", 
        hue="Model", 
        fill=True, 
        common_norm=False, 
        palette=palette, 
        alpha=.5, 
        linewidth=2,
        hue_order=model_order
    )

    plt.xlabel("Test RMSE")
    plt.ylabel("Density")
    
    # Move legend outside
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1), title="Model")
    
    output_path = Path("logs/xlx/kde_rmse.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"KDE plot saved to {output_path}")

if __name__ == "__main__":
    plot_kde()
