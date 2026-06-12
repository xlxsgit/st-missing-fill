import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.visualization.baseline_compare import plot_baseline_comparison

def update_all():
    # 1. Load Data
    csv_path = Path("logs/xlx/summary.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Standardize column mapping for the visualization function
    # Expected columns: model, pattern, pi, train, val, test
    viz_df = pd.DataFrame()
    viz_df["model"] = df["模型"]
    viz_df["pattern"] = df["缺失模式"]
    viz_df["pi"] = df["缺失率"]
    viz_df["train"] = df["train rmse"]
    viz_df["val"] = df["val rmse"]
    viz_df["test"] = df["test rmse"]

    # 2. Update Result Visualizations (train.png, test.png etc in logs/xlx)
    output_dir = Path("logs/xlx")
    print("Regenerating baseline comparison plots...")
    plot_baseline_comparison(
        summary_df=viz_df,
        output_dir=output_dir,
        run_id="xlx_update",
        annotate=False
    )
    print(f"Visualizations updated in {output_dir}")

if __name__ == "__main__":
    update_all()
