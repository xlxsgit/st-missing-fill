from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


import numpy as np

def _set_plot_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

def plot_baseline_comparison(
    summary_df: pd.DataFrame,
    output_dir: str | Path,
    run_id: str,
    annotate: bool = True,
    dpi: int = 350,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_plot_style()

    df = summary_df.copy()
    if df.empty:
        return {}
        
    df["pattern"] = df["pattern"].str.upper()
    df["pi"] = df["pi"].astype(float)
    
    available_splits = [s for s in ["train", "val", "test"] if s in df.columns and not df[s].isna().all()]
    patterns_order = ["MCAR", "SEQ", "SCM"]
    
    models = sorted(df["model"].unique().tolist())
    palette = dict(zip(models, sns.color_palette("Set2", n_colors=max(len(models), 1))))

    out = {}

    for split in available_splits:
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 8), constrained_layout=True)
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
            
        for ax, pat in zip(axes, patterns_order):
            pat_df = df[(df["pattern"] == pat) & (df[split].notna())].copy()
            if pat_df.empty:
                ax.axis("off")
                continue
                
            pis = sorted(pat_df["pi"].unique())
            x_positions = np.arange(len(pis))
            
            bar_width = 0.8 / len(models)
            
            for i, pi_val in enumerate(pis):
                pi_df = pat_df[pat_df["pi"] == pi_val].sort_values(by=split, ascending=True)
                
                for j, (_, row) in enumerate(pi_df.iterrows()):
                    model_name = row["model"]
                    rmse_val = row[split]
                    offset = (j - len(pi_df) / 2 + 0.5) * bar_width
                    ax.bar(
                        i + offset, 
                        rmse_val, 
                        width=bar_width, 
                        color=palette[model_name], 
                        edgecolor="white",
                        label=model_name if i == 0 else "" # Only add to legend once per model position inside a group
                    )
                    
                    if annotate:
                        ax.text(
                            i + offset, 
                            rmse_val + (rmse_val * 0.01), 
                            f"{rmse_val:.4f}", 
                            ha='center', 
                            va='bottom', 
                            rotation=90, 
                            fontsize=9
                        )
            
            ax.set_xticks(x_positions)
            ax.set_xticklabels([f"PI={p:g}" for p in pis])
            ax.set_ylabel(f"RMSE ({pat})", fontweight="bold")
            
            # Dynamic Y-axis limit to prevent label overflow
            y_max_val = pi_df[split].max() if not pi_df.empty else 1.0
            for i, pi_val in enumerate(pis):
                max_val = pat_df[pat_df["pi"] == pi_val][split].max()
                if not np.isnan(max_val) and max_val > y_max_val:
                    y_max_val = max_val
            ax.set_ylim(0, y_max_val * 1.15)
            
            # Only place single legend on the first ax
            if ax == axes[0]:
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                if by_label:
                    ax.legend(by_label.values(), by_label.keys(), title="Models", bbox_to_anchor=(1.01, 1), loc="upper left")
                
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
                
        file_path = output_dir / f"{split}.png"
        fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        out[f"bar_{split}"] = file_path

    return out
