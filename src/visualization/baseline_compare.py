from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _load_and_prepare(df: pd.DataFrame):
    required_cols = {"model", "pattern", "pi", "train", "val", "test"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"run dataframe missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError(f"run dataframe is empty")

    df["pattern"] = df["pattern"].str.upper()
    df["pi"] = df["pi"].astype(float)
    pattern_order = [p for p in ["MCAR", "SEQ", "SCM"] if p in set(df["pattern"])]
    pi_order = sorted(df["pi"].unique().tolist())
    combo_order = [f"{p}\npi={pi:g}" for p in pattern_order for pi in pi_order]
    df["combo"] = [f"{p}\npi={pi:g}" for p, pi in zip(df["pattern"], df["pi"])]

    # 1. Calculate the rank of each model per missingness setting
    df["train_rank"] = df.groupby("combo")["train"].rank(method="min", ascending=True)
    df["val_rank"] = df.groupby("combo")["val"].rank(method="min", ascending=True)
    df["test_rank"] = df.groupby("combo")["test"].rank(method="min", ascending=True)

    # 2. Sort the global model order by their Average Test Rank instead of Average RMSE
    model_order = (
        df.groupby("model", as_index=True)["test_rank"].mean().sort_values(ascending=True).index.tolist()
    )
    return df, combo_order, model_order


def _set_plot_style():
    sns.set_theme(style="white")
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
    """Generate concise, publication-style comparison figures directly inside a run_dir.

    Output:
    - One heatmap per split: train / val / test
    - One test-average ranking bar chart
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, combo_order, model_order = _load_and_prepare(summary_df.copy())
    _set_plot_style()

    split_cols = ["train", "val", "test"]
    split_titles = ["Train RMSE", "Validation RMSE", "Test RMSE"]
    
    num_models = len(model_order)
    # Use discrete ranked colors (light to dark indicates 1st to Nth)
    cmap = mcolors.ListedColormap(sns.color_palette("YlGnBu", n_colors=num_models))

    out = {}

    # 1) three separate heatmaps coloring by rank
    for split, title in zip(split_cols, split_titles):
        mat_rmse = (
            df.pivot_table(index="model", columns="combo", values=split, aggfunc="first")
            .reindex(index=model_order, columns=combo_order)
        )
        mat_rank = (
            df.pivot_table(index="model", columns="combo", values=f"{split}_rank", aggfunc="first")
            .reindex(index=model_order, columns=combo_order)
        )
        
        fig, ax = plt.subplots(figsize=(10.8, 6.4), constrained_layout=True)
        sns.heatmap(
            mat_rank, 
            annot=mat_rmse if annotate else False,
            fmt=".4f",
            ax=ax,
            cmap=cmap,
            vmin=1,
            vmax=num_models,
            linewidths=0.6,
            linecolor="#EFEFEF",
            cbar_kws={"shrink": 0.85, "label": "Model Rank (Darker=Worse)", "ticks": range(1, num_models + 1)},
        )
        ax.set_title(f"{title} (Colored by Rank)", fontweight="bold", pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("Model")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        file_path = output_dir / f"baseline_heatmap_{split}_{run_id}.png"
        fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        out[f"heatmap_{split}"] = file_path

    # 2) test-average ranking chart.
    rank_df = (
        df.groupby("model", as_index=False)["test_rank"].mean().sort_values("test_rank", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(8.8, 5.6), constrained_layout=True)
    sns.barplot(data=rank_df, x="test_rank", y="model", ax=ax, color="#4C78A8")
    ax.set_title("Model Ranking by Mean Test Rank", fontweight="bold", pad=10)
    ax.set_xlabel("Mean Test Rank (Lower is Better)")
    ax.set_ylabel("Model")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, v in enumerate(rank_df["test_rank"]):
        ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=9)

    rank_path = output_dir / f"baseline_rank_test_{run_id}.png"
    fig.savefig(rank_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    out["rank_test"] = rank_path

    return out
