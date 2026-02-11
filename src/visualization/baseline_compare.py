from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _load_and_prepare(summary_csv: Path, run_id: str | None):
    if not summary_csv.exists():
        raise FileNotFoundError(summary_csv)

    df = pd.read_csv(summary_csv)
    required_cols = {"run_id", "model", "pattern", "pi", "train", "val", "test"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"summary missing required columns: {sorted(missing)}")

    if run_id is None:
        run_id = str(df["run_id"].iloc[-1])
    df = df[df["run_id"] == run_id].copy()
    if df.empty:
        raise ValueError(f"run_id={run_id} not found in {summary_csv}")

    df["pattern"] = df["pattern"].str.upper()
    df["pi"] = df["pi"].astype(float)
    pattern_order = [p for p in ["MCAR", "SEQ", "SCM"] if p in set(df["pattern"])]
    pi_order = sorted(df["pi"].unique().tolist())
    combo_order = [f"{p}\npi={pi:g}" for p in pattern_order for pi in pi_order]
    df["combo"] = [f"{p}\npi={pi:g}" for p, pi in zip(df["pattern"], df["pi"])]

    model_order = (
        df.groupby("model", as_index=True)["test"].mean().sort_values(ascending=True).index.tolist()
    )
    return df, run_id, combo_order, model_order


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
    summary_csv: str | Path = "logs/summary.csv",
    output_dir: str | Path = "data/figs/baseline_compare",
    run_id: str | None = None,
    annotate: bool = True,
    dpi: int = 350,
) -> dict[str, Path]:
    """Generate concise, publication-style comparison figures.

    Output:
    - One heatmap per split: train / val / test
    - One test-average ranking bar chart
    """
    summary_csv = Path(summary_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, run_id, combo_order, model_order = _load_and_prepare(summary_csv, run_id)
    _set_plot_style()

    split_cols = ["train", "val", "test"]
    split_titles = ["Train RMSE", "Validation RMSE", "Test RMSE"]
    all_vals = df[split_cols].to_numpy().reshape(-1)
    vmin = float(all_vals.min())
    vmax = float(all_vals.max())
    cmap = sns.color_palette("YlGnBu", as_cmap=True)

    out = {}

    # 1) three separate heatmaps to avoid crowded all-in-one figure.
    for split, title in zip(split_cols, split_titles):
        mat = (
            df.pivot_table(index="model", columns="combo", values=split, aggfunc="first")
            .reindex(index=model_order, columns=combo_order)
        )
        fig, ax = plt.subplots(figsize=(10.8, 6.4), constrained_layout=True)
        sns.heatmap(
            mat,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            annot=annotate,
            fmt=".4f",
            linewidths=0.6,
            linecolor="#EFEFEF",
            cbar_kws={"shrink": 0.85, "label": "RMSE"},
        )
        ax.set_title(f"{title} Across Missingness Settings", fontweight="bold", pad=10)
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
        df.groupby("model", as_index=False)["test"].mean().sort_values("test", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(8.8, 5.6), constrained_layout=True)
    sns.barplot(data=rank_df, x="test", y="model", ax=ax, color="#4C78A8")
    ax.set_title("Model Ranking by Mean Test RMSE", fontweight="bold", pad=10)
    ax.set_xlabel("Mean Test RMSE (lower is better)")
    ax.set_ylabel("Model")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, v in enumerate(rank_df["test"]):
        ax.text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=9)

    rank_path = output_dir / f"baseline_rank_test_{run_id}.png"
    fig.savefig(rank_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    out["rank_test"] = rank_path

    return out
