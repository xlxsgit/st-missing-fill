import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.baseline_compare import plot_baseline_comparison


PART_PATTERNS = [
    re.compile(r"^\d{8}_\d{6}_part1_light_01234$"),
    re.compile(r"^\d{8}_\d{6}_part2_saits_01234$"),
    re.compile(r"^\d{8}_\d{6}_part3_grud_01234$"),
    re.compile(r"^\d{8}_\d{6}_part4_usgan_01234$"),
    re.compile(r"^\d{8}_\d{6}_part5_itransformer_01234$"),
]


def _find_latest_run(logs_dir: Path, pattern: re.Pattern) -> Path:
    candidates = [p for p in logs_dir.iterdir() if p.is_dir() and pattern.match(p.name)]
    if not candidates:
        raise FileNotFoundError(f"No run found for pattern: {pattern.pattern}")
    return sorted(candidates, key=lambda p: p.name)[-1]


def _load_part_df(run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "results_pivot.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    required = {"model", "pattern", "pi", "train", "val", "test"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
    return df[list(required)].copy()


def _plot_pattern_curves(summary_df: pd.DataFrame, output_dir: Path, run_id: str) -> Path:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )
    d = summary_df.copy()
    d["pattern"] = d["pattern"].str.upper()
    d["pi"] = d["pi"].astype(float)

    model_order = d.groupby("model", as_index=True)["test"].mean().sort_values().index.tolist()
    d["model"] = pd.Categorical(d["model"], categories=model_order, ordered=True)

    g = sns.relplot(
        data=d,
        x="pi",
        y="test",
        hue="model",
        col="pattern",
        kind="line",
        marker="o",
        linewidth=2,
        col_order=["MCAR", "SEQ", "SCM"],
        facet_kws={"sharey": True, "sharex": True},
        height=3.9,
        aspect=1.05,
    )
    g.set_axis_labels("Missing Rate (pi)", "Test RMSE")
    g.set_titles("{col_name}")
    g.figure.suptitle("Baseline Comparison Across Missingness Patterns", y=1.04, fontsize=14)
    for ax in g.axes.flat:
        ax.set_xticks(sorted(d["pi"].unique()))
        ax.grid(True, color="#EAEAEA", linewidth=0.8)

    out_path = output_dir / f"baseline_curve_test_by_pattern_{run_id}.png"
    g.figure.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close(g.figure)
    return out_path


def main() -> None:
    logs_dir = PROJECT_ROOT / "logs"
    figs_dir = PROJECT_ROOT / "data" / "figs" / "baseline_compare"
    figs_dir.mkdir(parents=True, exist_ok=True)

    runs = [_find_latest_run(logs_dir, pat) for pat in PART_PATTERNS]
    part_dfs = [_load_part_df(p) for p in runs]
    merged = pd.concat(part_dfs, axis=0, ignore_index=True).drop_duplicates(
        subset=["model", "pattern", "pi"], keep="last"
    )
    merged = merged.sort_values(["model", "pattern", "pi"]).reset_index(drop=True)

    run_id = "merged_parts_01234_latest"
    merged.insert(0, "run_id", run_id)

    # Attach timing stats if available.
    timing_frames = []
    for run_dir in runs:
        t_csv = run_dir / "timing_summary.csv"
        if t_csv.exists():
            tdf = pd.read_csv(t_csv)
            timing_cols = {"model", "pattern", "pi", "train_seconds", "infer_seconds", "total_seconds"}
            if timing_cols.issubset(set(tdf.columns)):
                timing_frames.append(tdf[list(timing_cols)].copy())
    if timing_frames:
        timing_df = (
            pd.concat(timing_frames, axis=0, ignore_index=True)
            .drop_duplicates(subset=["model", "pattern", "pi"], keep="last")
            .sort_values(["model", "pattern", "pi"])
        )
        merged = merged.merge(timing_df, on=["model", "pattern", "pi"], how="left")

    out_csv = logs_dir / "summary_all_parts.csv"
    merged.to_csv(out_csv, index=False, float_format="%.4f")

    outputs = plot_baseline_comparison(
        summary_csv=out_csv,
        output_dir=figs_dir,
        run_id=run_id,
        annotate=False,
        dpi=350,
    )
    curve_path = _plot_pattern_curves(merged, figs_dir, run_id)

    print(f"Merged summary saved: {out_csv}")
    for name, path in outputs.items():
        print(f"{name}: {path}")
    print(f"pattern_curve_test: {curve_path}")
    print("Merged runs:")
    for run_dir in runs:
        print(f"- {run_dir}")


if __name__ == "__main__":
    main()
