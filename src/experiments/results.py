import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def save_experiment_results(
    project_root: Path,
    run_dir: Path,
    run_id: str,
    args: dict,
    models: list[str],
    patterns: list[str],
    pis: list[float],
    shape_info: dict,
    df_long: pd.DataFrame,
) -> None:
    """Save all artifacts for a run: CSVs, pivot tables, logs, configs, metrics."""
    # 1. Long results
    if "combo_seed" in df_long.columns:
        df_long = df_long.drop(columns=["combo_seed"])
    result_csv = run_dir / "results_long.csv"
    df_long.to_csv(result_csv, index=False, float_format="%.4f")

    # 2. Pivot results
    pivot = (
        df_long.pivot_table(
            index=["model", "pattern", "pi"],
            columns="split",
            values="rmse",
            aggfunc="first",
        )
        .reset_index()
        .sort_values(["model", "pattern", "pi"])
    )
    pivot_csv = run_dir / "results_pivot.csv"
    pivot.to_csv(pivot_csv, index=False, float_format="%.4f")

    # 3. Timing summary
    timing_pivot = (
        df_long.groupby(["model", "pattern", "pi"], as_index=False)[
            ["total_seconds"]
        ]
        .first()
        .sort_values(["model", "pattern", "pi"])
    )
    timing_csv = run_dir / "timing_summary.csv"
    timing_pivot.to_csv(timing_csv, index=False, float_format="%.4f")

    # 4. Remove global summary as requested, we keep everything inside run_dir.
    summary_df = pivot.copy()
    summary_df = summary_df.merge(timing_pivot, on=["model", "pattern", "pi"], how="left")
    
    # 5. Generate validation figures locally inside run_dir
    from src.visualization.baseline_compare import plot_baseline_comparison
    plot_baseline_comparison(summary_df, output_dir=run_dir, run_id=run_id)

    # 5. Configs
    config_json = run_dir / "config.json"
    with config_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": args,
                "models": models,
                "patterns": patterns,
                "pis": pis,
                "shape": shape_info,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 6. Metrics
    metrics_json = run_dir / "metrics.json"
    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "run_time": datetime.now().isoformat(timespec="seconds"),
                "num_rows": int(len(df_long)),
                "num_combinations": int(len(pivot)),
                "results_pivot": pivot.to_dict(orient="records"),
                "results_long_head": df_long.head(30).to_dict(orient="records"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Removed verbose log lines for incremental saves to keep console clean
