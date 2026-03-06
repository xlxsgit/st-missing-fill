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
    """Save experiment artifacts: config.json, metrics.json, and visualization."""
    if "combo_seed" in df_long.columns:
        df_long = df_long.drop(columns=["combo_seed"])

    # Build pivot for internal use (visualization / metrics)
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

    # Generate validation figures locally inside run_dir
    from src.visualization.baseline_compare import plot_baseline_comparison
    plot_baseline_comparison(pivot, output_dir=run_dir, run_id=run_id)

    # Configs
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

    # Metrics
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
