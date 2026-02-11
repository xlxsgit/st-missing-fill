import argparse
import contextlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.load import load_data
from src.data.misser import simulate_missingness
from src.data.splitter import load_time_index, split_by_datetime
from src.models.baselines import SUPPORTED_MODELS, run_baseline_on_splits


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified experiment runner for missing patterns")
    parser.add_argument(
        "--models",
        type=str,
        default="locf,saits,grud,usgan,itransformer,knn,mice,vcaan",
    )
    parser.add_argument("--patterns", type=str, default="mcar,seq,scm")
    parser.add_argument("--pis", type=str, default="0.1,0.3,0.5")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=6 * 24)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--quiet-train", action="store_true")
    parser.add_argument("--seq-n1", type=int, default=24)
    parser.add_argument("--seq-p1", type=float, default=0.5)
    parser.add_argument("--seq-l-obse-base", type=int, default=10)
    parser.add_argument("--seq-p0", type=float, default=0.5)
    parser.add_argument("--scm-pi-hat", type=float, default=0.95)
    return parser


def parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_csv_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def build_missing_inputs(
    split_y: dict[str, np.ndarray],
    pattern: str,
    pi: float,
    clusters: list[int],
    base_seed: int | None,
    seq_params: dict,
    scm_params: dict,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    rng = np.random.default_rng(base_seed)
    split_seed = {
        "train": int(rng.integers(0, 2**31 - 1)),
        "val": int(rng.integers(0, 2**31 - 1)),
        "test": int(rng.integers(0, 2**31 - 1)),
    }
    masked = {}
    masks = {}
    for split_name in ["train", "val", "test"]:
        y_masked, mask = simulate_missingness(
            split_y[split_name],
            pi,
            pattern,
            S_cluster=clusters if pattern == "scm" else None,
            seed=split_seed[split_name],
            seq_params=seq_params,
            scm_params=scm_params,
        )
        masked[split_name] = y_masked
        masks[split_name] = mask
    return masked, masks, split_seed


def run_experiments(args, project_root: Path) -> None:
    models = parse_csv_list(args.models)
    patterns = parse_csv_list(args.patterns)
    pis = parse_csv_float_list(args.pis)

    for p in patterns:
        if p not in {"mcar", "seq", "scm"}:
            raise ValueError(f"Unknown pattern: {p}")
    for m in models:
        if m.lower() not in SUPPORTED_MODELS:
            raise ValueError(f"Unknown model: {m}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        run_id = f"{run_id}_{args.run_name}"
    run_dir = project_root / "data" / "res" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "train.log"

    with log_path.open("w", encoding="utf-8") as log_f:
        tee = Tee(sys.stdout, log_f)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            print(f"Run dir: {run_dir}")
            print(f"Args: {vars(args)}")

            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            print(f"Using device: {device}")

            ground_X, ground_y, all_stations, all_stations_cluster, vars_info = load_data()
            time_index = load_time_index(project_root)
            split_y = split_by_datetime(ground_y, time_index)
            print(
                f"Loaded ground_X={ground_X.shape}, ground_y={ground_y.shape}, "
                f"splits train={split_y['train'].shape}, val={split_y['val'].shape}, test={split_y['test'].shape}"
            )
            print(f"Stations={len(all_stations)}, target={vars_info['y']}")

            seq_params = {
                "n1": args.seq_n1,
                "p1": args.seq_p1,
                "L_obse_base": args.seq_l_obse_base,
                "p0": args.seq_p0,
            }
            scm_params = {"pi_hat": args.scm_pi_hat}

            rows = []
            combo_idx = 0
            for model_name in models:
                for pattern in patterns:
                    for pi in pis:
                        combo_seed = None if args.seed is None else args.seed + combo_idx * 1009
                        combo_idx += 1
                        print(
                            f"\n[RUN] model={model_name}, pattern={pattern}, pi={pi}, combo_seed={combo_seed}"
                        )
                        masked, masks, split_seed = build_missing_inputs(
                            split_y,
                            pattern,
                            pi,
                            all_stations_cluster,
                            combo_seed,
                            seq_params,
                            scm_params,
                        )
                        rmse_by_split = run_baseline_on_splits(
                            model_name=model_name,
                            split_y=split_y,
                            split_masked=masked,
                            split_masks=masks,
                            device=device,
                            window_size=args.window_size,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            max_windows=args.max_windows,
                            verbose=not args.quiet_train,
                        )

                        for split_name in ["train", "val", "test"]:
                            rows.append(
                                {
                                    "model": model_name,
                                    "pattern": pattern,
                                    "pi": pi,
                                    "split": split_name,
                                    "rmse": rmse_by_split[split_name],
                                    "missing_rate": float((masks[split_name] == 0).mean()),
                                    "combo_seed": combo_seed,
                                    "split_seed": split_seed[split_name],
                                }
                            )
                        print(
                            f"RMSE train/val/test: "
                            f"{rmse_by_split['train']:.4f}/{rmse_by_split['val']:.4f}/{rmse_by_split['test']:.4f}"
                        )

            df = pd.DataFrame(rows)
            result_csv = run_dir / "results_long.csv"
            df.to_csv(result_csv, index=False)

            pivot = (
                df.pivot_table(
                    index=["model", "pattern", "pi"],
                    columns="split",
                    values="rmse",
                    aggfunc="first",
                )
                .reset_index()
                .sort_values(["model", "pattern", "pi"])
            )
            pivot_csv = run_dir / "results_pivot.csv"
            pivot.to_csv(pivot_csv, index=False)

            summary_csv = project_root / "data" / "res" / "summary.csv"
            summary_df = pivot.copy()
            summary_df.insert(0, "run_id", run_id)
            summary_df.to_csv(summary_csv, index=False, float_format="%.4f")

            with (run_dir / "config.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "args": vars(args),
                        "models": models,
                        "patterns": patterns,
                        "pis": pis,
                        "shape": {
                            "ground_X": list(ground_X.shape),
                            "ground_y": list(ground_y.shape),
                            "train": list(split_y["train"].shape),
                            "val": list(split_y["val"].shape),
                            "test": list(split_y["test"].shape),
                        },
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            metrics_json = run_dir / "metrics.json"
            with metrics_json.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "run_id": run_id,
                        "run_time": datetime.now().isoformat(timespec="seconds"),
                        "num_rows": int(len(df)),
                        "num_combinations": int(len(pivot)),
                        "results_pivot": pivot.to_dict(orient="records"),
                        "results_long_head": df.head(30).to_dict(orient="records"),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            print("\nResults (pivot):")
            print(pivot.to_string(index=False))
            print(f"\nSaved log: {log_path}")
            print(f"Saved long results: {result_csv}")
            print(f"Saved pivot results: {pivot_csv}")
            print(f"Saved metrics: {metrics_json}")
            print(f"Updated summary: {summary_csv}")
