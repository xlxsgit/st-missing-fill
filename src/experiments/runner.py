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
from src.models.dispatcher import SUPPORTED_MODELS, run_baseline_on_splits
from src.experiments.results import save_experiment_results
from src.models.search_space import get_search_space_for_model
import optuna

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
    parser.add_argument("--knn-chunk-steps", type=int, default=1008)
    parser.add_argument("--mice-chunk-steps", type=int, default=720)
    parser.add_argument("--knn-neighbors", type=int, default=5)
    parser.add_argument("--mice-max-iter", type=int, default=8)
    parser.add_argument("--mice-tol", type=float, default=1e-3)
    parser.add_argument("--mice-show-warnings", action="store_true")
    parser.add_argument("--seq-n1", type=int, default=24)
    parser.add_argument("--seq-p1", type=float, default=0.5)
    parser.add_argument("--seq-l-obse-base", type=int, default=10)
    parser.add_argument("--seq-p0", type=float, default=0.5)
    parser.add_argument("--scm-pi-hat", type=float, default=0.95)
    parser.add_argument("--hpo-trials", type=int, default=0, help="Number of trials for hyperparameter optimization.")
    
    # 动态数据切分日期参数
    parser.add_argument("--train-start", type=str, default="2023-01-01")
    parser.add_argument("--train-end", type=str, default="2023-01-31")
    parser.add_argument("--val-start", type=str, default="2023-02-01")
    parser.add_argument("--val-end", type=str, default="2023-02-28")
    parser.add_argument("--test-start", type=str, default="2023-03-01")
    parser.add_argument("--test-end", type=str, default="2023-03-31")
    
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
    run_dir = project_root / "logs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    # 创建 latest 软链
    latest_dir = project_root / "logs" / "latest"
    if latest_dir.is_symlink() or latest_dir.exists():
        if hasattr(latest_dir, "unlink"):
            latest_dir.unlink(missing_ok=True)
    try:
        latest_dir.symlink_to(run_dir.name, target_is_directory=True)
    except Exception:
        pass

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
            
            split_args = {
                "train_start": args.train_start,
                "train_end": args.train_end,
                "val_start": args.val_start,
                "val_end": args.val_end,
                "test_start": args.test_start,
                "test_end": args.test_end
            }
            split_y = split_by_datetime(ground_y, time_index, **split_args)
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
                        # ========= HPO Objective ==========
                        best_hparams = None
                        if args.hpo_trials > 0:
                            from src.models.search_space import has_search_space
                            if has_search_space(model_name):
                                print(f"*** Starting HPO for {model_name} with {args.hpo_trials} trials ***")
                                
                                def objective(trial: optuna.Trial) -> float:
                                    # 1. 抽取超参
                                    hparams = get_search_space_for_model(model_name, trial)
                                    # 2. 调度执行
                                    rmse, _ = run_baseline_on_splits(
                                        model_name=model_name,
                                        split_y=split_y,
                                        split_masked=masked,
                                        split_masks=masks,
                                        device=device,
                                        window_size=args.window_size,
                                        epochs=hparams.get("epochs", args.epochs),
                                        batch_size=hparams.get("batch_size", args.batch_size),
                                        max_windows=args.max_windows,
                                        verbose=False,  # HPO期间强制静默
                                        knn_chunk_steps=args.knn_chunk_steps,
                                        mice_chunk_steps=args.mice_chunk_steps,
                                        knn_neighbors=hparams.get("knn_neighbors", args.knn_neighbors),
                                        mice_max_iter=hparams.get("mice_max_iter", args.mice_max_iter),
                                        mice_tol=hparams.get("mice_tol", args.mice_tol),
                                        mice_quiet_warnings=True, # HPO强制静音mice
                                        hparams_override=hparams
                                    )
                                    # 3. 目标为最小化在【验证集】上的 RMSE
                                    return rmse["val"]
                                
                                study = optuna.create_study(direction="minimize")
                                # 屏蔽 optuna 过多的INFO, 开启Error
                                optuna.logging.set_verbosity(optuna.logging.WARNING)
                                study.optimize(objective, n_trials=args.hpo_trials)
                                best_hparams = study.best_params
                                print(f"*** HPO Finished. Best Val RMSE: {study.best_value:.4f} with params: {best_hparams} ***")
                            else:
                                print(f"*** Skipped HPO for {model_name} (no search space defined) ***")

                        # ========= 最终模型运行与测试 ==========
                        print(f"[{'HPO BEST' if best_hparams else 'RUN'}] Executing final pass for {model_name}")
                        rmse_by_split, timing = run_baseline_on_splits(
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
                            knn_chunk_steps=args.knn_chunk_steps,
                            mice_chunk_steps=args.mice_chunk_steps,
                            knn_neighbors=args.knn_neighbors,
                            mice_max_iter=args.mice_max_iter,
                            mice_tol=args.mice_tol,
                            mice_quiet_warnings=not args.mice_show_warnings,
                            hparams_override=best_hparams
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
                                    "train_seconds": float(timing["train_seconds"]),
                                    "infer_seconds": float(timing["infer_seconds"]),
                                    "total_seconds": float(timing["total_seconds"]),
                                }
                            )
                        print(
                            f"RMSE train/val/test: "
                            f"{rmse_by_split['train']:.4f}/{rmse_by_split['val']:.4f}/{rmse_by_split['test']:.4f}"
                        )
                        print(
                            "Timing (s) train/infer/total: "
                            f"{timing['train_seconds']:.3f}/{timing['infer_seconds']:.3f}/{timing['total_seconds']:.3f}"
                        )
                        
                        # [INCREMENTAL SAVE] 递增存档以防止中断
                        df = pd.DataFrame(rows)
                        shape_info = {
                            "ground_X": list(ground_X.shape),
                            "ground_y": list(ground_y.shape),
                            "train": list(split_y["train"].shape),
                            "val": list(split_y["val"].shape),
                            "test": list(split_y["test"].shape),
                        }
                        save_experiment_results(
                            project_root, run_dir, run_id, vars(args), models, patterns, pis, shape_info, df
                        )

                        # 全量总结大双追加
                        global_summary_path = project_root / "logs" / "summary_all_parts.csv"
                        hparam_str = json.dumps(best_hparams) if best_hparams else "default"
                        combo_summary = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "run_id": run_id,
                            "model": model_name,
                            "pattern": pattern,
                            "pi": pi,
                            "hparams": hparam_str,
                            "train_rmse": rmse_by_split["train"],
                            "val_rmse": rmse_by_split["val"],
                            "test_rmse": rmse_by_split["test"],
                            "train_seconds": float(timing["train_seconds"]),
                            "infer_seconds": float(timing["infer_seconds"]),
                            "total_seconds": float(timing["total_seconds"])
                        }
                        df_combo = pd.DataFrame([combo_summary])
                        file_exists = global_summary_path.exists()
                        df_combo.to_csv(global_summary_path, mode="a", index=False, header=not file_exists)

            print(f"\nSaved log: {log_path}")
