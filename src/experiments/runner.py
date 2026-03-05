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
import optuna
import logging
from src.models.search_space import get_search_space_for_model

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
    parser.add_argument("--patience", type=int, default=10, help="Number of epochs to wait before early stopping")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["train", "test", "all"], default="all", help="Whether to train models, test pre-trained models, or do all.")
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

    if args.run_name:
        run_id = args.run_name
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_root / "logs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "train.log"

    with log_path.open("w", encoding="utf-8") as log_f:
        tee = Tee(sys.stdout, log_f)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            print(f"Run dir: {run_dir}")
            print("Args:")
            print(json.dumps(vars(args), indent=2))

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

            # Suppress pypots warnings
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning, module="pypots")
            import logging
            logging.getLogger("pypots").setLevel(logging.ERROR)
            
            try:
                from pypots.utils.logging import logger_creator as pypots_logger_creator
                pypots_logger_creator.set_level("error")
                pypots_logger_creator.set_level = lambda *args, **kwargs: None
                from tsdb.utils.logging import logger_creator as tsdb_logger_creator
                tsdb_logger_creator.set_level("error")
                tsdb_logger_creator.set_level = lambda *args, **kwargs: None
            except Exception:
                pass

            rows = []
            combo_idx = 0
            
            with contextlib.nullcontext():
                print(f"{'Model':<14} | {'Pattern':<9} | {'PI':<5} | {'Test RMSE':<10} | {'Time(s)':<8} | Status")
                print("-" * 65)
                for model_name in models:
                    for pattern in patterns:
                        for pi in pis:
                            msg_start = f"{model_name:<14} | {pattern:<9} | {pi:<5.2f} | {'-':<10} | {'-':<8} | ⏳"
                            print(msg_start, end="\r", flush=True)

                            combo_seed = None if args.seed is None else args.seed + combo_idx * 1009
                            combo_idx += 1
                            
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
                            if args.hpo_trials > 0 and args.mode in ("train", "all"):
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
                                            patience=args.patience,
                                            max_windows=args.max_windows,
                                            verbose=False,  # HPO期间强制静默
                                            knn_chunk_steps=args.knn_chunk_steps,
                                            mice_chunk_steps=args.mice_chunk_steps,
                                            knn_neighbors=hparams.get("knn_neighbors", args.knn_neighbors),
                                            mice_max_iter=hparams.get("mice_max_iter", args.mice_max_iter),
                                            mice_tol=hparams.get("mice_tol", args.mice_tol),
                                            mice_quiet_warnings=True, # HPO强制静音mice
                                            hparams_override=hparams,
                                            mode="train",
                                            project_root=project_root,
                                            run_dir=run_dir,
                                            pattern=pattern,
                                            pi=pi
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
                            rmse_by_split, timing = run_baseline_on_splits(
                                model_name=model_name,
                                split_y=split_y,
                                split_masked=masked,
                                split_masks=masks,
                                device=device,
                                window_size=args.window_size,
                                epochs=args.epochs,
                                batch_size=args.batch_size,
                                patience=args.patience,
                                max_windows=args.max_windows,
                                verbose=not args.quiet_train,
                                knn_chunk_steps=args.knn_chunk_steps,
                                mice_chunk_steps=args.mice_chunk_steps,
                                knn_neighbors=args.knn_neighbors,
                                mice_max_iter=args.mice_max_iter,
                                mice_tol=args.mice_tol,
                                mice_quiet_warnings=not args.mice_show_warnings,
                                hparams_override=best_hparams,
                                mode=args.mode,
                                project_root=project_root,
                                run_dir=run_dir,
                                pattern=pattern,
                                pi=pi
                            )
    
                            for split_name in ["train", "val", "test"]:
                                split_rmse = rmse_by_split.get(split_name, np.nan)
                                rows.append(
                                    {
                                        "model": model_name,
                                        "pattern": pattern,
                                        "pi": pi,
                                        "split": split_name,
                                        "rmse": split_rmse,
                                        "missing_rate": float((masks[split_name] == 0).mean()),
                                        "split_seed": split_seed[split_name],
                                        "total_seconds": float(timing["total_seconds"]),
                                    }
                                )
    
                            test_rmse = rmse_by_split.get('test', np.nan)
                            test_rmse_str = f"{test_rmse:<10.4f}" if not np.isnan(test_rmse) else f"{'N/A':<10}"
                            msg_done = f"{model_name:<14} | {pattern:<9} | {pi:<5.2f} | {test_rmse_str} | {timing['total_seconds']:<8.2f} | ✅"
                            print(msg_done, flush=True)
                            
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
                            global_summary_path = run_dir / "summary_all_parts.csv"
                            
                            hp_path = run_dir / "saved_models" / model_name / f"{model_name}_{pattern}_{pi}_hparams.json"
                            num_params = 0
                            hparam_str = "N/A"
                            if hp_path.exists():
                                with open(hp_path, "r") as f:
                                    hp_data = json.load(f)
                                    num_params = hp_data.get("num_params", 0)
                                    if "best_hparams" in hp_data:
                                        hparam_str = json.dumps(hp_data["best_hparams"])
                            elif best_hparams:
                                hparam_str = json.dumps(best_hparams)
                                
                            from src.models.search_space import get_search_bounds
                            search_bounds_str = get_search_bounds(model_name)

                            train_rmse = rmse_by_split.get("train", np.nan)
                            val_rmse = rmse_by_split.get("val", np.nan)
                            test_rmse = rmse_by_split.get("test", np.nan)
                            
                            train_time = timing["train_seconds"]
                            test_time = timing["infer_seconds"]
                            
                            new_row = {
                                "RUN_NAME": run_id,
                                "模型": model_name,
                                "缺失模式": pattern,
                                "缺失率": pi,
                                "参数量": num_params,
                                "超参数范围": search_bounds_str,
                                "最优超参数": hparam_str,
                                "epochs": args.epochs,
                                "batch size": args.batch_size,
                                "window size": args.window_size,
                                "train 范围": f"{args.train_start} to {args.train_end}",
                                "train rmse": train_rmse,
                                "val 范围": f"{args.val_start} to {args.val_end}",
                                "val rmse": val_rmse,
                                "test 范围": f"{args.test_start} to {args.test_end}",
                                "test rmse": test_rmse,
                                "训练耗时": float(train_time),
                                "推理耗时": float(test_time)
                            }
                            
                            if global_summary_path.exists():
                                df_global = pd.read_csv(global_summary_path)
                            else:
                                df_global = pd.DataFrame(columns=new_row.keys())
                                
                            mask = (df_global.get("RUN_NAME") == run_id) & \
                                   (df_global.get("模型") == model_name) & \
                                   (df_global.get("缺失模式") == pattern) & \
                                   (df_global.get("缺失率") == pi)
                                   
                            if not mask.any():
                                df_global = pd.concat([df_global, pd.DataFrame([new_row])], ignore_index=True)
                            else:
                                idx = df_global[mask].index[0]
                                if args.mode in ("train", "all"):
                                    df_global.loc[idx, "参数量"] = num_params
                                    df_global.loc[idx, "最优超参数"] = hparam_str
                                    df_global.loc[idx, "train 范围"] = new_row["train 范围"]
                                    if pd.notna(train_rmse):
                                        df_global.loc[idx, "train rmse"] = train_rmse
                                    df_global.loc[idx, "val 范围"] = new_row["val 范围"]
                                    if pd.notna(val_rmse):
                                        df_global.loc[idx, "val rmse"] = val_rmse
                                    df_global.loc[idx, "训练耗时"] = float(train_time)
                                if args.mode in ("test", "all"):
                                    df_global.loc[idx, "test 范围"] = new_row["test 范围"]
                                    if pd.notna(test_rmse):
                                        df_global.loc[idx, "test rmse"] = test_rmse
                                    df_global.loc[idx, "推理耗时"] = float(test_time)
                                    
                            df_global.to_csv(global_summary_path, index=False, float_format="%.4f")

                print("🏁 All combinations completed!")

            print(f"\nSaved log: {log_path}")
