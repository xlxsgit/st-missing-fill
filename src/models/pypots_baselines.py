import time
import numpy as np
import torch

from pypots.imputation import GRUD, SAITS, USGAN, iTransformer

from src.evaluate import rmse_on_missing_3d
from src.models.utils import prepare_window_data

def _build_pypots_model(
    model_name: str,
    n_steps: int,
    n_features: int,
    epochs: int,
    batch_size: int,
    patience: int,
    verbose: bool,
    device,
    **kwargs
):
    if model_name == "saits":
        return SAITS(
            n_steps=n_steps,
            n_features=n_features,
            n_layers=4,
            d_model=32,
            n_heads=4,
            d_k=8,
            d_v=32,
            d_ffn=64,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            device=device,
            verbose=verbose,
        )
    if model_name == "grud":
        return GRUD(
            n_steps=n_steps,
            n_features=n_features,
            rnn_hidden_size=64,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            device=device,
            verbose=verbose,
        )
    if model_name == "usgan":
        return USGAN(
            n_steps=n_steps,
            n_features=n_features,
            rnn_hidden_size=64,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            device=device,
            verbose=verbose,
        )
    if model_name == "itransformer":
        return iTransformer(
            n_steps=n_steps,
            n_features=n_features,
            n_layers=4,
            d_model=32,
            n_heads=4,
            d_k=8,
            d_v=32,
            d_ffn=64,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            device=device,
            verbose=verbose,
        )
    raise ValueError(f"Unknown PyPOTS baseline {model_name}")

def run_pypots_deep_on_splits(
    model_name: str,
    split_y: dict,
    split_masked: dict,
    split_masks: dict,
    window_size: int,
    epochs: int,
    batch_size: int,
    patience: int,
    max_windows: int | None,
    verbose: bool,
    device,
    hparams_override: dict | None = None,
    mode: str = "all",
    project_root: str | None = None,
    run_dir: str | None = None,
    pattern: str = "mcar",
    pi: float = 0.1,
) -> tuple[dict[str, float], dict[str, float]]:
    d = prepare_window_data(split_y, split_masked, split_masks, window_size, max_windows)
    build_kwargs = {
        "n_steps": d["y_train_masked_win"].shape[1],
        "n_features": d["y_train_masked_win"].shape[2],
        "epochs": epochs,
        "batch_size": batch_size,
        "patience": patience,
        "verbose": verbose,
        "device": device,
    }
    
    import json
    from pathlib import Path
    _run_dir = Path(run_dir) if run_dir else (Path(project_root) if project_root else Path.cwd())
    model_save_dir = _run_dir / "saved_models" / model_name
    model_save_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_save_dir / f"{model_name}_{pattern}_{pi}.pypots"
    hparams_path = model_save_dir / f"{model_name}_{pattern}_{pi}_hparams.json"

    if mode == "test" and hparams_override is None:
        if hparams_path.exists():
            with open(hparams_path, "r") as f:
                saved_config = json.load(f)
                hparams_override = saved_config.get("best_hparams", None)

    # 针对受限于 Mac MPS 同步延迟的长循环 RNN 网络 (GRUD/USGAN)，强制回到 CPU 算力
    if model_name in ["usgan", "grud"]:
        build_kwargs["device"] = torch.device("cpu")
    if hparams_override:
        # 只保留不是 epochs 和 batch_size 的模型网络解构超参数覆盖
        override = {k: v for k, v in hparams_override.items() if k not in ["epochs", "batch_size"]}
        build_kwargs.update(override)

    model = _build_pypots_model(model_name, **build_kwargs)
    
    if mode in ("train", "all"):
        t0 = time.perf_counter()
        model.fit({"X": d["y_train_masked_win"]}, val_set={"X": d["y_val_masked_win"], "X_ori": d["y_val_win"]})
        train_seconds = time.perf_counter() - t0
        
        # Save PyPOTS model
        model.save(str(model_path), overwrite=True)
        
        # Extract parameter count if possible
        num_params = 0
        if hasattr(model, "model"):
            num_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            
        clean_hparams = {k: v for k, v in build_kwargs.items() if k != "device"}
        hparams_to_save = {
            "model_name": model_name,
            "pattern": pattern,
            "pi": pi,
            "best_hparams": hparams_override or clean_hparams,
            "num_params": num_params,
            "epochs": epochs,
            "batch_size": batch_size,
            "window_size": window_size,
        }
        with open(hparams_path, "w") as f:
            json.dump(hparams_to_save, f, indent=4)
    else:
        train_seconds = 0.0
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found for inference: {model_path}. Please run in 'train' mode first.")
        model.load(str(model_path))

    def _predict_with_device_fallback(x: np.ndarray, chunk_size: int = 1500) -> np.ndarray:
        results = []
        for i in range(0, len(x), chunk_size):
            x_chunk = x[i : i + chunk_size]
            try:
                res = model.predict({"X": x_chunk})["imputation"]
            except RuntimeError as e:
                msg = str(e).lower()
                is_mps_oom = ("mps" in msg) and ("out of memory" in msg)
                if not is_mps_oom:
                    raise
                # Keep training on MPS, but safely finish inference on CPU.
                if hasattr(model, "model"):
                    model.model = model.model.to("cpu")
                if hasattr(model, "device"):
                    model.device = torch.device("cpu")
                res = model.predict({"X": x_chunk})["imputation"]
            
            results.append(res)
                
        return np.concatenate(results, axis=0) if results else np.zeros_like(x)

    t1 = time.perf_counter()
    train_pred = _predict_with_device_fallback(d["y_train_masked_win"]) if mode in ("train", "all") else np.zeros_like(d["y_train_masked_win"])
    val_pred = _predict_with_device_fallback(d["y_val_masked_win"]) if mode in ("train", "all") else np.zeros_like(d["y_val_masked_win"])
    test_pred = _predict_with_device_fallback(d["y_test_masked_win"]) if mode in ("test", "all") else np.zeros_like(d["y_test_masked_win"])
    infer_seconds = time.perf_counter() - t1
    
    # 强制清理基于 pypots 的 PyTorch 计算图，防止 Mac 内存溢出崩毁
    import gc
    del model
    gc.collect()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    return {
        "train": rmse_on_missing_3d(train_pred, d["y_train_win"], d["train_mask_win"]) if mode in ("train", "all") else np.nan,
        "val": rmse_on_missing_3d(val_pred, d["y_val_win"], d["val_mask_win"]) if mode in ("train", "all") else np.nan,
        "test": rmse_on_missing_3d(test_pred, d["y_test_win"], d["test_mask_win"]) if mode in ("test", "all") else np.nan,
    }, {
        "train_seconds": float(train_seconds),
        "infer_seconds": float(infer_seconds),
        "total_seconds": float(train_seconds + infer_seconds),
    }
