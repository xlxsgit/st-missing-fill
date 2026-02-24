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
    max_windows: int | None,
    verbose: bool,
    device,
    hparams_override: dict | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    d = prepare_window_data(split_y, split_masked, split_masks, window_size, max_windows)
    build_kwargs = {
        "n_steps": d["y_train_masked_win"].shape[1],
        "n_features": d["y_train_masked_win"].shape[2],
        "epochs": epochs,
        "batch_size": batch_size,
        "verbose": verbose,
        "device": device,
    }
    if hparams_override:
        # 只保留不是 epochs 和 batch_size 的模型网络解构超参数覆盖
        override = {k: v for k, v in hparams_override.items() if k not in ["epochs", "batch_size"]}
        build_kwargs.update(override)

    model = _build_pypots_model(model_name, **build_kwargs)
    t0 = time.perf_counter()
    model.fit({"X": d["y_train_masked_win"]}, val_set={"X": d["y_val_masked_win"], "X_ori": d["y_val_win"]})
    train_seconds = time.perf_counter() - t0

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
                if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
                if hasattr(model, "model"):
                    model.model = model.model.to("cpu")
                if hasattr(model, "device"):
                    model.device = torch.device("cpu")
                res = model.predict({"X": x_chunk})["imputation"]
            
            results.append(res)
            # Free memory promptly
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
                
        return np.concatenate(results, axis=0) if results else np.zeros_like(x)

    t1 = time.perf_counter()
    train_pred = _predict_with_device_fallback(d["y_train_masked_win"])
    val_pred = _predict_with_device_fallback(d["y_val_masked_win"])
    test_pred = _predict_with_device_fallback(d["y_test_masked_win"])
    infer_seconds = time.perf_counter() - t1
    return {
        "train": rmse_on_missing_3d(train_pred, d["y_train_win"], d["train_mask_win"]),
        "val": rmse_on_missing_3d(val_pred, d["y_val_win"], d["val_mask_win"]),
        "test": rmse_on_missing_3d(test_pred, d["y_test_win"], d["test_mask_win"]),
    }, {
        "train_seconds": float(train_seconds),
        "infer_seconds": float(infer_seconds),
        "total_seconds": float(train_seconds + infer_seconds),
    }
