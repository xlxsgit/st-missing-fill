import time
import warnings
import numpy as np
import torch

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import IterativeImputer, KNNImputer

from pypots.imputation import GRUD, LOCF, SAITS, USGAN, iTransformer

from src.data.splitter import reshape_mask_2d, reshape_windows_2d
from src.evaluate import rmse_on_missing_2d, rmse_on_missing_3d
from src.models.vcaan import run_vcaan_on_splits


SUPPORTED_MODELS = {
    "locf",
    "saits",
    "grud",
    "usgan",
    "itransformer",
    "knn",
    "mice",
    "vcaan",
}


def _truncate_splits_for_fast_run(
    split_y: dict,
    split_masked: dict,
    split_masks: dict,
    window_size: int,
    max_windows: int | None,
):
    if max_windows is None:
        return split_y, split_masked, split_masks
    keep_steps = max(window_size * max(max_windows, 1), 1)
    out_y = {}
    out_masked = {}
    out_masks = {}
    for split_name in ["train", "val", "test"]:
        t = split_y[split_name].shape[1]
        use_t = min(keep_steps, t)
        out_y[split_name] = split_y[split_name][:, :use_t]
        out_masked[split_name] = split_masked[split_name][:, :use_t, :]
        out_masks[split_name] = split_masks[split_name][:, :use_t]
    return out_y, out_masked, out_masks


def _prepare_window_data(
    split_y: dict,
    split_masked: dict,
    split_masks: dict,
    window_size: int,
    max_windows: int | None,
):
    y_train_win = reshape_windows_2d(split_y["train"], window_size)
    y_val_win = reshape_windows_2d(split_y["val"], window_size)
    y_test_win = reshape_windows_2d(split_y["test"], window_size)

    y_train_masked_win = reshape_windows_2d(split_masked["train"][..., 0], window_size)
    y_val_masked_win = reshape_windows_2d(split_masked["val"][..., 0], window_size)
    y_test_masked_win = reshape_windows_2d(split_masked["test"][..., 0], window_size)

    train_mask_win = reshape_mask_2d(split_masks["train"], window_size)
    val_mask_win = reshape_mask_2d(split_masks["val"], window_size)
    test_mask_win = reshape_mask_2d(split_masks["test"], window_size)

    if max_windows is not None:
        k = max(max_windows, 1)
        y_train_win = y_train_win[:k]
        y_val_win = y_val_win[:k]
        y_test_win = y_test_win[:k]
        y_train_masked_win = y_train_masked_win[:k]
        y_val_masked_win = y_val_masked_win[:k]
        y_test_masked_win = y_test_masked_win[:k]
        train_mask_win = train_mask_win[:k]
        val_mask_win = val_mask_win[:k]
        test_mask_win = test_mask_win[:k]
    return {
        "y_train_win": y_train_win,
        "y_val_win": y_val_win,
        "y_test_win": y_test_win,
        "y_train_masked_win": y_train_masked_win,
        "y_val_masked_win": y_val_masked_win,
        "y_test_masked_win": y_test_masked_win,
        "train_mask_win": train_mask_win,
        "val_mask_win": val_mask_win,
        "test_mask_win": test_mask_win,
    }


def _run_locf_on_splits(split_y: dict, split_masked: dict, split_masks: dict, device) -> tuple[dict[str, float], dict[str, float]]:
    model = LOCF(device=device)
    res = {}
    t0 = time.perf_counter()
    for split_name in ["train", "val", "test"]:
        y_hat = model.impute({"X": split_masked[split_name]}).squeeze()
        res[split_name] = rmse_on_missing_2d(y_hat, split_y[split_name], split_masks[split_name])
    infer_seconds = time.perf_counter() - t0
    timing = {
        "train_seconds": 0.0,
        "infer_seconds": float(infer_seconds),
        "total_seconds": float(infer_seconds),
    }
    return res, timing


def _run_sklearn_on_splits(
    split_y: dict,
    split_masked: dict,
    split_masks: dict,
    model_name: str,
    knn_chunk_steps: int,
    mice_chunk_steps: int,
    knn_neighbors: int,
    mice_max_iter: int,
    mice_tol: float,
    mice_quiet_warnings: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    # Chunked imputation to avoid very slow behavior on long time series.
    def _fit_transform_by_chunk(
        y_train_obs: np.ndarray,
        y_val_obs: np.ndarray,
        y_test_obs: np.ndarray,
        algo: str,
        chunk_steps: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        # each input shape: (S, T)
        s, t_train = y_train_obs.shape
        t_val = y_val_obs.shape[1]
        t_test = y_test_obs.shape[1]
        out_train = np.empty((s, t_train), dtype=float)
        out_val = np.empty((s, t_val), dtype=float)
        out_test = np.empty((s, t_test), dtype=float)
        train_seconds, infer_seconds = 0.0, 0.0

        num_chunks = max(t_train // chunk_steps + (1 if t_train % chunk_steps else 0), 1)
        for ci in range(num_chunks):
            st = ci * chunk_steps
            ed_train = min(st + chunk_steps, t_train)
            ed_val = min(st + chunk_steps, t_val)
            ed_test = min(st + chunk_steps, t_test)

            x_train = y_train_obs[:, st:ed_train].T
            x_val = y_val_obs[:, st:ed_val].T if st < t_val else np.empty((0, s))
            x_test = y_test_obs[:, st:ed_test].T if st < t_test else np.empty((0, s))

            valid_cols = ~np.all(np.isnan(x_train), axis=0)
            x_train_valid = x_train[:, valid_cols]

            if algo == "knn":
                imp = KNNImputer(n_neighbors=knn_neighbors)
            elif algo == "mice":
                imp = IterativeImputer(
                    random_state=42,
                    sample_posterior=False,
                    max_iter=mice_max_iter,
                    tol=mice_tol,
                )
            else:
                raise ValueError(f"Unknown sklearn baseline {algo}")

            if x_train_valid.shape[1] > 0:
                t0 = time.perf_counter()
                if algo == "mice" and mice_quiet_warnings:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="\\[IterativeImputer\\] Early stopping criterion not reached\\.",
                            category=ConvergenceWarning,
                        )
                        x_train_valid_hat = imp.fit_transform(x_train_valid)
                else:
                    x_train_valid_hat = imp.fit_transform(x_train_valid)
                train_seconds += time.perf_counter() - t0
            else:
                x_train_valid_hat = x_train_valid

            x_train_hat = np.full_like(x_train, np.nan, dtype=float)
            x_train_hat[:, valid_cols] = x_train_valid_hat
            x_train_hat = np.nan_to_num(x_train_hat, nan=0.0)
            out_train[:, st:ed_train] = x_train_hat.T

            if x_val.shape[0] > 0:
                if x_train_valid.shape[1] > 0:
                    t0 = time.perf_counter()
                    x_val_valid_hat = imp.transform(x_val[:, valid_cols])
                    infer_seconds += time.perf_counter() - t0
                else:
                    x_val_valid_hat = x_val[:, valid_cols]
                x_val_hat = np.full_like(x_val, np.nan, dtype=float)
                x_val_hat[:, valid_cols] = x_val_valid_hat
                x_val_hat = np.nan_to_num(x_val_hat, nan=0.0)
                out_val[:, st:ed_val] = x_val_hat.T

            if x_test.shape[0] > 0:
                if x_train_valid.shape[1] > 0:
                    t0 = time.perf_counter()
                    x_test_valid_hat = imp.transform(x_test[:, valid_cols])
                    infer_seconds += time.perf_counter() - t0
                else:
                    x_test_valid_hat = x_test[:, valid_cols]
                x_test_hat = np.full_like(x_test, np.nan, dtype=float)
                x_test_hat[:, valid_cols] = x_test_valid_hat
                x_test_hat = np.nan_to_num(x_test_hat, nan=0.0)
                out_test[:, st:ed_test] = x_test_hat.T

        return out_train, out_val, out_test, float(train_seconds), float(infer_seconds)

    train_hat, val_hat, test_hat, train_seconds, infer_seconds = _fit_transform_by_chunk(
        split_masked["train"][..., 0],
        split_masked["val"][..., 0],
        split_masked["test"][..., 0],
        model_name,
        chunk_steps=knn_chunk_steps if model_name == "knn" else mice_chunk_steps,
    )
    return {
        "train": rmse_on_missing_2d(train_hat, split_y["train"], split_masks["train"]),
        "val": rmse_on_missing_2d(val_hat, split_y["val"], split_masks["val"]),
        "test": rmse_on_missing_2d(test_hat, split_y["test"], split_masks["test"]),
    }, {
        "train_seconds": train_seconds,
        "infer_seconds": infer_seconds,
        "total_seconds": float(train_seconds + infer_seconds),
    }


def _build_pypots_model(
    model_name: str,
    n_steps: int,
    n_features: int,
    epochs: int,
    batch_size: int,
    verbose: bool,
    device,
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


def _run_pypots_deep_on_splits(
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
) -> tuple[dict[str, float], dict[str, float]]:
    d = _prepare_window_data(split_y, split_masked, split_masks, window_size, max_windows)
    model = _build_pypots_model(
        model_name,
        n_steps=d["y_train_masked_win"].shape[1],
        n_features=d["y_train_masked_win"].shape[2],
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        device=device,
    )
    t0 = time.perf_counter()
    model.fit({"X": d["y_train_masked_win"]}, val_set={"X": d["y_val_masked_win"], "X_ori": d["y_val_win"]})
    train_seconds = time.perf_counter() - t0

    def _predict_with_device_fallback(x: np.ndarray) -> np.ndarray:
        try:
            return model.predict({"X": x})["imputation"]
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
            return model.predict({"X": x})["imputation"]

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


def run_baseline_on_splits(
    model_name: str,
    split_y: dict,
    split_masked: dict,
    split_masks: dict,
    device,
    window_size: int,
    epochs: int,
    batch_size: int,
    max_windows: int | None,
    verbose: bool,
    knn_chunk_steps: int,
    mice_chunk_steps: int,
    knn_neighbors: int,
    mice_max_iter: int,
    mice_tol: float,
    mice_quiet_warnings: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    model_name = model_name.lower()
    fast_y, fast_masked, fast_masks = _truncate_splits_for_fast_run(
        split_y, split_masked, split_masks, window_size, max_windows
    )
    if model_name == "locf":
        return _run_locf_on_splits(fast_y, fast_masked, fast_masks, device)
    if model_name in {"saits", "grud", "usgan", "itransformer"}:
        return _run_pypots_deep_on_splits(
            model_name,
            fast_y,
            fast_masked,
            fast_masks,
            window_size,
            epochs,
            batch_size,
            max_windows,
            verbose,
            device,
        )
    if model_name in {"knn", "mice"}:
        return _run_sklearn_on_splits(
            fast_y,
            fast_masked,
            fast_masks,
            model_name,
            knn_chunk_steps=knn_chunk_steps,
            mice_chunk_steps=mice_chunk_steps,
            knn_neighbors=knn_neighbors,
            mice_max_iter=mice_max_iter,
            mice_tol=mice_tol,
            mice_quiet_warnings=mice_quiet_warnings,
        )
    if model_name == "vcaan":
        return run_vcaan_on_splits(fast_y, fast_masked, fast_masks)
    raise ValueError(f"Unknown model: {model_name}")
