import numpy as np

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
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


def _run_locf_on_splits(split_y: dict, split_masked: dict, split_masks: dict, device) -> dict[str, float]:
    model = LOCF(device=device)
    res = {}
    for split_name in ["train", "val", "test"]:
        y_hat = model.impute({"X": split_masked[split_name]}).squeeze()
        res[split_name] = rmse_on_missing_2d(y_hat, split_y[split_name], split_masks[split_name])
    return res


def _run_sklearn_on_splits(split_y: dict, split_masked: dict, split_masks: dict, model_name: str) -> dict[str, float]:
    # Chunked imputation to avoid very slow behavior on long time series.
    def _chunk_impute(y_obs_2d: np.ndarray, algo: str, chunk_steps: int = 2016) -> np.ndarray:
        # y_obs_2d shape: (S, T)
        s, t = y_obs_2d.shape
        out = np.empty((s, t), dtype=float)
        for st in range(0, t, chunk_steps):
            ed = min(st + chunk_steps, t)
            x_chunk = y_obs_2d[:, st:ed].T  # (time, station)
            valid_cols = ~np.all(np.isnan(x_chunk), axis=0)
            x_valid = x_chunk[:, valid_cols]
            if algo == "knn":
                imp = KNNImputer(n_neighbors=5)
            elif algo == "mice":
                imp = IterativeImputer(random_state=42, sample_posterior=False, max_iter=10)
            else:
                raise ValueError(f"Unknown sklearn baseline {algo}")
            if x_valid.shape[1] > 0:
                x_valid_hat = imp.fit_transform(x_valid)
            else:
                x_valid_hat = x_valid
            x_hat = np.full_like(x_chunk, np.nan, dtype=float)
            x_hat[:, valid_cols] = x_valid_hat
            # For all-NaN station-columns in this chunk, fallback to 0 to keep stable shape.
            x_hat = np.nan_to_num(x_hat, nan=0.0)
            out[:, st:ed] = x_hat.T
        return out

    train_hat = _chunk_impute(split_masked["train"][..., 0], model_name)
    val_hat = _chunk_impute(split_masked["val"][..., 0], model_name)
    test_hat = _chunk_impute(split_masked["test"][..., 0], model_name)
    return {
        "train": rmse_on_missing_2d(train_hat, split_y["train"], split_masks["train"]),
        "val": rmse_on_missing_2d(val_hat, split_y["val"], split_masks["val"]),
        "test": rmse_on_missing_2d(test_hat, split_y["test"], split_masks["test"]),
    }


def _build_pypots_model(model_name: str, n_steps: int, n_features: int, epochs: int, batch_size: int, verbose: bool):
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
            verbose=verbose,
        )
    if model_name == "grud":
        return GRUD(
            n_steps=n_steps,
            n_features=n_features,
            rnn_hidden_size=64,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )
    if model_name == "usgan":
        return USGAN(
            n_steps=n_steps,
            n_features=n_features,
            rnn_hidden_size=64,
            epochs=epochs,
            batch_size=batch_size,
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
) -> dict[str, float]:
    d = _prepare_window_data(split_y, split_masked, split_masks, window_size, max_windows)
    model = _build_pypots_model(
        model_name,
        n_steps=d["y_train_masked_win"].shape[1],
        n_features=d["y_train_masked_win"].shape[2],
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )
    model.fit({"X": d["y_train_masked_win"]}, val_set={"X": d["y_val_masked_win"], "X_ori": d["y_val_win"]})
    train_pred = model.predict({"X": d["y_train_masked_win"]})["imputation"]
    val_pred = model.predict({"X": d["y_val_masked_win"]})["imputation"]
    test_pred = model.predict({"X": d["y_test_masked_win"]})["imputation"]
    return {
        "train": rmse_on_missing_3d(train_pred, d["y_train_win"], d["train_mask_win"]),
        "val": rmse_on_missing_3d(val_pred, d["y_val_win"], d["val_mask_win"]),
        "test": rmse_on_missing_3d(test_pred, d["y_test_win"], d["test_mask_win"]),
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
) -> dict[str, float]:
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
        )
    if model_name in {"knn", "mice"}:
        return _run_sklearn_on_splits(fast_y, fast_masked, fast_masks, model_name)
    if model_name == "vcaan":
        return run_vcaan_on_splits(fast_y, fast_masked, fast_masks)
    raise ValueError(f"Unknown model: {model_name}")
