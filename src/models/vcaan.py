import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.evaluate import rmse_on_missing_2d


def _safe_corrcoef(y_hat: np.ndarray) -> np.ndarray:
    # y_hat shape: (S, T)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(y_hat)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(c, 0.0)
    return c


def _build_spatial_feature(y_hat: np.ndarray, corr: np.ndarray, t: int) -> np.ndarray:
    # Weighted neighborhood signal per station at timestamp t.
    denom = np.sum(np.abs(corr), axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return (corr @ y_hat[:, t : t + 1] / denom).reshape(-1)


def _locf_prefill(y_obs: np.ndarray) -> np.ndarray:
    # y_obs shape: (S, T), NaN indicates missing.
    # LOCF + BOCF as robust warm start.
    out = np.empty_like(y_obs, dtype=float)
    for s in range(y_obs.shape[0]):
        ser = pd.Series(y_obs[s])
        filled = ser.ffill().bfill().fillna(0.0).to_numpy(dtype=float)
        out[s] = filled
    return out


def _vcaan_single(y_obs: np.ndarray, mask: np.ndarray, max_iter: int = 8, tol: float = 1e-4) -> np.ndarray:
    # y_obs shape: (S, T), contains NaN at missing positions.
    # 1) warm-start with LOCF (requested baseline setup).
    y_hat = _locf_prefill(y_obs)

    stations, timesteps = y_hat.shape
    if timesteps < 3:
        return y_hat

    # 2) iterative temporal-spatial regression refinement.
    for _ in range(max_iter):
        y_prev = y_hat.copy()
        corr = _safe_corrcoef(y_hat)

        x_train = []
        y_train = []
        x_test = []
        test_idx = []

        for t in range(1, timesteps - 1):
            sp_prev = _build_spatial_feature(y_hat, corr, t - 1)
            sp_curr = _build_spatial_feature(y_hat, corr, t)
            sp_next = _build_spatial_feature(y_hat, corr, t + 1)
            for s in range(stations):
                feat = [
                    y_hat[s, t - 1],
                    y_hat[s, t + 1],
                    sp_prev[s],
                    sp_curr[s],
                    sp_next[s],
                ]
                if mask[s, t] == 1:
                    x_train.append(feat)
                    y_train.append(y_hat[s, t])
                else:
                    x_test.append(feat)
                    test_idx.append((s, t))

        if len(x_train) == 0 or len(x_test) == 0:
            break

        reg = LinearRegression()
        reg.fit(np.asarray(x_train, dtype=float), np.asarray(y_train, dtype=float))
        preds = reg.predict(np.asarray(x_test, dtype=float))

        for (s, t), pred in zip(test_idx, preds):
            y_hat[s, t] = float(pred)

        diff = np.nanmean(np.abs(y_hat[mask == 0] - y_prev[mask == 0]))
        if np.isnan(diff) or diff < tol:
            break

    return y_hat


def run_vcaan_on_splits(
    split_y: dict,
    split_masked: dict,
    split_masks: dict,
) -> tuple[dict[str, float], dict[str, float]]:
    # This is a practical VCAAN baseline adaptation:
    # temporal-spatial iterative regression initialized by LOCF prefill.
    res = {}
    split_runtime = {}
    for split_name in ["train", "val", "test"]:
        t0 = time.perf_counter()
        y_obs = split_masked[split_name][..., 0]
        mask = split_masks[split_name]
        y_hat = _vcaan_single(y_obs, mask)
        res[split_name] = rmse_on_missing_2d(y_hat, split_y[split_name], mask)
        split_runtime[split_name] = time.perf_counter() - t0
    timing = {
        "train_seconds": float(split_runtime["train"]),
        "infer_seconds": float(split_runtime["val"] + split_runtime["test"]),
        "total_seconds": float(sum(split_runtime.values())),
    }
    return res, timing
