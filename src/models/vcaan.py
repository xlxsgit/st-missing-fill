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


    pass


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
        y_prev_iter = y_hat.copy()
        corr = _safe_corrcoef(y_hat)

        denom = np.sum(np.abs(corr), axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        
        # 全局计算空间特征: (S, S) @ (S, T) -> (S, T)
        spatial_all = (corr @ y_hat) / denom
        
        # 提取各个偏移量的时间切片 (S, T-2)
        y_prev_t = y_hat[:, :-2]
        y_curr_t = y_hat[:, 1:-1]
        y_next_t = y_hat[:, 2:]
        
        sp_prev_t = spatial_all[:, :-2]
        sp_curr_t = spatial_all[:, 1:-1]
        sp_next_t = spatial_all[:, 2:]
        
        mask_curr = mask[:, 1:-1]
        
        # 堆叠特征 (S, T-2, 5)
        features = np.stack([y_prev_t, y_next_t, sp_prev_t, sp_curr_t, sp_next_t], axis=-1)
        
        train_mask_1d = (mask_curr == 1)
        test_mask_1d = (mask_curr == 0)
        
        x_train = features[train_mask_1d]
        y_train = y_curr_t[train_mask_1d]
        x_test = features[test_mask_1d]

        if len(x_train) == 0 or len(x_test) == 0:
            break

        reg = LinearRegression()
        reg.fit(x_train, y_train)
        preds = reg.predict(x_test)

        # 更新原始矩阵中的缺失值预测
        y_curr_update = y_curr_t.copy()
        y_curr_update[test_mask_1d] = preds
        y_hat[:, 1:-1] = y_curr_update

        diff = np.nanmean(np.abs(y_hat[mask == 0] - y_prev_iter[mask == 0]))
        if np.isnan(diff) or diff < tol:
            break

    return y_hat


def run_vcaan_on_splits(
    split_y: dict,
    split_masked: dict,
    split_masks: dict,
    mode: str = "all",
) -> tuple[dict[str, float], dict[str, float]]:
    # This is a practical VCAAN baseline adaptation:
    # temporal-spatial iterative regression initialized by LOCF prefill.
    res = {}
    split_runtime = {}
    for split_name in ["train", "val", "test"]:
        if mode == "test" and split_name != "test":
            res[split_name] = np.nan
            split_runtime[split_name] = 0.0
            continue
            
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
