import numpy as np


def rmse_on_missing_2d(y_hat, y_true, mask):
    """2D RMSE on missing positions only.

    y_hat/y_true: (S, T)
    mask: (S, T), where 0 means missing.
    """
    missing_idx = mask == 0
    if not np.any(missing_idx):
        return float("nan")
    err = y_hat - y_true
    return float(np.sqrt(np.mean((err[missing_idx]) ** 2)))


def rmse_on_missing_3d(y_hat, y_true, mask_2d):
    """3D RMSE on missing positions only.

    y_hat/y_true: (N, W, 1)
    mask_2d: (N, W), where 0 means missing.
    """
    missing_idx = mask_2d == 0
    if not np.any(missing_idx):
        return float("nan")
    err = y_hat[..., 0] - y_true[..., 0]
    return float(np.sqrt(np.mean((err[missing_idx]) ** 2)))


def evaluate_rmse(y_hat, ground_y, mask):
    # Backward-compatible alias used by old scripts.
    return round(rmse_on_missing_2d(y_hat, ground_y, mask), 4)
