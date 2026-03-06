import numpy as np

from src.data.splitter import reshape_mask_2d, reshape_windows_2d


def truncate_splits_for_fast_run(
    split_y: dict,
    split_masked: dict,
    split_masks: dict,
    window_size: int,
    max_windows: int | None,
):
    """Truncate the time steps of all splits for fast smoke testing / debugging."""
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


def prepare_window_data(
    split_y: dict,
    split_masked: dict,
    split_masks: dict,
    window_size: int,
    max_windows: int | None,
):
    """Reshape 2D splits into 3D windowed arrays suitable for PyPOTS models."""
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
