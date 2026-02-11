from pathlib import Path

import numpy as np
import pandas as pd


def load_time_index(project_root: Path) -> pd.DatetimeIndex:
    df_all = pd.read_parquet(project_root / "data" / "processed" / "all_data.parquet")
    if not isinstance(df_all.index, pd.DatetimeIndex):
        raise ValueError("all_data.parquet index must be DatetimeIndex")
    if not df_all.index.is_monotonic_increasing:
        raise ValueError("DatetimeIndex must be monotonically increasing")
    return df_all.index


def split_by_datetime(
    ground_y: np.ndarray,
    time_index: pd.DatetimeIndex,
) -> dict[str, np.ndarray]:
    if ground_y.shape[1] != len(time_index):
        raise ValueError(
            f"ground_y timesteps ({ground_y.shape[1]}) != index length ({len(time_index)})"
        )

    tz = time_index.tz if time_index.tz is not None else None
    train_start = pd.Timestamp("2023-01-01 00:00:00", tz=tz)
    train_end = pd.Timestamp("2023-12-31 23:50:00", tz=tz)
    val_start = pd.Timestamp("2024-01-01 00:00:00", tz=tz)
    val_end = pd.Timestamp("2024-06-30 23:50:00", tz=tz)
    test_start = pd.Timestamp("2024-07-01 00:00:00", tz=tz)
    test_end = pd.Timestamp("2024-12-31 23:50:00", tz=tz)

    m_train = (time_index >= train_start) & (time_index <= train_end)
    m_val = (time_index >= val_start) & (time_index <= val_end)
    m_test = (time_index >= test_start) & (time_index <= test_end)
    if (m_train.astype(int) + m_val.astype(int) + m_test.astype(int)).max() > 1:
        raise ValueError("Time split overlap detected")

    split = {
        "train": ground_y[:, m_train],
        "val": ground_y[:, m_val],
        "test": ground_y[:, m_test],
    }
    if any(v.shape[1] == 0 for v in split.values()):
        raise ValueError("One of splits is empty. Check datetime index coverage.")
    return split


def reshape_windows_2d(y_2d: np.ndarray, window_size: int) -> np.ndarray:
    stations, timesteps = y_2d.shape
    usable_steps = (timesteps // window_size) * window_size
    if usable_steps == 0:
        raise ValueError(
            f"window_size={window_size} is larger than available timesteps={timesteps}"
        )
    y_trim = y_2d[:, :usable_steps]
    y_win = y_trim.reshape(stations, -1, window_size).reshape(-1, window_size)
    return y_win[..., np.newaxis].astype(np.float32)


def reshape_mask_2d(mask_2d: np.ndarray, window_size: int) -> np.ndarray:
    stations, timesteps = mask_2d.shape
    usable_steps = (timesteps // window_size) * window_size
    if usable_steps == 0:
        raise ValueError(
            f"window_size={window_size} is larger than available timesteps={timesteps}"
        )
    mask_trim = mask_2d[:, :usable_steps]
    return mask_trim.reshape(stations, -1, window_size).reshape(-1, window_size)
