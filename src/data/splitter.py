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
    train_start: str = "2023-01-01",
    train_end: str = "2023-01-31",
    val_start: str = "2023-02-01",
    val_end: str = "2023-02-28",
    test_start: str = "2023-03-01",
    test_end: str = "2023-03-31",
) -> dict[str, np.ndarray]:
    if ground_y.shape[1] != len(time_index):
        raise ValueError(
            f"ground_y timesteps ({ground_y.shape[1]}) != index length ({len(time_index)})"
        )

    tz = time_index.tz if time_index.tz is not None else None
    
    # 支持 yyyy-mm-dd 等输入格式，添加当天截止时间 23:50:00 的宽容处理以防丢失末尾数据
    def _parse_ts(ts_str: str, is_end: bool = False) -> pd.Timestamp:
        ts = pd.Timestamp(ts_str, tz=tz)
        if len(ts_str) <= 10 and is_end:
            ts = ts.replace(hour=23, minute=50, second=0)
        elif len(ts_str) <= 10 and not is_end:
            ts = ts.replace(hour=0, minute=0, second=0)
        return ts

    ts_train_start = _parse_ts(train_start, False)
    ts_train_end = _parse_ts(train_end, True)
    ts_val_start = _parse_ts(val_start, False)
    ts_val_end = _parse_ts(val_end, True)
    ts_test_start = _parse_ts(test_start, False)
    ts_test_end = _parse_ts(test_end, True)

    m_train = (time_index >= ts_train_start) & (time_index <= ts_train_end)
    m_val = (time_index >= ts_val_start) & (time_index <= ts_val_end)
    m_test = (time_index >= ts_test_start) & (time_index <= ts_test_end)
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
