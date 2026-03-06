import time

from pypots.imputation import LOCF

from src.evaluate import rmse_on_missing_2d


def run_locf_on_splits(
    split_y: dict, split_masked: dict, split_masks: dict, device
) -> tuple[dict[str, float], dict[str, float]]:
    """Run LOCF baseline on the provided splits."""
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
