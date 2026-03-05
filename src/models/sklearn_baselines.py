import time
import warnings
import numpy as np

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import IterativeImputer, KNNImputer

from src.evaluate import rmse_on_missing_2d


def run_sklearn_on_splits(
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
    mode: str = "all",
    project_root: str | None = None,
    run_dir: str | None = None,
    pattern: str = "mcar",
    pi: float = 0.1,
) -> tuple[dict[str, float], dict[str, float]]:
    """Run KNN or MICE baseline on splits using chunked imputation to handle long series."""

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
                raise ValueError(f"Unknown sklearn baseline {algo}")

            # For chunked ML models, we'll only do inference if mode == 'test'.
            # However, sklearn IterativeImputer doesn't easily persist all chunk states in this script.
            # We will just run fit_transform anyway because MICE/KNN are relatively fast and memory-bound.
            # But we skip updating the test set if mode == 'train'.
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

    import json
    from pathlib import Path
    _run_dir = Path(run_dir) if run_dir else (Path(project_root) if project_root else Path.cwd())
    model_save_dir = _run_dir / "saved_models" / model_name
    model_save_dir.mkdir(parents=True, exist_ok=True)
    hparams_path = model_save_dir / f"{model_name}_{pattern}_{pi}_hparams.json"
    
    # We dump hparams so that the export script can find it. ML has 0 learnable weights.
    if mode in ("train", "all"):
        hparams_to_save = {
            "model_name": model_name,
            "pattern": pattern,
            "pi": pi,
            "best_hparams": {"knn_neighbors": knn_neighbors, "mice_max_iter": mice_max_iter, "mice_tol": mice_tol},
            "num_params": 0,
            "epochs": mice_max_iter if model_name == "mice" else 0,
        }
        with open(hparams_path, "w") as f:
            json.dump(hparams_to_save, f, indent=4)

    train_hat, val_hat, test_hat, train_seconds, infer_seconds = _fit_transform_by_chunk(
        split_masked["train"][..., 0],
        split_masked["val"][..., 0],
        split_masked["test"][..., 0],
        model_name,
        chunk_steps=knn_chunk_steps if model_name == "knn" else mice_chunk_steps,
    )
    return {
        "train": rmse_on_missing_2d(train_hat, split_y["train"], split_masks["train"]) if mode in ("train", "all") else np.nan,
        "val": rmse_on_missing_2d(val_hat, split_y["val"], split_masks["val"]) if mode in ("train", "all") else np.nan,
        "test": rmse_on_missing_2d(test_hat, split_y["test"], split_masks["test"]) if mode in ("test", "all") else np.nan,
    }, {
        "train_seconds": train_seconds,
        "infer_seconds": infer_seconds,
        "total_seconds": float(train_seconds + infer_seconds),
    }
