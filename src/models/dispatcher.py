from src.models.utils import truncate_splits_for_fast_run
from src.models.statistical_baselines import run_locf_on_splits
from src.models.sklearn_baselines import run_sklearn_on_splits
from src.models.pypots_baselines import run_pypots_deep_on_splits
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
    "mymodel",
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
    patience: int,
    max_windows: int | None,
    verbose: bool,
    knn_chunk_steps: int,
    mice_chunk_steps: int,
    knn_neighbors: int,
    mice_max_iter: int,
    mice_tol: float,
    mice_quiet_warnings: bool,
    hparams_override: dict | None = None,
    mode: str = "all",
    project_root: str | None = None,
    run_dir: str | None = None,
    pattern: str = "mcar",
    pi: float = 0.1,
    ground_X_splits: dict | None = None,
    all_stations: list | None = None,
    vars_info: dict | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    model_name = model_name.lower()
    fast_y, fast_masked, fast_masks = truncate_splits_for_fast_run(
        split_y, split_masked, split_masks, window_size, max_windows
    )
    if model_name == "locf":
        return run_locf_on_splits(fast_y, fast_masked, fast_masks, device)
    if model_name in {"saits", "grud", "usgan", "itransformer"}:
        return run_pypots_deep_on_splits(
            model_name=model_name,
            split_y=fast_y,
            split_masked=fast_masked,
            split_masks=fast_masks,
            window_size=window_size,
            epochs=hparams_override.get("epochs", epochs) if hparams_override else epochs,
            batch_size=hparams_override.get("batch_size", batch_size) if hparams_override else batch_size,
            patience=hparams_override.get("patience", patience) if hparams_override else patience,
            max_windows=max_windows,
            verbose=verbose,
            device=device,
            hparams_override=hparams_override,
            mode=mode,
            project_root=project_root,
            run_dir=run_dir,
            pattern=pattern,
            pi=pi,
        )
    if model_name in {"knn", "mice"}:
        return run_sklearn_on_splits(
            fast_y,
            fast_masked,
            fast_masks,
            model_name,
            knn_chunk_steps=knn_chunk_steps,
            mice_chunk_steps=mice_chunk_steps,
            knn_neighbors=hparams_override.get("knn_neighbors", knn_neighbors) if hparams_override else knn_neighbors,
            mice_max_iter=hparams_override.get("mice_max_iter", mice_max_iter) if hparams_override else mice_max_iter,
            mice_tol=hparams_override.get("mice_tol", mice_tol) if hparams_override else mice_tol,
            mice_quiet_warnings=mice_quiet_warnings,
            mode=mode,
            project_root=project_root,
            run_dir=run_dir,
            pattern=pattern,
            pi=pi,
        )
    if model_name == "vcaan":
        return run_vcaan_on_splits(fast_y, fast_masked, fast_masks)
    if model_name == "mymodel":
        from src.models.mymodel.runner import run_mymodel_on_splits
        return run_mymodel_on_splits(
            split_y=fast_y,
            split_masked=fast_masked,
            split_masks=fast_masks,
            device=device,
            window_size=window_size,
            epochs=hparams_override.get("epochs", epochs) if hparams_override else epochs,
            batch_size=hparams_override.get("batch_size", batch_size) if hparams_override else batch_size,
            patience=hparams_override.get("patience", patience) if hparams_override else patience,
            max_windows=max_windows,
            verbose=verbose,
            hparams_override=hparams_override,
            mode=mode,
            project_root=project_root,
            run_dir=run_dir,
            pattern=pattern,
            pi=pi,
            ground_X_splits=ground_X_splits,
            all_stations=all_stations,
            vars_info=vars_info,
        )
    raise ValueError(f"Unknown model: {model_name}")
