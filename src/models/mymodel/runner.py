from __future__ import annotations
from collections import OrderedDict
from functools import lru_cache
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from src.evaluate import rmse_on_missing_2d, rmse_on_missing_3d
from src.models.mymodel.model import STAT_Model

_CACHE_MAXSIZE = 24
_X_WIN_CACHE: OrderedDict[tuple, np.ndarray] = OrderedDict()
_SPLIT_TENSOR_CACHE: OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = OrderedDict()


def _cache_get(cache: OrderedDict, key):
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _cache_put(cache: OrderedDict, key, value) -> None:
    cache[key] = value
    cache.move_to_end(key)
    if len(cache) > _CACHE_MAXSIZE:
        cache.popitem(last=False)


@lru_cache(maxsize=1)
def _load_topo_feature_dict() -> dict:
    df_stations = pd.read_csv("data/processed/all_stations.csv")
    return {
        row["Station"]: [row["X"], row["Y"], row["Height"], row["ASPECT"]]
        for _, row in df_stations.iterrows()
    }


def build_topo_features(all_stations: list) -> torch.Tensor:
    """
    Extracts topographic features for each station in `all_stations` from the dataframe.
    Returns tensor of shape [num_stations, topo_dim] (X, Y, Z, ASPECT).
    """
    topo_dict = _load_topo_feature_dict()
    topo_list = [topo_dict[s] for s in all_stations]
    return torch.tensor(np.array(topo_list), dtype=torch.float32)


class STAT_Loss(nn.Module):
    def __init__(self, lambda_clean=0.5):
        super().__init__()
        self.lambda_clean = lambda_clean
        self.mse = nn.MSELoss()
        
    def forward(self, Y_hat, Y_clean, Y_true, mask):
        mask_sum = mask.sum()
        if mask_sum == 0:
            return torch.tensor(0.0, device=Y_hat.device, requires_grad=True)
            
        diff_final = (Y_hat - Y_true) * mask
        loss_final = (diff_final ** 2).sum() / (mask_sum + 1e-8)
        
        diff_clean = (Y_clean - Y_true) * mask
        loss_clean = (diff_clean ** 2).sum() / (mask_sum + 1e-8)
        
        rmse_final = torch.sqrt(loss_final + 1e-8)
        rmse_clean = torch.sqrt(loss_clean + 1e-8)
        
        return rmse_final + self.lambda_clean * rmse_clean


def prepare_spatial_windows(y_2d: np.ndarray, window_size: int, max_windows: int = None) -> np.ndarray:
    """
    Reshapes [Stations, Timesteps] -> [NumWindows, Stations, WindowSize]
    Preserves the spatial dimension!
    """
    stations, timesteps = y_2d.shape
    usable_steps = (timesteps // window_size) * window_size
    if usable_steps == 0:
        raise ValueError(f"window_size={window_size} is larger than timesteps={timesteps}")
        
    y_trim = y_2d[:, :usable_steps] # [S, Usable_T]
    
    # [S, Num_Windows, WindowSize]
    y_reshape = y_trim.reshape(stations, -1, window_size)
    
    # Transpose to: [Num_Windows, S, WindowSize]
    y_win = y_reshape.transpose(1, 0, 2)
    
    if max_windows is not None:
        y_win = y_win[:max(1, max_windows)]
        
    return y_win.astype(np.float32)


def build_spatial_dataloader(split_y: np.ndarray, split_masked: np.ndarray, split_masks: np.ndarray, 
                             X_win: np.ndarray, wind_dir_idx: int, 
                             window_size: int, batch_size: int, shuffle: bool, max_windows: int = None,
                             use_pin_memory: bool = False):
    """
    ground_X_split: [Stations, Timesteps, Covariates]
    """
    cache_key = (
        id(split_y), split_y.shape,
        id(split_masked), split_masked.shape,
        id(split_masks), split_masks.shape,
        id(X_win), X_win.shape,
        wind_dir_idx, window_size, max_windows,
    )
    cached = _cache_get(_SPLIT_TENSOR_CACHE, cache_key)
    if cached is None:
        # 1. Target and Masks
        Y_true_win = prepare_spatial_windows(split_y, window_size, max_windows)  # [B, S, W]

        # split_masked usually has 3rd dim as 1 (from simulate_missingness). Squeeze it if needed.
        if split_masked.ndim == 3 and split_masked.shape[-1] == 1:
            split_masked = split_masked.squeeze(-1)
        Y_raw_win = prepare_spatial_windows(split_masked, window_size, max_windows)  # [B, S, W]
        mask_win = prepare_spatial_windows(split_masks, window_size, max_windows)  # [B, S, W]

        wind_dir_win = np.ascontiguousarray(X_win[..., wind_dir_idx])  # [B, S, W]
        covariates_win = np.ascontiguousarray(X_win)  # [B, S, W, C]

        # Convert to CPU tensors once; HPO trials can reuse these tensors directly.
        cached = (
            torch.from_numpy(np.ascontiguousarray(Y_raw_win)).float(),
            torch.from_numpy(np.ascontiguousarray(mask_win)).float(),
            torch.from_numpy(covariates_win).float(),
            torch.from_numpy(wind_dir_win).float(),
            torch.from_numpy(np.ascontiguousarray(Y_true_win)).float(),
        )
        _cache_put(_SPLIT_TENSOR_CACHE, cache_key, cached)

    t_Y_raw, t_mask, t_cov, t_wdir, t_Y_true = cached
    
    dataset = TensorDataset(t_Y_raw, t_mask, t_cov, t_wdir, t_Y_true)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=use_pin_memory,
        persistent_workers=False,
    )


def evaluate_mymodel(model, data_loader, device):
    model.eval()
    mse_sum = torch.zeros((), device=device)
    count = torch.zeros((), device=device)
    station_ids = torch.arange(model.num_stations, device=device)
    with torch.no_grad():
        for batch in data_loader:
            Y_raw, mask, cov, wdir, Y_true = [b.to(device) for b in batch]
            
            # The station IDs and Topo are handled as static attributes or passed in
            # topo_batch: [B, S, W, 4]
            # Topography placeholder for hyper-network projection input.
            B, S, W = Y_raw.shape
            topo_batch = Y_raw.new_zeros(B, S, W, 4) # Placeholder for hypernet
            
            Y_hat, _ = model(Y_raw, mask, cov, topo_batch, wdir, station_ids)
            
            diff = (Y_hat - Y_true) * mask
            mse_sum = mse_sum + (diff ** 2).sum()
            count = count + mask.sum()
                
            # del variables to allow reuse, but avoid empty_cache sync here
            del Y_raw, mask, cov, wdir, Y_true, Y_hat, topo_batch
                
    count_value = float(count.item())
    if count_value <= 0:
        return float('inf')
    return float(torch.sqrt(mse_sum / (count + 1e-8)).item())


def run_mymodel_on_splits(
    split_y: dict,
    split_masked: dict,
    split_masks: dict,
    ground_X_splits: dict, # Added to prevent duplicate reloading
    all_stations: list,    # Added
    vars_info: dict,       # Added
    device,
    window_size: int,
    epochs: int,
    batch_size: int,
    patience: int,
    max_windows: int | None,
    verbose: bool,
    hparams_override: dict | None = None,
    mode: str = "all",
    project_root: str | None = None,
    run_dir: str | None = None,
    pattern: str = "mcar",
    pi: float = 0.1,
) -> tuple[dict[str, float], dict[str, float]]:
    
    # 1. Load static topography and covariates
    wind_dir_idx = vars_info['x'].index('wind_direction') if 'wind_direction' in vars_info['x'] else 0
    num_covariates = len(vars_info['x'])
    
    # Extract LV95 Topo Features: [S, 4]
    topo_features = build_topo_features(all_stations).to(device)
    num_stations = len(all_stations)
    
    # ground_X_splits already contains time-sliced X for each split
    # ground_X_splits[split_name] is [S, C, T]. We need it to be [Num_Windows, S, W, C]
    X_win_splits = {}
    if mode in ("train", "all"):
        splits_to_process = ['train', 'val', 'test']
    elif mode == "hpo":
        splits_to_process = ['train', 'val']
    else:
        splits_to_process = ['test']
        
    for split_name in splits_to_process:
        X_split = ground_X_splits[split_name] # [S, C, T]
        x_key = (id(X_split), X_split.shape, window_size, max_windows)
        cached_x = _cache_get(_X_WIN_CACHE, x_key)
        if cached_x is None:
            S, C, T = X_split.shape
            usable_steps = (T // window_size) * window_size
            X_trim = X_split[:, :, :usable_steps] # [S, C, UsableT]
            # reshape to [S, C, Num_Windows, W]
            X_reshape = X_trim.reshape(S, C, -1, window_size)
            # transpose to [Num_Windows, S, W, C]
            X_win = X_reshape.transpose(2, 0, 3, 1)
            if max_windows is not None:
                X_win = X_win[:max(1, max_windows)]
            cached_x = np.ascontiguousarray(X_win, dtype=np.float32)
            _cache_put(_X_WIN_CACHE, x_key, cached_x)
        X_win_splits[split_name] = cached_x
        
    # 2. Build DataLoaders
    # Enforce a maximum batch size for mymodel to prevent MPS OOM during Transformer autograd
    effective_batch_size = min(batch_size, 8)
    
    loaders = {}
    use_pin_memory = (str(device) == "cuda")
    for split_name in splits_to_process:
        loaders[split_name] = build_spatial_dataloader(
            split_y=split_y[split_name],
            split_masked=split_masked[split_name],
            split_masks=split_masks[split_name],
            X_win=X_win_splits[split_name],
            wind_dir_idx=wind_dir_idx,
            window_size=window_size,
            batch_size=effective_batch_size,
            shuffle=(split_name == 'train'),
            max_windows=max_windows,
            use_pin_memory=use_pin_memory,
        )

    import json
    from pathlib import Path
    
    # Assumes run_dir is passed correctly when mode separates train and test
    _run_dir = Path(run_dir) if run_dir else (Path(project_root) if project_root else Path.cwd())
    model_save_dir = _run_dir / "saved_models" / "mymodel"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_save_dir / f"mymodel_{pattern}_{pi}.pt"
    hparams_path = model_save_dir / f"mymodel_{pattern}_{pi}_hparams.json"

    # 3. Model Initialization
    if mode == "test" and hparams_override is None:
        if hparams_path.exists():
            with open(hparams_path, "r") as f:
                saved_config = json.load(f)
                hparams_override = saved_config.get("best_hparams", None)

    # Applied optimized default parameters from Optuna run
    d_model = hparams_override.get('d_model', 128) if hparams_override else 128
    l_lags = hparams_override.get('l_lags', 2) if hparams_override else 2
    num_layers = hparams_override.get('num_layers', 1) if hparams_override else 1
    nhead = hparams_override.get('nhead', 4) if hparams_override else 4
    
    model = STAT_Model(
        num_stations=num_stations,
        num_covariates=num_covariates,
        seq_len=window_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        l_lags=l_lags,
        topo_dim=4, # X, Y, Z, ASPECT
        device=device
    ).to(device)
    
    # Warm up priors cache once to avoid first-step latency spikes.
    with torch.no_grad():
        dummy_wind = torch.zeros(1, num_stations, window_size, device=device)
        dummy_topo = topo_features.unsqueeze(0).unsqueeze(2).expand(1, -1, window_size, -1)
        model.priors(dummy_topo, dummy_wind, dummy_wind)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = STAT_Loss(lambda_clean=0.5)
    station_ids = torch.arange(num_stations, device=device)
    zero_topo_template = torch.zeros(1, num_stations, window_size, 4, device=device)
    
    # Train or Load logic
    t0 = time.perf_counter()
    best_val_rmse = float('inf')
    early_stop_counter = 0
    
    if mode in ("train", "all", "hpo"):
        for epoch in range(epochs):
            model.train()
            train_loss_sum = torch.zeros((), device=device) if verbose else None
            for batch in loaders['train']:
                Y_raw, mask, cov, wdir, Y_true = [b.to(device) for b in batch]
                
                # Prepare static inputs
                B, S, W = Y_raw.shape
                # Passing a dummy with correct S dimension to satisfy HyperNetwork input projection
                # Physics priors will use their own cached matrices instead.
                topo_batch = zero_topo_template[:, :S, :W, :].expand(B, -1, -1, -1)
                
                optimizer.zero_grad()
                Y_hat, Y_clean = model(Y_raw, mask, cov, topo_batch, wdir, station_ids)
                
                loss = criterion(Y_hat, Y_clean, Y_true, mask)
                loss.backward()
                optimizer.step()
                if verbose:
                    train_loss_sum = train_loss_sum + loss.detach()
                
                # Ruthlessly prune PyTorch retained graph buffers per-batch
                # BUT don't call empty_cache inside the loop as it is too slow
                del Y_hat, Y_clean, loss, Y_raw, mask, cov, wdir, Y_true, topo_batch
                
            # Validation
            val_rmse = evaluate_mymodel(model, loaders['val'], device)
            if verbose:
                train_loss_val = float(train_loss_sum.item()) if train_loss_sum is not None else float("nan")
                print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss_val:.4f} | Val RMSE: {val_rmse:.4f}")
                
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                early_stop_counter = 0
                # Save best state dict
                if mode != "hpo":
                    torch.save(model.state_dict(), str(model_path))
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                    break
                    
            # Explicitly force MPS to drop the VRAM cache at epoch boundaries
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
        
        # Finally, save hparams config
        if mode != "hpo":
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            hparams_to_save = {
                "model_name": "mymodel",
                "pattern": pattern,
                "pi": pi,
                "best_hparams": hparams_override or {"d_model": d_model, "l_lags": l_lags, "num_layers": num_layers, "nhead": nhead},
                "num_params": num_params,
                "epochs": epochs,
                "batch_size": batch_size,
                "window_size": window_size,
            }
            with open(str(hparams_path), "w") as f:
                json.dump(hparams_to_save, f, indent=4)
                
            # Ensure we evaluate the best model state instead of the last epoch
            if model_path.exists():
                model.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=True), strict=False)

    else: # mode == "test"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found for inference: {model_path}. Run with '--mode train' first.")
        model.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=True), strict=False)
        
    train_seconds = time.perf_counter() - t0

    if mode == "hpo":
        # HPO only needs validation score; skip full prediction concatenation path.
        t1 = time.perf_counter()
        val_rmse = evaluate_mymodel(model, loaders["val"], device)
        infer_seconds = time.perf_counter() - t1
        timing = {
            "train_seconds": float(train_seconds),
            "infer_seconds": float(infer_seconds),
            "total_seconds": float(train_seconds + infer_seconds),
        }
        del loaders, X_win_splits, topo_features, model
        import gc
        gc.collect()
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        return {"train": np.nan, "val": float(val_rmse), "test": np.nan}, timing
    
    # Optional: Clear cache and collect garbage after training to free up MPS memory before inference
    import gc
    gc.collect()
    if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    
    # 5. Full Evaluation (Inference)
    t1 = time.perf_counter()
    
    def predict_split(loader):
        model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in loader:
                Y_raw, mask, cov, wdir, _ = [b.to(device) for b in batch]
                B, S, W = Y_raw.shape
                # topo_batch: satisfy hypernet input projection
                topo_batch = zero_topo_template[:, :S, :W, :].expand(B, -1, -1, -1)
                
                Y_hat, _ = model(Y_raw, mask, cov, topo_batch, wdir, station_ids)
                all_preds.append(Y_hat.cpu().numpy())
                
                # Prevent inference OOM
                del Y_hat, Y_raw, mask, cov, wdir, topo_batch
                
        preds_concat = np.concatenate(all_preds, axis=0) # [B, S, W]
        # Reshape to [S, B*W] to match original flattened shape and mask format
        S_dim = preds_concat.shape[1]
        preds_2d = preds_concat.transpose(1, 0, 2).reshape(S_dim, -1)
        
        del all_preds, preds_concat
        import gc
        gc.collect()
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
            
        return preds_2d
        
    train_pred_2d = predict_split(loaders['train']) if mode in ("train", "all") else np.empty((split_y['train'].shape[0], 0))
    val_pred_2d = predict_split(loaders['val']) if mode in ("train", "all", "hpo") else np.empty((split_y['val'].shape[0], 0))
    test_pred_2d = predict_split(loaders['test']) if mode in ("test", "all") else np.empty((split_y['test'].shape[0], 0))
    
    infer_seconds = time.perf_counter() - t1
    
    # 极致清理：在预测结束后，由于已经拿到了 numpy 结果，可以销毁所有 DataLoader 和 Tensor
    del loaders, X_win_splits, topo_features, model
    import gc
    gc.collect()
    if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()

    # 6. Calculate True RMSE
    # Need to match the exact truncated length that predict_split returned
    # because the DataLoader drops the last partial window
    def truncate_to_match(original_arr, pred_arr):
        pred_len = pred_arr.shape[1]
        return original_arr[:, :pred_len]
        
    rmses = {
        "train": rmse_on_missing_2d(train_pred_2d, 
                                    truncate_to_match(split_y['train'], train_pred_2d), 
                                    truncate_to_match(split_masks['train'], train_pred_2d)) if mode in ("train", "all") else np.nan,
        "val": rmse_on_missing_2d(val_pred_2d, 
                                  truncate_to_match(split_y['val'], val_pred_2d), 
                                  truncate_to_match(split_masks['val'], val_pred_2d)) if mode in ("train", "all", "hpo") else np.nan,
        "test": rmse_on_missing_2d(test_pred_2d, 
                                   truncate_to_match(split_y['test'], test_pred_2d), 
                                   truncate_to_match(split_masks['test'], test_pred_2d)) if mode in ("test", "all") else np.nan,
    }
    
    timing = {
        "train_seconds": float(train_seconds),
        "infer_seconds": float(infer_seconds),
        "total_seconds": float(train_seconds + infer_seconds),
    }
    
    return rmses, timing
