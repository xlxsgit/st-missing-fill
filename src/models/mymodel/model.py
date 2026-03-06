import torch
import torch.nn as nn
from src.models.mymodel.components import PhysicalAlignmentPriors, STAT_HyperNetwork

class STAT_Model(nn.Module):
    """
    Spatio-Temporal Aligned Transformer (STAT) model for Wind Speed Imputation.
    Operates Non-Iteratively (One-Shot Execute) via a Bidirectional STAR physics equation
    guided by a Transformer Hyper-Network.
    """
    def __init__(self, num_stations, num_covariates, seq_len=144, d_model=128, nhead=8, num_layers=4, 
                 l_lags=3, topo_dim=4, dropout=0.1, device='cpu'):
        super().__init__()
        self.device = device
        self.l_lags = l_lags
        self.num_stations = num_stations
        
        # in_features: 
        #   1 (Y_raw) + 1 (Mask) + 1 (Wind Dir) 
        #   + num_covariates (e.g., Temp, Press, Humidity, Prec, Sun) 
        #   + topo_dim (e.g., X, Y, Z, Aspect)
        self.in_features = 1 + 1 + 1 + num_covariates + topo_dim
        
        self.hyper_net = STAT_HyperNetwork(
            num_stations=num_stations,
            in_features=self.in_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            l_lags=l_lags,
            num_covariates=num_covariates,
            dropout=dropout
        )
        
        self.priors = PhysicalAlignmentPriors(
            num_stations=num_stations,
            device=device
        )
        
        # We need an array of the l_lags we care about: [-L, ..., -1, 1, ..., L]
        l_arr = [i for i in range(-l_lags, l_lags + 1) if i != 0]
        self.l_values = torch.tensor(l_arr, dtype=torch.float32, device=device)
        
    def forward(self, Y_raw, mask, covariates, topo_features, wind_dir, station_ids):
        """
        Inputs:
            Y_raw:            [B, S, W] 
            mask:             [B, S, W] (1 for observed, 0 for missing)
            covariates:       [B, S, W, num_covariates]
            topo_features:    [B, S, W, topo_dim] (X, Y, Z, Aspect)
            wind_dir:         [B, S, W]
            station_ids:      [S]
        """
        B, S, W = Y_raw.shape
        
        # Replace NaNs in Y_raw with 0 for the network
        Y_safe = torch.where(torch.isnan(Y_raw), torch.zeros_like(Y_raw), Y_raw)
        
        # Batch-wise Z-score normalization for high-variance physics features
        # Prevents Neural Network gradient explosion and bounds the arithmetic coefficient scales.
        eps = 1e-6
        cov_mean = covariates.mean(dim=(0, 1, 2), keepdim=True)
        cov_std = covariates.std(dim=(0, 1, 2), keepdim=True)
        cov_std = torch.clamp(cov_std, min=eps)
        cov_norm = (covariates - cov_mean) / cov_std
        
        topo_mean = topo_features.mean(dim=(0, 1, 2), keepdim=True)
        topo_std = topo_features.std(dim=(0, 1, 2), keepdim=True)
        topo_std = torch.clamp(topo_std, min=eps)
        topo_norm = (topo_features - topo_mean) / topo_std
        
        # 1. Prepare Input for Hyper-network utilizing normalized values
        x_in = torch.cat([
            Y_safe.unsqueeze(-1), 
            mask.unsqueeze(-1), 
            wind_dir.unsqueeze(-1),
            cov_norm, 
            topo_norm
        ], dim=-1) # [B, S, W, in_features]
        
        # 2. Hyper-Network Generates Coefficients & Representation
        Y_clean, A_coeff, beta_coeff, B_coeff, gamma_coeff = self.hyper_net(x_in.to(self.device), station_ids.to(self.device))
        
        # 3. Form Hybrid Clean Value Y_star
        # Observed ones use Y_raw, missing ones use the transformer generated Y_clean
        Y_star = mask * Y_safe + (1 - mask) * Y_clean # [B, S, W]
        
        # 4. Generate Physics Priors
        alpha_tilde, tau = self.priors(topo_features.to(self.device), Y_star, wind_dir.to(self.device))
        
        # Omega for l=0: [B, W, S_tgt, S_src, 1]
        zero_lag = torch.tensor([0.0], dtype=torch.float32, device=self.device)
        omega_0 = self.priors.compute_omega(tau, zero_lag)
        
        # --- Assembling the Non-Iterative STAR One-Shot Equation ---
        
        # We need to construct shifted versions of Y_star for +/- L lags
        # Y_shifted: [B, S, W, 2L]
        Y_shifted = torch.zeros(B, S, W, 2 * self.l_lags, device=self.device)
        
        # Fill Y_shifted. Simple zero-padding for out-of-bounds delays.
        for idx, l in enumerate(self.l_values.long()):
            if l < 0:
                shift = abs(l)
                Y_shifted[:, :, shift:, idx] = Y_star[:, :, :-shift]
            elif l > 0:
                shift = l
                Y_shifted[:, :, :-shift, idx] = Y_star[:, :, shift:]
                
        # A) Temporal Bidirectional Smoothing: Sum_L (A * Y_star(t+l))
        # A_coeff: [B, S, W, 2L]
        term_temporal = torch.sum(A_coeff * Y_shifted, dim=-1) # [B, S, W]
        
        # B) Spatio-Temporal Lags:
        # Avoid OOM on MPS by processing Window steps W in blocks!
        # This reduces Python loop overhead while keeping VRAM well below 24GB.
        WINDOW_BLOCK_SIZE = 48 
        
        spatial_lag_sum = torch.zeros(B, S, W, 2 * self.l_lags, device=self.device)
        spatial_zero_sum = torch.zeros(B, S, W, device=self.device)
        
        # We move compute_omega inside the block loop to avoid allocating 
        # a monolithic [B, W, S, S, 2L] tensor (~3GB for B=64).
        for w_start in range(0, W, WINDOW_BLOCK_SIZE):
            w_end = min(w_start + WINDOW_BLOCK_SIZE, int(W))
            W_curr = w_end - w_start
            
            # tau_block: [B, W_curr, S_tgt, S_src]
            tau_block = tau[:, w_start:w_end, :, :]
            # omega_block: [B, W_curr, S, S, 2L]
            omega_block = self.priors.compute_omega(tau_block, self.l_values)
            
            # alpha_block: [B, W_curr, S_tgt, S_src, 1]
            alpha_block = alpha_tilde[:, w_start:w_end, :, :].unsqueeze(-1)
            
            # Y_shift_block: [B, S_src, W_curr, 2L] -> [B, W_curr, S_src, 2L]
            Y_shift_block = Y_shifted[:, :, w_start:w_end, :].transpose(1, 2)
            # Reshape to broadcast: [B, W_curr, 1, S_src, 2L]
            Y_shift_block_exp = Y_shift_block.unsqueeze(2)
            
            # Vectorized multiply and sum over S_src (dim 3)
            # Result: [B, W_curr, S_tgt, 2L]
            block_lag_sum = torch.sum(omega_block * alpha_block * Y_shift_block_exp, dim=3)
            spatial_lag_sum[:, :, w_start:w_end, :] = block_lag_sum.transpose(1, 2)
            
            # Process l=0 Contemporaneous Interaction
            # alpha_0_block: [B, W_curr, S_tgt, S_src]
            alpha_0_block = alpha_tilde[:, w_start:w_end, :, :]
            # omega_0_block: [B, W_curr, S, S]
            omega_0_block = omega_0[:, w_start:w_end, :, :, 0]
            
            # Y_star_block: [B, S_src, W_curr] -> [B, W_curr, 1, S_src]
            Y_star_block = Y_star[:, :, w_start:w_end].transpose(1, 2).unsqueeze(2)
            
            # Block zero sum: [B, W_curr, S_tgt]
            block_zero_sum = torch.sum(omega_0_block * alpha_0_block * Y_star_block, dim=3)
            spatial_zero_sum[:, :, w_start:w_end] = block_zero_sum.transpose(1, 2)
            
            # Force cleanup of the block's heavy intermediate omega
            del omega_block, alpha_block, Y_shift_block, Y_shift_block_exp, tau_block

        # Multiply by beta_coeff and sum over lags 2L
        term_spatiotemporal = torch.sum(beta_coeff * spatial_lag_sum, dim=-1) # [B, S, W]
        
        # C) Contemporaneous Spatial Interaction (l=0)
        term_zero_spatial = gamma_coeff.squeeze(-1) * spatial_zero_sum # [B, S, W]
        
        # D) Covariates baseline: Sum_p (B_coeff * X_p_norm)
        term_covariates = torch.sum(B_coeff * cov_norm.to(self.device), dim=-1) # [B, S, W]
        
        # Final Output Equation
        Y_hat = term_temporal + term_spatiotemporal + term_zero_spatial + term_covariates
        
        return Y_hat, Y_clean

