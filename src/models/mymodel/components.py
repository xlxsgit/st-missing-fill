import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PhysicalAlignmentPriors(nn.Module):
    """
    Computes the Physics-informed Priors for wind Spatio-Temporal Auto-Regression.
    This includes:
      1. Precise distance matrix using LV95 coordinates.
      2. 3D Topographic Friction alpha_tilde.
      3. Expected physical travel time tau (based on distance and effective wind speed).
      4. Spatio-Temporal Alignment Kernel Omega.
    """
    def __init__(self, num_stations, device, sigma_d=10000.0, sigma_z=500.0, lambda_topo=0.5, sigma_tau=1.0):
        super().__init__()
        self.num_stations = num_stations
        self.device = device
        
        # Learnable parameters for physics priors
        self.sigma_d = nn.Parameter(torch.tensor(sigma_d, dtype=torch.float32))
        self.sigma_z = nn.Parameter(torch.tensor(sigma_z, dtype=torch.float32))
        self.lambda_topo = nn.Parameter(torch.tensor(lambda_topo, dtype=torch.float32))
        self.sigma_tau = nn.Parameter(torch.tensor(sigma_tau, dtype=torch.float32))
        
    def forward(self, topo_features, wind_speed, wind_direction):
        """
        topo_features: [batch_size, num_stations, window_size, topo_dim]
                       Assuming topo_dim includes at least X, Y, Z, ASPECT
        wind_speed:    [batch_size, num_stations, window_size]
        wind_direction:[batch_size, num_stations, window_size]
        
        Returns:
            alpha_tilde: [batch_size, window_size, num_stations, num_stations] (target, source)
            tau:         [batch_size, window_size, num_stations, num_stations]
        """
        B, S, W = wind_speed.shape
        
        # We assume topo_features are static over time, so we just take the first window step
        # Let's say topo_features order: [X, Y, Z, ASPECT, SLOPE, TPI, ...]
        # Here we extract X, Y, Z, ASPECT
        # Shape: [B, S, topo_dim]
        X = topo_features[:, :, 0, 0]
        Y = topo_features[:, :, 0, 1]
        Z = topo_features[:, :, 0, 2]
        ASPECT = topo_features[:, :, 0, 3] # in degrees
        
        # Calculate Distance Matrix D: [B, S, S]
        # D_s_s' means from source s' to target s
        X_diff = X.unsqueeze(1) - X.unsqueeze(2) # [B, S (target), S (source)]
        Y_diff = Y.unsqueeze(1) - Y.unsqueeze(2)
        D = torch.sqrt(X_diff**2 + Y_diff**2 + 1e-8) # [B, S, S]
        
        # Calculate Azimuth Phi: from source s' to target s
        # atan2(y, x) -> standard math coords. LV95 usually Y is North, X is East.
        # Wind Dir: 0=North, 90=East. Map atan2 to Wind Dir coords:
        # Phi = 90 - (180/pi) * atan2(Y_diff, X_diff)
        azimuth = 90.0 - (180.0 / math.pi) * torch.atan2(Y_diff, X_diff)
        azimuth = torch.fmod(azimuth + 360.0, 360.0) # [B, S, S]
        
        # Expand over window size
        D_exp = D.unsqueeze(1).expand(B, W, S, S)             # [B, W, S_tgt, S_src]
        azimuth_exp = azimuth.unsqueeze(1).expand(B, W, S, S)
        Z_tgt = Z.unsqueeze(1).unsqueeze(3).expand(B, W, S, S) # [B, W, S_tgt, S_src]
        Z_src = Z.unsqueeze(1).unsqueeze(2).expand(B, W, S, S)
        ASPECT_tgt = ASPECT.unsqueeze(1).unsqueeze(3).expand(B, W, S, S)

        # Wind angle differences (delta theta)
        # Theta is source wind direction at time t
        wind_direction_trans = wind_direction.transpose(1, 2) # [B, W, S_src]
        Theta_src = wind_direction_trans.unsqueeze(2).expand(B, W, S, S) # [B, W, S_tgt, S_src]
        
        diff = torch.abs(Theta_src - azimuth_exp)
        delta_theta = torch.minimum(diff, 360.0 - diff)
        
        # Elevation Penalty
        Z_penalty = torch.exp(-torch.abs(Z_tgt - Z_src) / (self.sigma_z + 1e-6))
        
        # Aspect Gain
        # Difference between source wind direction and target aspect
        aspect_diff = torch.abs(Theta_src - ASPECT_tgt)
        mu = F.relu(torch.cos(aspect_diff * math.pi / 180.0))
        
        # Topographic friction alpha_tilde
        alpha_tilde = torch.cos(delta_theta * math.pi / 180.0) * \
                      torch.exp(-D_exp / (self.sigma_d + 1e-6)) * \
                      Z_penalty * \
                      (1.0 + self.lambda_topo * mu)
                      
        # Cutoff wind that is blowing > 90 degrees away
        mask_wind = (delta_theta <= 90.0).float()
        alpha_tilde = alpha_tilde * mask_wind
        
        # Prevent self-loop in spatial propagation (diagonal = 0)
        eye = torch.eye(S, device=self.device).view(1, 1, S, S)
        alpha_tilde = alpha_tilde * (1 - eye)

        # Travel Time tau calculation
        # Effective wind speed: v_eff = wind_speed * cos(delta_theta)
        wind_speed_trans = wind_speed.transpose(1, 2)
        wind_speed_src = wind_speed_trans.unsqueeze(2).expand(B, W, S, S)
        v_eff = wind_speed_src * torch.cos(delta_theta * math.pi / 180.0)
        v_eff = torch.clamp(v_eff, min=0.1) # Prevent division by zero
        
        # D_exp is in meters, v_eff is in m/s. 
        # t_seconds = D / v_eff. 
        # Model steps are 10 minutes = 600 seconds.
        tau = (D_exp / v_eff) / 600.0
        
        return alpha_tilde, tau

    def compute_omega(self, tau, l_values):
        """
        Calculates the Time Alignment Kernel Omega.
        tau: [B, W, S_tgt, S_src]
        l_values: Tensor of shape [num_l] containing the lag steps (e.g., [-3, -2, -1, 1, 2, 3])
        Returns Omega: [B, W, S_tgt, S_src, num_l]
        """
        B, W, S, _ = tau.shape
        num_l = len(l_values)
        tau_exp = tau.unsqueeze(-1).expand(B, W, S, S, num_l)
        l_exp = l_values.view(1, 1, 1, 1, num_l).expand(B, W, S, S, num_l).to(self.device)
        
        # Omega = exp( - (l - tau)^2 / sigma_tau )
        omega = torch.exp(-torch.pow(l_exp - tau_exp, 2) / (self.sigma_tau + 1e-6))
        return omega


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, length):
        return self.pe[:length, :]


class STAT_HyperNetwork(nn.Module):
    """
    Spatio-Temporal Transformer Encoder generating dynamic AR coefficients and clean representations.
    """
    def __init__(self, num_stations, in_features, d_model=128, nhead=8, num_layers=4, 
                 l_lags=3, num_covariates=10, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.l_lags = l_lags
        
        # Initial projection from raw high-dimensional features
        self.input_proj = nn.Linear(in_features, d_model)
        
        # Temporal Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Spatial Station Embedding
        self.station_embed = nn.Embedding(num_stations, d_model)
        
        # 1. Temporal Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, 
                                                    dropout=dropout, batch_first=True)
        self.temporal_transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 2. Spatial Transformer Encoder
        spatial_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, 
                                                    dropout=dropout, batch_first=True)
        self.spatial_transformer = nn.TransformerEncoder(spatial_layers, num_layers)
        
        # Decoupled Projection Heads
        # 1. Clean Filler Head (Outputs cleaned Y_clean)
        self.head_filler = nn.Linear(d_model, 1)
        
        # 2. Temporal AR Head (Outputs A(t +/- L)) - excluded l=0 implies 2L coefficients
        self.head_ar = nn.Linear(d_model, 2 * l_lags)
        
        # 3. Spatial & Covariates Head
        # Outputs: beta(+/- L) -> 2L coefficients
        # Outputs: B_p -> P coefficients for local covariates
        # Outputs: gamma -> 1 coefficient for instant l=0 spatial alignment
        self.head_spatial_cov = nn.Linear(d_model, 2 * l_lags + num_covariates + 1)
        
    def forward(self, x, station_ids):
        """
        x: [B, S, W, in_features]
        station_ids: [S] long tensor representing station indices
        
        Returns:
            Y_clean: [B, S, W]
            A:       [B, S, W, 2L]
            beta:    [B, S, W, 2L]
            B_cov:   [B, S, W, P]
            gamma:   [B, S, W, 1]
        """
        B, S, W, _ = x.shape
        
        # Treat [B*S] as the batch dimension, and [W] as the sequence length.
        # This performs independent Temporal Self-Attention per station, 
        # reducing memory from O((SW)^2) to O(S*W^2).
        # x_time: [B*S, W, in_features]
        x_time = x.reshape(B*S, W, -1)
        
        # Project
        h = self.input_proj(x_time) # [B*S, W, d_model]
        
        # Add Time Positional Encoding
        # pe shape: [W, d_model] -> expand to [B*S, W, d_model]
        pe = self.pos_encoder(W).unsqueeze(0).expand(B*S, -1, -1)
        
        # Add Station Embedding
        # st_embed: [S, d_model]
        st_embed = self.station_embed(station_ids)
        st_embed_expanded = st_embed.unsqueeze(0).unsqueeze(2).expand(B, -1, W, -1).reshape(B*S, W, -1)
        
        # Final input to Transformer
        h = h + pe + st_embed_expanded
        
        # --- 1. Temporal Attention Pass ---
        # out_time: [B*S, W, d_model]
        out_time = self.temporal_transformer(h) 
        
        # --- 2. Spatial Attention Pass ---
        # We need to reshape the output to perform attention across the S dimension.
        # out_time is currently [B*S, W, d_model]. 
        # Unpack B*S: [B, S, W, d_model]
        out_unpacked = out_time.view(B, S, W, self.d_model)
        
        # Transpose to group Stations together for each Time step: [B, W, S, d_model]
        # Then flatten to [B*W, S, d_model] to treat B*W as the independent batch dimension
        h_spatial = out_unpacked.transpose(1, 2).reshape(B*W, S, self.d_model)
        
        out_spatial = self.spatial_transformer(h_spatial) # [B*W, S, d_model]
        
        # --- 3. Final Reshape to Output ---
        # Reshape back to [B, W, S, d_model] -> [B, S, W, d_model]
        out_reshaped = out_spatial.view(B, W, S, self.d_model).transpose(1, 2)
        
        # Decoupled projection heads
        Y_clean = self.head_filler(out_reshaped).squeeze(-1) # [B, S, W]
        
        # Apply Tanh to bound the coefficients dynamically so they don't blow up raw data
        A = torch.tanh(self.head_ar(out_reshaped))           # [B, S, W, 2L]
        
        cov_out = torch.tanh(self.head_spatial_cov(out_reshaped)) # [B, S, W, 2L + P + 1]
        
        beta = cov_out[..., :2 * self.l_lags]
        # Remaining outputs: first P are B_cov, last 1 is gamma
        B_cov = cov_out[..., 2 * self.l_lags:-1]
        gamma = cov_out[..., -1:]
        
        return Y_clean, A, beta, B_cov, gamma
