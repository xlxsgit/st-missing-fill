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
        
        # Cache station-static geometry terms. They do not depend on batch size,
        # so changing the final mini-batch shape won't trigger expensive rebuilds.
        self._D = None
        self._azimuth = None
        self._Z_abs_diff = None
        self._ASPECT = None
        self._eye = None
        
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
        
        # 1. Lazy initialization of station-static topographic matrices.
        # Topography is static by station; use the first sample/time snapshot.
        if self._D is None or self._D.shape[0] != S:
            topo_ref = topo_features[0, :, 0, :]
            X = topo_ref[:, 0]
            Y = topo_ref[:, 1]
            Z = topo_ref[:, 2]
            aspect = topo_ref[:, 3]  # in degrees

            X_diff = X.unsqueeze(0) - X.unsqueeze(1)  # [S_tgt, S_src]
            Y_diff = Y.unsqueeze(0) - Y.unsqueeze(1)
            self._D = torch.sqrt(X_diff**2 + Y_diff**2 + 1e-8)

            azimuth = 90.0 - (180.0 / math.pi) * torch.atan2(Y_diff, X_diff)
            self._azimuth = torch.fmod(azimuth + 360.0, 360.0)

            Z_tgt = Z.unsqueeze(0).unsqueeze(-1)  # [1, S_tgt, 1]
            Z_src = Z.unsqueeze(0).unsqueeze(-2)  # [1, 1, S_src]
            self._Z_abs_diff = torch.abs(Z_tgt - Z_src).squeeze(0)  # [S, S]

            # Keep target-station aspect repeated across source dimension, same as original layout.
            self._ASPECT = aspect.unsqueeze(1).expand(S, S).contiguous()
            self._eye = torch.eye(S, device=self.device).view(1, 1, S, S)

        D_exp = self._D.view(1, 1, S, S).expand(B, W, S, S)
        azimuth_exp = self._azimuth.view(1, 1, S, S).expand(B, W, S, S)
        Z_abs_diff = self._Z_abs_diff.view(1, 1, S, S).expand(B, W, S, S)
        aspect_tgt = self._ASPECT.view(1, 1, S, S).expand(B, W, S, S)
            
        # Reconstruct Parameter-dependent static penalty
        Z_penalty = torch.exp(-Z_abs_diff / (self.sigma_z + 1e-6))

        # Wind angle differences (delta theta)
        wind_direction_trans = wind_direction.transpose(1, 2) # [B, W, S_src]
        Theta_src = wind_direction_trans.unsqueeze(2).expand(B, W, S, S) # [B, W, S_tgt, S_src]
        
        diff = torch.abs(Theta_src - azimuth_exp)
        delta_theta = torch.minimum(diff, 360.0 - diff)
        
        # Aspect Gain
        aspect_diff = torch.abs(Theta_src - aspect_tgt)
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
        alpha_tilde = alpha_tilde * (1 - self._eye)

        # Travel Time tau calculation
        # Effective wind speed: v_eff = wind_speed * cos(delta_theta)
        wind_speed_trans = wind_speed.transpose(1, 2)
        wind_speed_src = wind_speed_trans.unsqueeze(2).expand(B, W, S, S)
        v_eff = wind_speed_src * torch.cos(delta_theta * math.pi / 180.0)
        v_eff = torch.clamp(v_eff, min=0.1) # Prevent division by zero
        
        # Travel Time tau calculation
        tau = (D_exp / v_eff) / 600.0
        
        return alpha_tilde, tau

    def compute_omega(self, tau, l_values):
        """
        Calculates the Time Alignment Kernel Omega.
        tau: [B, W, S_tgt, S_src]
        l_values: Tensor of shape [num_l] containing the lag steps (e.g., [-3, -2, -1, 1, 2, 3])
        Returns Omega: [B, W, S_tgt, S_src, num_l]
        """
        # omega calculation via native broadcasting instead of massive explicit expansions
        # tau is [B, W, S, S] -> [B, W, S, S, 1]
        tau_usq = tau.unsqueeze(-1)
        
        # l_values is [num_l] -> [1, 1, 1, 1, num_l]
        l_usq = l_values.view(1, 1, 1, 1, -1)
        
        # PyTorch will broadcast the subtraction natively without allocating massive memory in-between
        # Omega = exp( - (l - tau)^2 / sigma_tau )
        omega = torch.exp(-torch.pow(l_usq - tau_usq, 2) / (self.sigma_tau + 1e-6))
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
