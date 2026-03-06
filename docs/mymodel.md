# STAR-STAT Model Architecture (MyModel)

The **Spatio-Temporal Aligned Transformer (STAT)** is a novel architecture designed to handle severe contiguous missing data scenarios (such as Sequential Block Missing and Spatially Correlated Missing patterns) in meteorological and environmental multivariate time series.

## 1. Optimal Structural Configuration

Based on intensive Hyperparameter Optimization (HPO) utilizing Optuna across diverse missingness ratios, the optimal architectural blueprint for the hyper-network is determined to be:

- **Embed Dimension (`d_model`)**: `128` (Provides sufficient expressivity to encode multivariate temporal paths alongside topographic constraints).
- **Attention Heads (`nhead`)**: `4` (Balances representation subspaces without overfitting small local topologies).
- **Transformer Encoder Layers (`num_layers`)**: `1` (A shallow block proved highly resistant to overfitting, particularly on sparse short-range validation slices).
- **Spatio-Temporal Lags (`l_lags`)**: `2` (A window lag size of 2 perfectly aligns with physical time-delay properties over local station distances).

## 2. Factorized Spatio-Temporal Attention Mechanisms

To counter `SEQ` (extended time block missing) and `SCM` (spatial cluster missing) degradation, the system deploys a decoupled attention pipeline rather than a single flattened mechanism:

### 2.1. Temporal Pass (Intra-Station Auto-regressive Scanning)
The input matrix is reshaped to `[Batch * Stations, sequence_len, d_model]`. Temporal Self-Attention allows each individual spatial node to build its own localized autoregressive pattern independently, analyzing its own historical trends.

### 2.2. Spatial Pass (Inter-Station Cross-Communication)
Following the temporal encoding, the state tensor is dynamically transposed to `[Batch * sequence_len, Stations, d_model]`. Spatial Self-Attention steps in to propagate weather states across different Topographical locations at the exact same time step. This is mathematically critical for `SCM` scenarios where a sensor node is physically blinded for the duration. It learns to 'borrow' structural wind covariates from its geographical neighbors.

## 3. Physical Priors & Instance Normalization

The STAT model is unique in that the Hyper-Network does not directly output the final physical metrics. Instead, the Transformer layers deduce high-dimensional spatial-temporal coefficients: $A(t)$, $\beta(t)$, $\gamma(t)$, and $B(t)$.

* **Zero-Mean Unit-Variance Scaling**: Raw data inputs mapping strictly into the hyper-network are Z-score normalized point-in-time to eliminate matrix explosion (gradient instabilities).
* **Deterministic Priors**: LV95 Topographical distances are explicitly used to calculate frictional delays and physical influence factors, strictly limiting the scope of what the Transformer must learn. The network simply maps raw residuals to these explicit physical laws.
