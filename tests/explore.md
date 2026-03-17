# Transformer Architecture Exploration

## Objective
Analyze the input, layers, output, and dimensions of the Transformer model in `mymodel`.

## Findings

### 1. Transformer Input Preparation
The input to the `STAT_HyperNetwork` is constructed by concatenating several features:
- `Y_safe`: `[B, S, W, 1]` (Raw wind speed with NaNs replaced by 0)
- `mask`: `[B, S, W, 1]` (Observation mask, 1 for observed, 0 for missing)
- `wind_dir`: `[B, S, W, 1]` (Wind direction)
- `covariates (norm)`: `[B, S, W, P]` (Normalized meteorological features like Temperature, Pressure, etc.)
- `topo_features (norm)`: `[B, S, W, topo_dim]` (Normalized topographic features like X, Y, Z, Aspect)

**Total Input Dimension (`in_features`)**: $1 + 1 + 1 + P + \text{topo\_dim}$
**Input Shape**: `[B, S, W, in_features]`

### 2. Hyper-Network Internal Layers
The `STAT_HyperNetwork` processes the input through three main stages:

#### A. Embedding and Projection
- **Input Projection**: A linear layer projects `in_features` to `d_model`.
  - Shape: `[B*S, W, in_features] -> [B*S, W, d_model]`
- **Temporal Positional Encoding**: Added to the sequence dimension `W`.
  - Shape: `[W, d_model]`
- **Spatial Station Embedding**: A learnable embedding for each station is added.
  - Shape: `[S, d_model]` expanded to `[B*S, W, d_model]`

#### B. Duel Transformer Encoders (Spatio-Temporal Attention)
1. **Temporal Transformer**:
   - `num_layers` of Transformer blocks.
   - Operates on the time dimension `W` for each station independently.
   - Input: `[B*S, W, d_model]`
   - Output: `[B*S, W, d_model]`
2. **Spatial Transformer**:
   - First, reshapes and transposes to group stations for each time step.
   - Shape: `[B*S, W, d_model] -> [B, S, W, d_model] -> [B, W, S, d_model] -> [B*W, S, d_model]`
   - `num_layers` of Transformer blocks.
   - Operates on the station dimension `S` for each time step independently.
   - Input: `[B*W, S, d_model]`
   - Output: `[B*W, S, d_model]`

#### C. Output Projection Heads
Reshapes back to `[B, S, W, d_model]` and splits into several linear heads:
1. **Filler Head**: `Linear -> Y_clean`
   - Shape: `[B, S, W, d_model] -> [B, S, W]`
2. **Temporal AR Head**: `Linear + Tanh -> A_coeff`
   - Shape: `[B, S, W, d_model] -> [B, S, W, 2L]` (where $2L$ is the number of lags, e.g., $2 \times 3 = 6$)
3. **Spatial & Covariate Head**: `Linear + Tanh -> concat(beta, B_cov, gamma)`
   - Shape: `[B, S, W, d_model] -> [B, S, W, 2L + P + 1]`
   - **beta**: `[B, S, W, 2L]`
   - **B_cov**: `[B, S, W, P]`
   - **gamma**: `[B, S, W, 1]`

### 3. Summary of Dimensions
- `B`: Batch size
- `S`: Number of stations
- `W`: Sequence length (Window)
- `P`: Number of covariates
- `topo_dim`: Topographic feature dimension
- `d_model`: Transformer hidden dimension
- `2L`: Double the number of temporal lags
- `in_features`: $3 + P + \text{topo\_dim}$

### 4. Network Structure Diagram (ASCII)

```text
    [Input Features]
    (Y_raw, Mask, Wind_dir, Covariates, Topo)
    Shape: [B, S, W, in_features]
           |
           v
    [Linear Projection] -> [B, S, W, d_model]
           |
           v
    [Add Positional Encoding (Time)] 
    [Add Station Embedding (Space)]
           |
           v
+----------+----------+
|  Temporal Encoder   |  <-- Reshape to [B*S, W, d_model]
| (Self-Attention @W) |      (Processes each station's timeline)
+----------+----------+
           |
           v
+----------+----------+
|   Spatial Encoder   |  <-- Reshape to [B*W, S, d_model]
| (Self-Attention @S) |      (Processes each time step's space)
+----------+----------+
           |
           v
    [Final Representation]
    Shape: [B, S, W, d_model]
           |
           +-----------------------+-----------------------+
           |                       |                       |
    [Filler Head]           [AR Head]            [Spatial/Cov Head]
    (Linear)                (Linear + Tanh)      (Linear + Tanh)
           |                       |                       |
    [Y_clean]               [A_coeff]            [beta, B_cov, gamma]
    [B, S, W]               [B, S, W, 2L]        [B, S, W, 2L+P+1]
```

## 3. 最终命名方案 (中英双语版)

为了确保学术严谨性，避开歧义，并突出“自回归”这一统计学核心，最终建议如下：

### 3.1 总体架构命名

**主模型名称**: **Spatio-Temporal Attention-based Autoregressive Network**
**中文名称**: **基于时空注意力的自回归插补网络**
**核心缩写**: **STA-AR**

---

### 3.2 章节详细命名

| 章节 | 英文全称 (Full English Name) | 中文全称 (Full Chinese Name) | 缩写 (Abbr.) |
| :--- | :--- | :--- | :--- |
| **主标题** | **Spatio-Temporal Attention-based Autoregressive Network** | **基于时空注意力的自回归插补网络** | **STA-AR** |
| **4.3 节** | **Spatio-Temporal Aligned Autoregression** | **时空对齐自回归** | **STAR** |
| **4.4 节** | **Spatio-Temporal Attention HyperNetwork** | **时空注意力超网络** | **ST-AHN** |

---

### 3.3 命名动机 (Rationale)

1.  **自回归为本 (AR-Centric)**: 将 **Autoregressive** 置于大标题的中心位置，符合统计学硕士论文对模型本质的界定。
2.  **避开歧义 (No Ambiguity)**: 弃用了 STAT 等带有双关可能的词汇，改用 **STA** (Spatio-Temporal Attention)，更加中立且准确地描述了技术途径。
3.  **对齐与注意力 (Alignment & Attention)**: 
    - **4.3 节的 STAR**: 强调了物理上的“对齐（Aligned）”，这是该章基于风轴物理推导的核心。
    - **4.4 节的 ST-AHN**: 强调了神经网络部分的“注意力（Attention）”和“超网络（HyperNetwork）”属性，避开了对 Transformer 完整性的争议。
4.  **学术协调**: 整体叫 **STA-AR**，内部逻辑模块叫 **STAR** 和 **ST-AHN**，在缩写上具有逻辑相关性，便于读者记忆。
