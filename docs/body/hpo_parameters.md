# Hyperparameter Optimization (HPO) Configuration

本研究采用 Optuna 框架对各算法的结构性超参数进行了自动化寻优，旨在确保各模型在特定缺失模式与缺失比例下的公平对比。对于深度学习模型（如 SAITS、iTransformer 等），寻优重点在于 Transformer 层数及隐藏层维度；而针对本研究提出的 STAT 模型，进一步涵盖了时间滞后项 `l_lags` 与空间注意力头的搜索。经典的基准模型（如 LOCF、MICE、VCAAN）则采用了固定的建议配置，以模拟实际业务中常见的基准表现。具体参数搜索空间详见下表及附件中的 [hpo_parameters.xlsx](file:///Users/lxx/Documents/codes/st-missing-fill/docs/body/hpo_parameters.xlsx)。

## HPO Sparse Matrix

| Algorithm | d_model | nhead | num_layers | l_lags | n_layers | n_heads | d_ffn | rnn_hidden_size | knn_neighbors | max_iter | tol |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **STAT (Proposed)** | [64, 128] | [2, 4, 8] | {1, 2, 3} | {1, 2, 3, 4} | | | | | | | |
| **SAITS** | [32, 64] | | | | [2, 4] | [2, 4] | [64, 128] | | | | |
| **iTransformer** | [32, 64] | | | | [2, 4] | [2, 4] | [64, 128] | | | | |
| **GRUD** | | | | | | | | [32, 64] | | | |
| **USGAN** | | | | | | | | [32, 64] | | | |
| **KNN** | | | | | | | | | {3, 5, 7} | | |
| **MICE** | | | | | | | | | | 10 | 0.001 |
| **VCAAN** | | | | | | | | | | 8 | 1e-4 |
| **LOCF** | | | | | | | | | | | |

> [!TIP]
> This sparse format allows for easy comparison of structural parameters across different model families.
