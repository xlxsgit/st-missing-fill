# st-missing-fill

## 项目结构

```angular2html
.
├── README.md
├── data
│   ├── meteo-swiss-data
│   │   ├── 2020.parquet
│   │   ├── 2021.parquet
│   │   ├── 2022.parquet
│   │   ├── 2023.parquet
│   │   ├── 2024.parquet
│   │   └── 2025.parquet
│   └── meteo-swiss-description
│       ├── disclaimer.txt
│       ├── installation.parquet
│       ├── parameters.parquet
│       └── stations.parquet
├── main.py
├── model
├── pyproject.toml
├── src
│   ├── core
│   └── process
├── tests
│   ├── test.ipynb
│   └── test.py
└── uv.lock
```

## 数据来源

使用的数据集下载自 Hugging Face：
[MeteoSwiss/PeakWeather](https://huggingface.co/datasets/MeteoSwiss/PeakWeather/tree/main)

```angular2html
Disclaimer:

The dataset is published under the Creative Commons Licence CC BY 4.0. Reproduction and redistribution of the data is only permitted with proper attribution (source: MeteoSwiss).
```

## 配置信息
- 使用设备：MacBook Air M4, 24GB RAM
- 虚拟环境管理工具：uv
- Python 版本：3.12
- 依赖文件：[pyproject.toml](pyproject.toml), [uv.lock](uv.lock)

```bash
uv add \
        tqdm pyyaml\
        jupyter notebook ipykernel \
        openpyxl xlrd pyarrow fastparquet \
        numpy pandas scipy \
        matplotlib seaborn \
        scikit-learn statsmodels \
        torch torchvision
```

