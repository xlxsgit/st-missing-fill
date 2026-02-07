# st-missing-fill

A Python project for filling missing values in time series data using statistical methods and machine learning techniques.

---

## PROJECT STRUCTURE

```
.
├── README.md
├── data # 数据
│   ├── figs # 图片
│   ├── processed # 处理后的数据
│   └── raw
│       ├── geo_data # 地理数据与变量描述
│       └── ts_data # 原始时序数据
├── docs # 文档
│   ├── old_codes # 旧版代码（sci）
│   ├── papers # 论文
├── logs  # 日志
├── models  # 保存的模型
├── notebooks  # 笔记本
│   └── tmp.ipynb  # 临时笔记本
├── pyproject.toml  # 项目依赖
├── scripts  # 脚本
├── src  # 源代码
│   ├── config  # 配置
│   │   └── config.yaml  # 全局配置文件
│   ├── data  # 数据处理
│   ├── features  # 特征工程
│   ├── models  # 模型
│   └── utils.py  # 工具函数
├── tests  # 测试
└── uv.lock  # 依赖锁文件
```


## 1. 🗃️ DATA SOURCE

The dataset is downloaded from Hugging Face: [MeteoSwiss/PeakWeather](https://huggingface.co/datasets/MeteoSwiss/PeakWeather/tree/main)

---

## 2. ⚙️ CONFIGURATION

- **OS**: macOS Tahoe 26.2  
- **Hardware**: Apple Silicon M4, CPU 10 cores, GPU 10 cores, 24GB RAM  
- **Python version**: 3.12  
- **Virtual environment tool**: `uv`  
- **Dependency files**: [`pyproject.toml`](pyproject.toml), [`uv.lock`](uv.lock)

```bash
uv add \
        tqdm pyyaml \
        jupyter notebook ipykernel \
        openpyxl xlrd pyarrow fastparquet \
        numpy pandas scipy \
        matplotlib seaborn folium\
        scikit-learn statsmodels pypots\
        torch torchvision
```

---

## 3. 🚀 WORKFLOW

### Step 1: Configure Processing Parameters
Edit `src/config/config.yaml` to set the data range you want to process:

```yaml
processing:
  start_year: 2023
  end_year: 2024
```

### Step 2: Run Data Processing
Execute the processing script to merge data and generate a single consolidated dataset in `data/processed/all_data.parquet`.

```bash
uv run scripts/run_processing.py
```

### Step 3: Analyze Data (EDA)
Open the notebooks to explore data distribution and quality.

```bash
uv run jupyter notebook notebooks/tmp.ipynb
```
