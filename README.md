# st-missing-fill

A Python project for filling missing values in time series data using statistical methods and machine learning techniques.

---

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
