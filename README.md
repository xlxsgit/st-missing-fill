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
        matplotlib seaborn \
        scikit-learn statsmodels \
        torch torchvision
```