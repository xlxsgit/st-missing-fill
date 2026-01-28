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
        scikit-learn statsmodels \
        torch torchvision
```

---

## 3. 🚀 WORKFLOW

### Step 1: Configure Processing Parameters
Edit `config.yaml` to set the data range you want to process:

```yaml
processing:
  start_year: 2023
  end_year: 2024
```

### Step 2: Run Data Processing
Execute the processing script to merge data and generate variable-specific datasets in `data/processed-data/`.

```bash
uv run scripts/run_processing.py
```

### Step 3: Analyze Data (EDA)
Open the EDA notebook to explore data distribution and quality.

```bash
uv run jupyter notebook tests/eda.ipynb
```

### Step 4: Validate Data Loading (Optional)
Run the test notebook to verify data paths and station visualization.

```bash
uv run jupyter notebook tests/test.ipynb
```

---

## 4. 🎯 MISSING PATTERN GENERATION

This project includes functionality to generate three types of missing data patterns for time series:

### Supported Missing Patterns

1. **MCAR (Missing Completely At Random)**: Values are missing completely at random across all time points and stations
2. **SEQ (Sequential Missing)**: Continuous sequences of missing values at individual stations with random lengths
3. **SPATIAL (Spatially Correlated Sequential Missing)**: Missing sequences occur at spatially neighboring stations simultaneously

### Quick Start

```python
from src.missing_generator import MissingDataGenerator
import pandas as pd

# Load your data
data = pd.read_parquet('data/processed-data/temperature.parquet')

# Create generator
generator = MissingDataGenerator(data, random_state=42)

# Apply MCAR pattern with 20% missing rate
data_missing = generator.apply_mcar(missing_rate=0.2)

# Or use the generic interface
data_missing = generator.apply_pattern(
    pattern_type='seq',  # 'mcar', 'seq', or 'spatial'
    missing_rate=0.3,
    max_length=100
)
```

### Run Example Script

```bash
uv run python scripts/example_missing_patterns.py
```

### Documentation

- [Detailed Usage Guide](docs/MISSING_PATTERN_USAGE.md)
- [API Documentation](src/missing_patterns.py)
- [Test Examples](tests/test_missing_patterns.py)

### EDA Notebook

The `tests/eda.ipynb` notebook now includes:
- Analysis of existing missing patterns
- Station distance and spatial relationship analysis
- Determination of spatial neighborhood threshold
- Visualization of all three missing patterns
- Comparison of different patterns

---

## 5. 🧪 TESTING

Run unit tests to validate the missing pattern generation:

```bash
uv run pytest tests/test_missing_patterns.py -v
```

All tests cover:
- Pattern shape and type validation
- Missing rate accuracy
- Sequential continuity for SEQ pattern
- Spatial neighborhood constraints
- Reproducibility with random seeds