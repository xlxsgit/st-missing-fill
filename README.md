# st-missing-fill

## 项目结构

```angular2html
.
├── README.md
├── data
├── main.py
├── model
├── pyproject.toml
├── src
│   └── core
├── tests
└── uv.lock
```

## 配置信息
- 使用设备：MacBook Air M4, 24GB RAM
- 虚拟环境管理工具：uv
- Python 版本：3.12
- 依赖文件：[pyproject.toml](pyproject.toml), [uv.lock](uv.lock)

```bash
uv add requests pandas matplotlib seaborn numpy scipy tqdm \
jupyter notebook ipykernel scikit-learn statsmodels openpyxl xlrd torch torchvision \
pyarrow fastparquet

```
