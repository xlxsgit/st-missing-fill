# st-missing-fill

时序缺失构造与插补实验项目。  
本项目的主旨是：建立一个高内聚、易扩展的模型测试实验台，**业务逻辑层层切割**。

## 核心架构与功能流转图

```text
[用户层] 
   ├─ run_experiments.sh       <-- 快捷启动脚本（暴露模型列表、时间切分、HPO配置等）
   └─ main.py                  <-- 命令行解析入口（封装 argparse）
            │
            ▼
[调度层: src/experiments/runner.py]
   │  (它是整个实验的大群控，负责循环组合 [模型 x 缺失模式 x 缺失率])
   ├─ 1. 加载源数据 ──> src/data/load.py
   ├─ 2. 时间域切分 ──> src/data/splitter.py (按指令切成 train/val/test 及窗函数划窗)
   ├─ 3. 施加随机破坏 ──> src/data/misser.py (支持 mcar 完全随机 / seq 连续缺失 / scm 空间相关缺失)
   │
   └─ 4. 派发模型集 ──> [分发器: src/models/dispatcher.py]
            │          (包含 Optuna HPO 和最终推理)
            │
            ├─ 深度模型 ──> src/models/pypots_baselines.py (saits, grud, itransformer 等)
            ├─ 传统基线 ──> src/models/sklearn_baselines.py (knn, mice, 内部包含了分段推理防 OOM)
            ├─ 快速填补 ──> src/models/statistical_baselines.py (locf)
            └─ 最新实验 ──> src/models/vcaan.py (基于关联的迭代基线)
            │
            ▼
[持久层: src/experiments/results.py]
   ├─ metrics.json         <-- 指标摘要
   ├─ results_long.csv     <-- 展平的长数据表
   ├─ summary_all_parts.csv<-- 全局累加总表（汇集每次实验最佳成绩）
   └─ logs/latest/         <-- 热链，直达上一次运行产物
```

## 1. 项目组织结构

```text
.
├── main.py                            # 统一入口
├── run_experiments.sh                 # 快捷启动跑批脚本（支持快速改配、改时间、挂载 HPO）
├── pyproject.toml                     # 项目依赖清单
├── src
│   ├── data
│   │   ├── processing.py              # 数据清洗
│   │   ├── load.py                    # 数据加载
│   │   ├── misser.py                  # 缺失构造
│   │   └── splitter.py                # 动态时间集切分
│   ├── models
│   │   ├── dispatcher.py              # 基线与优化统一派发器
│   │   ├── search_space.py            # Optuna HPO 超参搜寻空间
│   │   ├── pypots_baselines.py        # 深度学习基线
│   │   ├── sklearn_baselines.py       # 机器学习基线 
│   │   ├── statistical_baselines.py   # LOCF基线
│   │   └── vcaan.py                   # VCAAN基线
│   ├── experiments
│   │   ├── runner.py                  # 实验大盘编排
│   │   └── results.py                 # 分离的报表生成逻辑
│   └── evaluate.py                    # 纯原位 RMSE 评估
├── data
│   └── raw/processed                  # 隔离的数据资源
└── logs                               # 最新 run 的汇总目录 / 增量报表
```

## 2. 深入理解代码运作机制

整个实验流水线设计为高度解耦，各司其职：

### 2.1 实验大盘编排器 (src/experiments/runner.py)
最核心的文件，它就像一个总指挥，拿到外部脚本（`run_experiments.sh`）赋予的诸如“训练几月份到几月份”、“要哪些模型比对”的指令后，启动一个嵌套循环。
在每次抵达一个诸如 `(model=saits, pattern=mcar, pi=0.1)` 的确定组合时，它会将源数据丢给底层的 `misser.py` 挖洞，然后唤起 `dispatcher.py` 求解。

### 2.2 统一派发器与 HPO 超参寻优框架 (src/models/dispatcher.py)
这是重构后新引入的关键模块。它的职能有两个：
- **挂载 Optuna 框架进行自动调参 (HPO)**：如果外界指令 `--hpo-trials > 0`，派发器会去 `search_space.py` (内置了各个核心模型的最佳预留设定) 里抓取搜索边界。它会在 **验证集 (Val)** 上对该组合反复试错，直到找出最佳参数。
- **派发推理**：若配置不包含 HPO 亦或 HPO 执行完毕，它会将带有最佳超参的任务抛掷给各自的真实算法基线进行落地的网络构建和拟合。

### 2.3 基线的隔离与防护机制
为了保证不让庞杂的模型污染流程，当前算法分别被隔离装在了 `src/models/` 目录下：
- **`pypots_baselines.py`**：集成了以 PyPOTS 为基础的所有深度学习模型。值得一提的是内含了专门的分块机制（chunking）和显存强制释放 `torch.mps.empty_cache()`，用于确保在强算压力下 MPS Mac 设备不会 Out-Of-Memory。
- **`sklearn_baselines.py`**：应对传统机器学习插补方法 `knn` 以及 `mice`。特别是 `mice` 多重弥补，由于底层实现占用了巨大的内存运算通道，该文件内定制了特有的 `mice_chunk_steps` 滑动小窗机制强行规避资源死锁。

## 3. 当前支持模型与缺失模式

### 3.1 Baselines (模型列阵)
- **深度模型（基于 PyPOTS）**：`saits`, `grud`, `usgan`, `itransformer`
- **传统插值模型**：`locf`, `knn`, `mice`
- **自主构造基准**：`vcaan` (一种基于特征工程相关性辅助并结合迭代调优补全的模型)

### 3.2 缺失模式验证
项目支持对源数据进行三种工业模拟级别的破坏：
- `mcar` (完全独立随机缺失)
- `seq` (连续时间切片状黑障缺失)
- `scm` (引入基于站点强空间相关性的区块崩解)

## 4. 常用运行方式

我们已废弃了早期繁琐的组合命令或者手写长参数。当前全部参数均在入口脚本顶层暴露！

**最推荐的运行方案：**
```bash
./run_experiments.sh
```
你可以随意打该文件来修改执行配比。包括：支持运行多大范围的数据集（1月到3月等）、需要启动的基线（`locf`等）、以及最重要的：**是否开启带有超参数搜索优化的机制（`--hpo-trials`）**。

## 5. 结果输出文件说明

当启动任意测试后，除了将在 `logs/xxxxx_name/` 下属生成单独文件夹外，系统会：
- 将全局的每次试验都递增写入统括表：`logs/summary_all_parts.csv`
- 创建一个随跑随更新的快捷目录软交点：`logs/latest/`
- `results_pivot.csv`：透视表（train/val/test）
- `config.json`：配置快照
- `metrics.json`：指标摘要

全局汇总：
- `logs/summary.csv`：最新 run 的汇总（浮点保留 4 位）

## 6. 注意事项

- `main.py` 是唯一实验入口，不再依赖 `tests/tmp.py`。
- `KNN/MICE` 已改为分块插补（避免大矩阵一次性插补过慢）。
- `VCAAN` 使用 LOCF 作为预插补，再做迭代优化。
- `mice` 常见 `ConvergenceWarning`，不影响流程执行；若需更稳定可增加迭代次数或调小数据规模。
- `vcaan` 中相关系数计算可能出现 `RuntimeWarning`（常见于低方差列），已做数值兜底处理，流程可继续。

## 7. 环境与依赖

- Python `3.12`
- 包管理：`uv`
- 推荐安装：

```bash
uv sync
```

如需手动安装，保留原依赖清单：

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
