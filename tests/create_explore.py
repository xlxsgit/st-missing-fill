import nbformat as nbf

nb = nbf.v4.new_notebook()

text = """\
# 需求分析与代码探索

## 1. 额外测试不覆盖原始图且只推理额外时间范围
- **目标文件**: `src/experiments/runner.py`
- **发现**: 目前在 `mode == "test"` 时，会在 378 行的循环中强行分别处理 `train`、`val` 和 `test`（其中 test 会被重命名为 `extra_test`）。这意味着即使用的是仅仅推理旧模型，它仍然也会重新遍历 `train` 和 `val` 切片并保存记录，进而使得它们之前的原图被重新覆盖（或者触发重新画图生成相同图像）。
- **修改方案**: 在 `src/experiments/runner.py` 的 378 行遍历不同 splits 记录到 `rows` 时，如果 `args.mode == "test"`，则通过 `if split_name != "test": continue` 直接跳过 `train` 和 `val` 分支，从而保证 `df_long` 只包含 `extra_test`，避免触发旧图重新绘制。并且这样也节省了对 `train` 和 `val` 再次进行无意义推理的时间。

## 2. 图像样式修改
- **目标文件**: `src/visualization/baseline_compare.py`
- **当前状态**: 
    - 图片名称前缀有 `baseline_grouped_bar_`。
    - 图有总体 `suptitle` 以及子图 `set_title`。
    - 图例标题带有带有括号内容：`title="Models (Sorted by RMSE)"`。
    - 图片具有 `figsize=(14, 16)` 的尺寸，让三大行子图拼起来显得柱状很高。
- **修改方案**:
    - 文件名修改为直接使用 `{split}.png` (例: `extra_test.png`)。
    - 移除 `.suptitle()` 和 `.set_title()` 的调用，从而移除所有标题。
    - 把 `.legend()` 中的 `title` 参数改为单纯的 `"Models"`。
    - 降低图的高度，将 `figsize=(14, 16)` 变更为 `figsize=(14, 8)`。

## 3. 运行配置的修改
- **目标文件**: `run_experiments.sh`
- **修改方案**: 在脚本中将 `RUN_NAME="xlx"` 修改为 `RUN_NAME="tmp"`。将作为独立运行空间避免污染原用户的环境。
"""

cells = [nbf.v4.new_markdown_cell(text)]
nb['cells'] = cells

with open('explore.ipynb', 'w') as f:
    nbf.write(nb, f)
