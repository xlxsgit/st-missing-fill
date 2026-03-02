import nbformat as nbf
import os
notebook_path = 'tests/explore.ipynb'
if os.path.exists(notebook_path):
    nb = nbf.read(notebook_path, as_version=4)
else:
    nb = nbf.v4.new_notebook()

md_content = '''### 🐛 深入修复: 进度条在 Shell 脚本中不渲染的问题
在使用 `./run_experiments.sh` 时，我发现之前利用 `rich` 添加的实时进度条 (Live & Group) 虽然在直接敲击 Python 命令时有效，但在被 Shell 跑批脚本调用时却“隐身”了。

1. **寻根溯源**:
   主要原因是我们在 `runner.py` 里运用了双路日志分发技巧(`Tee`)，把所有的标准输出拦截并写入了 `train.log` 中。而 `rich.Console()` 默认绑定到了当前的 `sys.stdout` (此时已被 `Tee` 劫持)，这就导致 rich 库底层的控制台终端检测机制受挫，直接丢弃或无法输出带有 ANSI 动画转义码的字符串。

2. **解决手段**:
   必须让 `rich` 渲染层“越过”我们的 `Tee` 拦截网。于是我在初始化阶段强制告诉 `rich`：“你要直接跟最底层的真实屏幕流对话！” 
   ```python
   import sys as _sys
   console = Console(file=_sys.__stdout__)
   ```
   **通过直接绑定 `sys.__stdout__`**，我们完美拆分了输出流：
   - 所有的单纯 `print` 或外部模型的普通日志依然会被 `Tee` 捕捉并忠实地记录进 `train.log` 中。
   - 而这根炫酷且实时的 `rich` 进度条及结果表格，则独享一条 V.I.P. 通道直接在终端闪烁刷新，且不会用脏乱的 ANSI 码污染纯文本的日志文件！
'''
nb['cells'].append(nbf.v4.new_markdown_cell(md_content))
nbf.write(nb, notebook_path)
