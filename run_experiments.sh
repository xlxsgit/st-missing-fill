#!/bin/bash

# ==========================================
# 🚀 快速跑批实验启动脚本 (Fast Experiment Runner)
# ==========================================
# 你只需要在这里修改参数，然后在这个项目根目录下运行: 
# ./run_experiments.sh
# ==========================================

# 1. 想要运行的模型 (用逗号分隔，不要有空格)
# 可选项: locf,saits,grud,usgan,itransformer,knn,mice,vcaan
MODELS="locf,saits,grud,itransformer,knn,mice,vcaan"

# 2. 缺失模式 (用逗号分隔，不要有空格)
# 可选项: mcar(完全随机), seq(连续块缺失), scm(空间相关缺失)
PATTERNS="mcar,seq,scm"

# 3. 缺失率 (用逗号分隔，不要有空格，结尾不要留逗号)
# 例如 0.1,0.2 代表 10% 和 20%
PIS="0.1,0.3,0.5"

# 4. 数据切分时间范围 (数据量越小跑的越快，但容易让模型欠拟合)
# Train 范围
TRAIN_START="2023-01-01"
TRAIN_END="2023-01-31"
# Val 范围
VAL_START="2023-02-01"
VAL_END="2023-02-28"
# Test 范围
TEST_START="2023-03-01"
TEST_END="2023-03-31"

# 5. 深度学习相关的训练大周期数 (Epochs)
# 快速测试写 1，要出成果写 10~30
EPOCHS=1

# 6. 超参数搜索次数 (HPO Trials)
# 为 0 则关闭模型结构参数搜索 (极速)，如果你想追求更好性能，可以设为 2, 5 甚至 10
HPO_TRIALS=0

# 7. 本次实验文件夹后缀名字
RUN_NAME="fast_script_test"


# ==========================================
# 👇 下面的核心命令不要修改 👇 
# ==========================================

echo "=========================================================="
echo "⚡️ 准备启动实验，将生成的日志名称为: ${RUN_NAME}"
echo "📦 模型列表: ${MODELS}"
echo "📅 训练数据: [${TRAIN_START}] - [${TRAIN_END}]"
echo "📅 验证/测试 : [${VAL_START}] - [${TEST_END}]"
echo "🧩 缺失模式: ${PATTERNS} | 比例: ${PIS}"
echo "🔄 Epochs: ${EPOCHS} | HPO 寻参: ${HPO_TRIALS}次"
echo "=========================================================="

# 跑环境内的 main.py 程序
uv run python main.py \
  --models "${MODELS}" \
  --patterns "${PATTERNS}" \
  --pis "${PIS}" \
  --train-start "${TRAIN_START}" \
  --train-end "${TRAIN_END}" \
  --val-start "${VAL_START}" \
  --val-end "${VAL_END}" \
  --test-start "${TEST_START}" \
  --test-end "${TEST_END}" \
  --epochs "${EPOCHS}" \
  --hpo-trials "${HPO_TRIALS}" \
  --quiet-train \
  --run-name "${RUN_NAME}"

echo "=========================================================="
echo "✅ 跑批完毕！"
echo "你可以前往 logs/latest 或者打开 logs/summary_all_parts.csv 查看最新汇总数据！"
echo "=========================================================="
