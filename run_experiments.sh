#!/bin/bash

# ==========================================
# 🚀 快速跑批实验启动脚本 (Fast Experiment Runner)
# ==========================================
# 你只需要在这里修改参数，然后在这个项目根目录下运行: 
# ./run_experiments.sh [可选参数: train | test | all(默认)]
# ==========================================

# 1. 想要运行的模型 (用逗号分隔，不要有空格)
# 可选项: mymodel,vcaan,locf,knn,mice,saits,itransformer,grud,usgan(❌，太慢了)
MODELS="mymodel,vcaan,locf,knn,mice,saits,itransformer,grud"

# 2. 缺失模式 (用逗号分隔，不要有空格)
# 可选项: mcar(完全随机), seq(连续块缺失), scm(空间相关缺失)
PATTERNS="mcar,seq,scm"

# 3. 缺失率 (用逗号分隔，不要有空格，结尾不要留逗号)
PIS="0.1,0.3,0.5,0.7"

# 4. 数据切分时间范围 (数据量越小跑的越快，但容易让模型欠拟合)
# Train 范围
TRAIN_START="2023-01-01"
TRAIN_END="2023-01-31"
# Val 范围
VAL_START="2023-02-01"
VAL_END="2023-02-28"
# 默认测试 范围 (用于训练过程中和紧随其后的默认评估)
TEST_START="2023-03-01"
TEST_END="2023-03-31"

# 🚀 5. 额外测试专属时间范围 (当主动执行 ./run_experiments.sh test 时启用)
EXTRA_TEST_START="2023-04-01"
EXTRA_TEST_END="2023-04-30"

# 6. 深度学习相关的训练大周期数 (Epochs)
# 快速测试写 1，要出成果写 10~30
EPOCHS=5

# 7. 超参数搜索次数 (HPO Trials)
# 为 0 则关闭模型结构参数搜索 (极速)，如果你想追求更好性能，可以设为 2, 5 甚至 10
HPO_TRIALS=2

# 8. 早停耐心值 (Patience)
# 验证集 RMSE 连续多少个 Epoch 没有下降则触发早停 (Early Stopping)
PATIENCE=1

# 9. 本次实验文件夹后缀名字
RUN_NAME="xlx"

# 10. 运行模式 (支持外部传入命令参数，例如 ./run_experiments.sh test)
# train: 只训练模型并保存 | test: 仅加载保存的模型进行推理 | all: 训练+推理
MODE="${1:-all}"


# ==========================================
# 👇 下面的核心命令不要修改 👇 
# ==========================================

echo "=========================================================="
# Start timing
START_TIME=$SECONDS
echo "⚡️ 准备启动实验，将生成的日志名称为: ${RUN_NAME}"
echo "🕹️ 当前运行模式: ${MODE}"
echo "📦 模型列表: ${MODELS}"
echo "📅 训练数据: [${TRAIN_START}] - [${TRAIN_END}]"
echo "📅 验证/默认测试 : [${VAL_START}] - [${TEST_END}]"
if [ "${MODE}" = "test" ]; then
  echo "🚀 额外测试 : [${EXTRA_TEST_START}] - [${EXTRA_TEST_END}]"
fi
echo "🧩 缺失模式: ${PATTERNS} | 比例: ${PIS}"
echo "🔄 Epochs: ${EPOCHS} | Patience: ${PATIENCE} | HPO 寻参: ${HPO_TRIALS}次"
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
  --extra-test-start "${EXTRA_TEST_START}" \
  --extra-test-end "${EXTRA_TEST_END}" \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --hpo-trials "${HPO_TRIALS}" \
  --quiet-train \
  --run-name "${RUN_NAME}" \
  --mode "${MODE}"

echo "=========================================================="
echo "✅ 跑批完毕！"
ELAPSED=$(( SECONDS - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))
SECS=$(( ELAPSED % 60 ))
echo "⏱️  整个跑批过程总耗时: ${HOURS}小时 ${MINS}分钟 ${SECS}秒"
echo "你可以前往 logs/${RUN_NAME} 查看日志、可视化图表及 summary.csv 汇总数据！"
echo "=========================================================="
