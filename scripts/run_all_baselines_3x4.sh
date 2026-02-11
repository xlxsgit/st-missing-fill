#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

PATTERNS="${PATTERNS:-mcar,seq,scm}"
PIS="${PIS:-0.1,0.2,0.3,0.4}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_WINDOWS="${MAX_WINDOWS:-}"
WATERMARK="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.0}"

COMMON_ARGS=(
  --patterns "$PATTERNS"
  --pis "$PIS"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
)

if [[ -n "$MAX_WINDOWS" ]]; then
  COMMON_ARGS+=(--max-windows "$MAX_WINDOWS")
fi

run_part() {
  local models="$1"
  local run_name="$2"
  echo ">>> Running $run_name | models=$models | patterns=$PATTERNS | pis=$PIS"
  PYTORCH_MPS_HIGH_WATERMARK_RATIO="$WATERMARK" \
    "$PYTHON_BIN" "$ROOT_DIR/main.py" \
    --models "$models" \
    "${COMMON_ARGS[@]}" \
    --run-name "$run_name"
}

run_part "locf,knn,mice,vcaan" "part1_light_01234"
run_part "saits" "part2_saits_01234"
run_part "grud" "part3_grud_01234"
run_part "usgan" "part4_usgan_01234"
run_part "itransformer" "part5_itransformer_01234"

echo "All batches finished."
