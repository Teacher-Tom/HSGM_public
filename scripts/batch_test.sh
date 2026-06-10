#!/usr/bin/env bash
# =============================================================================
# batch_test.sh — Multi-episode batch evaluation for HSGM
#
# Usage:
#   bash scripts/batch_test.sh [TASK] [BEGIN] [END] [MODEL_NAME]
#
#   TASK       : r2r | rxr | objectnav (default: r2r)
#   BEGIN      : start episode index (default: 0)
#   END        : end episode index, exclusive; -1 for all (default: -1)
#   MODEL_NAME : VLM model name (default: $OPENAI_MODEL or gpt-4o)
#
# Environment variables (required for VLM):
#   OPENAI_BASE_URL  — OpenAI-compatible API endpoint
#   OPENAI_API_KEY   — API key ("EMPTY" for local vLLM)
#   OPENAI_MODEL     — model name (used as fallback)
#
# Additional controls:
#   MAX_STEPS    — max agent steps per episode (default: 100)
#   COMMENT      — run comment tag (default: batch_test)
#
# Examples:
#   export OPENAI_BASE_URL=http://localhost:8000/v1
#   export OPENAI_API_KEY=EMPTY
#   bash scripts/batch_test.sh r2r 0 10 Qwen/Qwen3.6-27B
#
#   # With custom comment and step limit
#   COMMENT=my_run MAX_STEPS=150 bash scripts/batch_test.sh r2r 0 50
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${PROJECT_ROOT}/src"

TASK="${1:-r2r}"
BEGIN="${2:-0}"
END="${3:--1}"
MODEL_NAME="${4:-${OPENAI_MODEL:-gpt-4o}}"
MAX_STEPS="${MAX_STEPS:-100}"
COMMENT="${COMMENT:-batch_test}"

case "${TASK}" in
    r2r)       CONFIG="../config/vlnce_test.yaml" ;;
    rxr)       CONFIG="../config/vlnce_test.yaml" ;;
    objectnav) CONFIG="../config/objnav_test.yaml" ;;
    *) echo "[ERROR] Unknown task: ${TASK} (use: r2r | rxr | objectnav)"; exit 1 ;;
esac

echo "============================================================"
echo " HSGM Batch Test"
echo "   task        : ${TASK}"
echo "   episodes    : [${BEGIN}, ${END})"
echo "   model       : ${MODEL_NAME}"
echo "   max steps   : ${MAX_STEPS}"
echo "   comment     : ${COMMENT}"
echo "   config      : ${CONFIG}"
echo "   base_url    : ${OPENAI_BASE_URL:-not set}"
echo "============================================================"

cd "${SRC_DIR}"

python run_experiments.py \
    --task "${TASK}" \
    --config "${CONFIG}" \
    --model_name "${MODEL_NAME}" \
    --begin_idx "${BEGIN}" \
    --end_idx "${END}" \
    --max_steps "${MAX_STEPS}" \
    --comment "${COMMENT}"
