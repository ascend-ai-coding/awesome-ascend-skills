#!/usr/bin/env bash
set -euo pipefail

# All paths are driven by environment variables — no hardcoded paths.
# Required: TTK_DIR, OPS_NN_CONV_DIR, CASES_CSV, OUT_DIR
# Required: CONDA_SH, CONDA_ENV, CANN_SETENV (or pre-sourced environment)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GUIDE_DIR="${SCRIPT_DIR}"

# --- environment ---
if [[ -n "${CONDA_SH:-}" ]]; then
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV:-base}"
fi
if [[ -n "${CANN_SETENV:-}" ]]; then
    source "${CANN_SETENV}"
fi

# --- validate required vars ---
for var in TTK_DIR OPS_NN_CONV_DIR CASES_CSV OUT_DIR; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: ${var} is not set. Export it before running this script."
        echo "  export TTK_DIR=<path to ops-test-kit>"
        echo "  export OPS_NN_CONV_DIR=<path to ops-nn/conv/conv2d_v2>"
        echo "  export CASES_CSV=<path to cases.csv>"
        echo "  export OUT_DIR=<output directory>"
        exit 1
    fi
done

# --- golden / input plugins ---
GOLDEN="${OPS_NN_CONV_DIR}/tests/assets/golden.py"
INPUT="${OPS_NN_CONV_DIR}/tests/assets/input.py"
if [[ ! -f "${GOLDEN}" ]]; then
    echo "WARN: golden.py not found at ${GOLDEN}"
fi
if [[ ! -f "${INPUT}" ]]; then
    echo "WARN: input.py not found at ${INPUT}"
fi

# --- run TTK ---
cd "${TTK_DIR}"
mkdir -p "${OUT_DIR}"
RESULT_CSV="${OUT_DIR}/ttk_kernel_results.csv"

python3 -m ttk kernel \
  -i "${CASES_CSV}" \
  -d=false -b=release \
  --dev 0 --pc=1 --run=3 \
  --compare close \
  --plugin "${GOLDEN},${INPUT}" \
  -o "${RESULT_CSV}" 2>&1 | tee "${OUT_DIR}/ttk_run.log"

# --- post-processing ---
python3 "${GUIDE_DIR}/summarize_results.py" \
  --cases "${CASES_CSV}" \
  --results "${RESULT_CSV}" \
  --out "${OUT_DIR}/summary_table.csv"

python3 "${GUIDE_DIR}/decode_tiling.py" \
  --results "${RESULT_CSV}" \
  --out "${OUT_DIR}/tiling_decoded.json"

echo "Done: ${OUT_DIR}/summary_table.csv"
echo "Tiling: ${OUT_DIR}/tiling_decoded.csv"
