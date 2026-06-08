#!/usr/bin/env bash
# Generic torchair migration driver: prepare a model, run a graph-mode sanity
# inference, then a 3-backend (torchair / npu_eager / cpu) accuracy+perf benchmark.
#
# Usage:
#   bash run_benchmark.sh <model_name_or_path> [output_subdir]
#
# Examples:
#   bash run_benchmark.sh google/siglip2-base-patch16-224
#   bash run_benchmark.sh /data/models/my-vit  my-vit
#
# Tunables (env): NPU_ID DTYPE BATCH_SIZE WARMUP ITERS BACKENDS OUT_DIR
#                 WEIGHT_MAX_TIME SKIP_DOWNLOAD HF_ENDPOINT TRUST_REMOTE_CODE
set -uo pipefail

MODEL="${1:?usage: run_benchmark.sh <model_name_or_path> [output_subdir]}"
SUBDIR="${2:-$(basename "${MODEL}")}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NPU_ID="${NPU_ID:-0}"
DTYPE="${DTYPE:-float16}"                 # float16|bfloat16|float32 (NPU); CPU golden is fp32
BATCH_SIZE="${BATCH_SIZE:-1}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-100}"
BACKENDS="${BACKENDS:-torchair,npu_eager,cpu}"
OUT_DIR="${OUT_DIR:-benchmark_results}"
WEIGHT_MAX_TIME="${WEIGHT_MAX_TIME:-1800}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
TRC=""; [ "${TRUST_REMOTE_CODE:-0}" = "1" ] && TRC="--trust_remote_code"

# Source CANN runtime (provides ATC/ACL libs torchair needs).
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && source /usr/local/Ascend/ascend-toolkit/set_env.sh
log() { echo -e "\n\033[1;34m[run] $*\033[0m"; }

# If MODEL is a repo id (not an existing dir) and download is allowed, fetch the
# minimum files from the mirror. Only KEEP a weights file that downloaded fully
# (curl exit 0) — a truncated partial would crash from_pretrained.
resolve_model() {
  local m="$1"
  if [ -d "${m}" ]; then echo "${m}"; return 0; fi
  if [ "${SKIP_DOWNLOAD}" = "1" ]; then echo "${m}"; return 0; fi  # let HF cache handle it
  local dest="models/$(basename "${m}")"
  if [ -s "${dest}/model.safetensors" ] || [ -s "${dest}/pytorch_model.bin" ]; then echo "${dest}"; return 0; fi
  mkdir -p "${dest}"
  local base="${HF_ENDPOINT}/${m}/resolve/main"
  log "fetching ${m} from ${HF_ENDPOINT} (budget ${WEIGHT_MAX_TIME}s) ..." 1>&2
  for f in config.json preprocessor_config.json tokenizer.json; do
    curl -fsSL --retry 3 --connect-timeout 15 --max-time 120 -o "${dest}/${f}" "${base}/${f}" 2>/dev/null || true
  done
  for w in model.safetensors pytorch_model.bin; do
    if curl -fSL --retry 3 --connect-timeout 15 --max-time "${WEIGHT_MAX_TIME}" \
            -o "${dest}/${w}.part" "${base}/${w}" 2>/dev/null; then
      mv "${dest}/${w}.part" "${dest}/${w}"; break
    else rm -f "${dest}/${w}.part"; fi
  done
  if [ -s "${dest}/model.safetensors" ] || [ -s "${dest}/pytorch_model.bin" ]; then echo "${dest}"; else echo "${m}"; fi
}

MP="$(resolve_model "${MODEL}")"
log "model resolved to: ${MP}"

log "=== graph-mode sanity inference (torchair) ==="
python "${HERE}/torch_air_infer.py" --model_name_or_path "${MP}" ${TRC} \
  --device npu --npu_id "${NPU_ID}" --backend torchair --dtype "${DTYPE}" \
  --batch_size "${BATCH_SIZE}" --warmup 3 --iterations 10

log "=== 3-backend accuracy + performance benchmark ==="
python "${HERE}/benchmark.py" --model_name_or_path "${MP}" ${TRC} \
  --backends "${BACKENDS}" --device npu --npu_id "${NPU_ID}" --dtype "${DTYPE}" \
  --batch_size "${BATCH_SIZE}" --warmup "${WARMUP}" --iterations "${ITERS}" \
  --output_dir "${OUT_DIR}/${SUBDIR}"

log "done. results: ${OUT_DIR}/${SUBDIR}/benchmark_results.{json,md}"
