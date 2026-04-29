#!/bin/bash
# vLLM 服务启动脚本
# 无硬编码，所有参数通过环境变量或命令行传入

set -e

# 参数（全部可配置）
CONTAINER_NAME="${CONTAINER_NAME:-}"
MODEL_PATH="${MODEL_PATH:-}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-5000}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.94}"
LOG_FILE="${LOG_FILE:-}"
MODE="${MODE:-local}"
SERVER_IP="${SERVER_IP:-}"
SSH_USER="${SSH_USER:-root}"
HOST="${HOST:-0.0.0.0}"

# 解析命令行参数（覆盖环境变量）
while [[ $# -gt 0 ]]; do
  case $1 in
    --container) CONTAINER_NAME="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --tp-size) TP_SIZE="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --gpu-memory-util) GPU_MEMORY_UTIL="$2"; shift 2 ;;
    --log-file) LOG_FILE="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --server) SERVER_IP="$2"; shift 2 ;;
    --user) SSH_USER="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

# 参数校验
if [[ -z "$CONTAINER_NAME" ]] || [[ -z "$MODEL_PATH" ]]; then
  echo "Error: --container and --model-path are required"
  exit 1
fi

if [[ -z "$LOG_FILE" ]]; then
  echo "Error: --log-file or LOG_FILE is required"
  exit 1
fi

# 环境变量配置
ENV_VARS="
export PYTORCH_NPU_ALLOC_CONF=\"expandable_segments:True\"
export HCCL_OP_EXPANSION_MODE=\"AIV\"
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
export TASK_QUEUE_ENABLE=1
"

# vLLM 启动命令
VLLM_CMD="
$ENV_VARS
vllm serve $MODEL_PATH \\
    --served-model-name \"$(basename $MODEL_PATH)\" \\
    --host 0.0.0.0 \\
    --port $PORT \\
    --tensor-parallel-size $TP_SIZE \\
    --max-model-len $MAX_MODEL_LEN \\
    --max-num-batched-tokens 16384 \\
    --max-num-seqs 128 \\
    --gpu-memory-utilization $GPU_MEMORY_UTIL \\
    --trust-remote-code \\
    --async-scheduling \\
    --enforce-eager \\
    > $LOG_FILE 2>&1 &
"

# 执行启动
if [[ "$MODE" == "remote" ]]; then
  echo "远程启动服务: ${SERVER_IP}"
  ssh ${SSH_USER}@${SERVER_IP} "docker exec -d $CONTAINER_NAME bash -c '$VLLM_CMD'"
else
  echo "本地启动服务"
  docker exec -d $CONTAINER_NAME bash -c "$VLLM_CMD"
fi

echo "服务启动中..."
echo "日志文件：$LOG_FILE"
echo "访问地址：http://<server-ip>:$PORT"
