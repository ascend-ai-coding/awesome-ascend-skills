#!/bin/bash
# 容器创建脚本 - 支持本地/远程部署
# 完整 NPU 设备映射和卷挂载配置
# 无硬编码，所有参数通过环境变量或命令行传入

set -e

# 参数（全部可配置，无默认硬编码值）
MODE="${MODE:-local}"
SERVER_IP="${SERVER_IP:-}"
SSH_USER="${SSH_USER:-root}"
IMAGE="${IMAGE:-vllm-ascend:latest}"
MODEL_PATH="${MODEL_PATH:-}"
CONTAINER_NAME="${CONTAINER_NAME:-}"
PORT="${PORT:-8000}"
WORK_DIR="${WORK_DIR:-/mnt2/hbw}"
NPU_COUNT="${NPU_COUNT:-}"  # 可手动指定，否则自动检测

# 解析命令行参数（覆盖环境变量）
while [[ $# -gt 0 ]]; do
  case $1 in
    --mode) MODE="$2"; shift 2 ;;
    --server) SERVER_IP="$2"; shift 2 ;;
    --user) SSH_USER="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --container-name) CONTAINER_NAME="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --npu-count) NPU_COUNT="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

# 参数校验
if [[ -z "$MODEL_PATH" ]]; then
  echo "Error: --model-path or MODEL_PATH is required"
  exit 1
fi

if [[ -z "$CONTAINER_NAME" ]]; then
  CONTAINER_NAME="vllm-$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')"
fi

# 执行远程命令的辅助函数
run_cmd() {
  if [[ "$MODE" == "remote" ]]; then
    ssh ${SSH_USER}@${SERVER_IP} "$1"
  else
    eval "$1"
  fi
}

# 检测 NPU 卡数（如果未手动指定）
detect_npu_count() {
  if [[ -n "$NPU_COUNT" ]]; then
    echo "$NPU_COUNT"
    return
  fi
  
  local count
  if [[ "$MODE" == "remote" ]]; then
    count=$(ssh ${SSH_USER}@${SERVER_IP} "npu-smi info 2>/dev/null | grep -c 'Ascend'" || echo "0")
  else
    count=$(npu-smi info 2>/dev/null | grep -c 'Ascend' || echo "0")
  fi
  
  if [[ "$count" -eq 0 ]]; then
    echo "Warning: 无法检测 NPU 卡数，使用默认值 8" >&2
    count=8
  fi
  
  echo "$count"
}

NPU_COUNT=$(detect_npu_count)
TP_SIZE=$NPU_COUNT

echo "检测到 NPU 卡数：$NPU_COUNT, TP_SIZE=$TP_SIZE"

# 构建动态 NPU 设备映射
build_npu_devices() {
  local devices=""
  for ((i=0; i<NPU_COUNT; i++)); do
    devices+="--device=/dev/davinci$i "
  done
  devices+="--device=/dev/davinci_manager "
  devices+="--device=/dev/devmm_svm "
  devices+="--device=/dev/hisi_hdc "
  echo "$devices"
}

NPU_DEVICES=$(build_npu_devices)

# 构建 docker run 命令
# 参考：https://support.huawei.com/enterprise/zh/doc/EDOC1100273826
DOCKER_CMD="docker run -itd \
    --network host \
    --shm-size 16G \
    --privileged \
    ${NPU_DEVICES}\
    -v /var/log/npu/:/usr/slog \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /usr/local/sbin/:/usr/local/sbin/ \
    -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
    -v /var/log/npu/slog/:/var/log/npu/slog \
    -v /var/log/npu/profiling/:/var/log/npu/profiling \
    -v /var/log/npu/dump/:/var/log/npu/dump \
    -v /usr/lib/jvm/:/usr/lib/jvm \
    -v ${MODEL_PATH}:${MODEL_PATH} \
    -v ${WORK_DIR}:${WORK_DIR} \
    -w ${WORK_DIR} \
    --name=${CONTAINER_NAME} \
    --entrypoint=/bin/bash \
    ${IMAGE}"

# 执行创建
if [[ "$MODE" == "remote" ]]; then
  echo "远程创建容器：${SERVER_IP}"
  ssh ${SSH_USER}@${SERVER_IP} "$DOCKER_CMD"
else
  echo "本地创建容器"
  eval "$DOCKER_CMD"
fi

echo ""
echo "✅ 容器创建成功：$CONTAINER_NAME"
echo "   NPU 卡数: $NPU_COUNT"
echo "   TP_SIZE: $TP_SIZE"
echo ""
echo "下一步：运行 scripts/start_service.sh 启动 vLLM 服务"
