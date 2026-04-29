#!/bin/bash
# ============================================================================
# start_verl.sh — 上层脚本模板
# 职责：NPU 配置、网络检测、Ray 集群启动、SwanLab 环境变量、调用下层训练脚本
#
# 由 generate_training.sh 通过 sed 替换占位符生成最终脚本
# ============================================================================

# ==========================================
# 1. NPU 卡号配置
# ==========================================
export ASCEND_RT_VISIBLE_DEVICES="NPU_DEVICES_PLACEHOLDER"

# 自动计算卡数
DEVICE_COUNT=$(echo "$ASCEND_RT_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

# ==========================================
# 2. 集群拓扑配置
# ==========================================
export NNODES="${NNODES:-1}"
export NPU_PER_NODE="${NPU_PER_NODE:-$DEVICE_COUNT}"

# ==========================================
# 3. 网络配置
# ==========================================
_auto_detect_iface() {
    local iface
    iface=$(grep -Po '^(\S+)(?=\s+00000000)' /proc/net/route 2>/dev/null | head -n 1)
    if [ -z "$iface" ] && command -v ip >/dev/null 2>&1; then
        iface=$(ip route get 1.1.1.1 2>/dev/null | grep -oP 'dev \K\S+')
    fi
    if [ -z "$iface" ]; then
        iface=$(ls /sys/class/net 2>/dev/null | grep -vE 'lo|docker|veth' | head -n 1)
    fi
    echo "$iface"
}

_auto_detect_ip() {
    local res=""
    local iface="SOCKET_IFNAME_PLACEHOLDER"
    if [ -n "$iface" ]; then
        if command -v ip >/dev/null 2>&1; then
            res=$(ip -4 addr show "$iface" 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -n 1)
        elif command -v ifconfig >/dev/null 2>&1; then
            res=$(ifconfig "$iface" 2>/dev/null | grep -Eo '([0-9]{1,3}\.){3}[0-9]{1,3}' | grep -v '255$' | head -n 1)
        fi
    fi
    if [ -z "$res" ]; then
        res=$(python3 -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print(s.getsockname()[0])" 2>/dev/null)
    fi
    if [ -z "$res" ]; then
        res=$(hostname -I 2>/dev/null | awk '{print $1}')
    fi
    echo "$res"
}

SOCKET_IFNAME="SOCKET_IFNAME_PLACEHOLDER"
export HCCL_SOCKET_IFNAME="$SOCKET_IFNAME"
export GLOO_SOCKET_IFNAME="$SOCKET_IFNAME"

CURRENT_IP="CURRENT_IP_PLACEHOLDER"
MASTER_ADDR="MASTER_ADDR_PLACEHOLDER"

# ==========================================
# 4. Ray 端口配置
# ==========================================
RAY_PORT="${RAY_PORT:-9182}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8260}"

# ==========================================
# 5. SwanLab 配置
# === SWANLAB_CONFIG_START ===
export SWANLAB_HOST="${SWANLAB_HOST:?SWANLAB_HOST 未设置}"
export SWANLAB_API_KEY="${SWANLAB_API_KEY:?SWANLAB_API_KEY 未设置}"
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
export SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE:-TrainingMaster}"
export SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-/mnt/project/verl/swanlab_logs}"
export PROJECT_NAME="${PROJECT_NAME:-verl_hlm}"
# === SWANLAB_CONFIG_END ===

# ==========================================
# 6. 训练脚本配置
# ==========================================
DEFAULT_SH="./run_training.sh"

# ==========================================
# 7. 环境设置
# ==========================================
pkill -9 python 2>/dev/null || true
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray

export RAY_DEDOP_LOGS=0
export HYDRA_FULL_ERROR=1
export TASK_QUEUE_ENABLE=1
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600

export VLLM_ASCEND_ENABLE_NZ=0

ulimit -n 32768
mkdir -p logs

# ==========================================
# 8. 打印配置摘要
# ==========================================
echo "=========================================="
echo "Verl 训练启动配置"
echo "=========================================="
echo "  NPU 卡号:       $ASCEND_RT_VISIBLE_DEVICES (${DEVICE_COUNT} 卡)"
echo "  节点数:         $NNODES"
echo "  每节点 NPU 数:  $NPU_PER_NODE"
echo "  通信网卡:       $SOCKET_IFNAME"
echo "  当前 IP:        $CURRENT_IP"
echo "  主节点 IP:      $MASTER_ADDR"
echo "  特性掩码:       FEATURE_MASK_PLACEHOLDER"
echo "  SwanLab:        $([ -n "${SWANLAB_API_KEY:-}" ] && echo '已配置 ('"${SWANLAB_API_KEY:0:8}"'...)' || echo '未配置')"
echo "=========================================="

# ==========================================
# 9. 启动 Ray 集群
# ==========================================
if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
  ray start --head --port "$RAY_PORT" \
    --dashboard-host="$MASTER_ADDR" \
    --node-ip-address="$CURRENT_IP" \
    --dashboard-port="$DASHBOARD_PORT" \
    --resources='{"NPU": '"$NPU_PER_NODE"'}' \
    --temp-dir=/tmp/ray_session_$(date +%s)

  while true; do
      ray_status_output=$(ray status 2>&1) || true
      npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1) || npu_count=""
      if [ -z "$npu_count" ]; then
          echo "等待 Ray 就绪..."
          sleep 5
          continue
      fi
      npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
      device_count=$((npu_count_int / NPU_PER_NODE))

      if [ "$device_count" -eq "$NNODES" ]; then
          echo "Ray 集群就绪: $device_count 个节点, 启动训练"
          ray status || true
          bash "$DEFAULT_SH"
          break
      else
          echo "等待 Ray 分配 $NNODES 个节点, 当前: $device_count"
          sleep 5
      fi
  done
else
  while true; do
      ray start --address="$MASTER_ADDR:$RAY_PORT" \
        --resources='{"NPU": '"$NPU_PER_NODE"'}' \
        --node-ip-address="$CURRENT_IP" || true

      if ray status 2>&1; then
          echo "成功连接到 Ray 集群!"
          break
      else
          echo "连接 Ray 集群失败, 5 秒后重试..."
          sleep 5
      fi
  done
fi

sleep 600
