#!/bin/bash
# 服务监控脚本 - 检测日志状态并调用 welink-notify skill 发送通知
# 无硬编码，所有参数通过环境变量或命令行传入

set -e

# 参数（全部可配置，无默认硬编码值）
CONTAINER_NAME="${CONTAINER_NAME:-}"
LOG_FILE="${LOG_FILE:-}"
MODEL_PATH="${MODEL_PATH:-}"
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"
NOTIFY_MODE="${NOTIFY_MODE:-welink}"
RECEIVER="${RECEIVER:-}"
AUTH_TOKEN="${AUTH_TOKEN:-}"
MODE="${MODE:-local}"
SERVER_IP="${SERVER_IP:-}"
SSH_USER="${SSH_USER:-root}"
SUCCESS_KEYWORD="${SUCCESS_KEYWORD:-Uvicorn running}"
ERROR_KEYWORD="${ERROR_KEYWORD:-Error|Exception|Failed|Traceback}"

# 解析命令行参数（覆盖环境变量）
while [[ $# -gt 0 ]]; do
  case $1 in
    --container) CONTAINER_NAME="$2"; shift 2 ;;
    --log-file) LOG_FILE="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --interval) CHECK_INTERVAL="$2"; shift 2 ;;
    --notify) NOTIFY_MODE="$2"; shift 2 ;;
    --receiver) RECEIVER="$2"; shift 2 ;;
    --auth-token) AUTH_TOKEN="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --server) SERVER_IP="$2"; shift 2 ;;
    --user) SSH_USER="$2"; shift 2 ;;
    --success-keyword) SUCCESS_KEYWORD="$2"; shift 2 ;;
    --error-keyword) ERROR_KEYWORD="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

# 参数校验
if [[ -z "$CONTAINER_NAME" ]]; then
  echo "Error: --container or CONTAINER_NAME is required"
  exit 1
fi

if [[ -z "$LOG_FILE" ]]; then
  echo "Error: --log-file or LOG_FILE is required"
  exit 1
fi

if [[ -z "$RECEIVER" ]] && [[ "$NOTIFY_MODE" == "welink" ]]; then
  echo "Error: WeLink notification mode requires --receiver or RECEIVER"
  exit 1
fi

# 获取最新日志
get_latest_logs() {
  if [[ "$MODE" == "remote" ]]; then
    ssh ${SSH_USER}@${SERVER_IP} "docker exec $CONTAINER_NAME tail -n 20 $LOG_FILE"
  else
    docker exec "$CONTAINER_NAME" tail -n 20 "$LOG_FILE"
  fi
}

# 检测启动成功
check_success() {
  local logs="$1"
  if echo "$logs" | grep -q "$SUCCESS_KEYWORD"; then
    return 0
  fi
  return 1
}

# 检测错误
check_error() {
  local logs="$1"
  if echo "$logs" | grep -qiE "$ERROR_KEYWORD"; then
    return 0
  fi
  return 1
}

# 调用 welink-notify skill API 发送通知
send_welink() {
  local status="$1"
  local message="$2"
  
  if [[ -z "$RECEIVER" ]] || [[ -z "$AUTH_TOKEN" ]]; then
    echo "Warning: RECEIVER or AUTH_TOKEN not configured, skipping notification"
    return
  fi
  
  # 构建 JSON payload
  local payload="{\"receiver\":\"${RECEIVER}\",\"auth\":\"${AUTH_TOKEN}\",\"content\":\"${message}\"}"
  
  # 调用小鲁班 API (welink-notify skill 底层接口)
  if [[ "$MODE" == "remote" ]]; then
    ssh ${SSH_USER}@${SERVER_IP} "curl -s -X POST http://xiaoluban.rnd.huawei.com:80/ -H 'Content-Type: application/json' -d '${payload}'"
  else
    curl -s -X POST http://xiaoluban.rnd.huawei.com:80/ -H "Content-Type: application/json" -d "${payload}"
  fi
}

# 主循环
echo "开始监控: $CONTAINER_NAME"
echo "日志文件：$LOG_FILE"
echo "检查间隔：${CHECK_INTERVAL}s"

PREV_STATUS=""

while true; do
  LOGS=$(get_latest_logs)
  
  if check_success "$LOGS"; then
    STATUS="running"
    if [[ -n "$MODEL_PATH" ]]; then
      MSG="✅ 服务启动成功\n\n访问地址：http://<server-ip>:8000\n模型：$(basename $MODEL_PATH)"
    else
      MSG="✅ 服务启动成功\n\n访问地址：http://<server-ip>:8000"
    fi
  elif check_error "$LOGS"; then
    STATUS="error"
    MSG="❌ 服务启动失败\n\n错误日志:\n$(echo "$LOGS" | grep -i "error\|exception" | tail -5)"
  else
    STATUS="starting"
    MSG="⏳ 服务启动中...\n\n最新日志:\n$(echo "$LOGS" | tail -3)"
  fi
  
  # 状态变化时发送通知
  if [[ "$STATUS" != "$PREV_STATUS" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 状态变更：$PREV_STATUS -> $STATUS"
    
    if [[ "$NOTIFY_MODE" == "welink" ]]; then
      send_welink "$STATUS" "$MSG"
    fi
    
    PREV_STATUS="$STATUS"
    
    # 终态退出
    if [[ "$STATUS" == "running" ]] || [[ "$STATUS" == "error" ]]; then
      echo "监控完成"
      break
    fi
  fi
  
  sleep $CHECK_INTERVAL
done
