#!/bin/bash
# run_dapo.sh - Verl 异步 DAPO 训练启动脚本
#
# 特性管理策略：
# 1. 用户未指定 → 使用默认性能特性（全开）
# 2. 用户显式指定 → 优先级最高
# 3. OOM 时 → 自动追加显存特性重试（即使指定了特性也允许追加）
#
# 监控策略：
# - 启动后输出 SwanLab 链接
# - 后台静默运行
# - 仅在错误时通知用户

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"
source "$SCRIPT_DIR/common.sh"
source "$SCRIPT_DIR/feature_manager.sh"

cd /verl

# ==========================================
# 配置参数
# ==========================================
FRAMEWORK="${FRAMEWORK:-megatron}"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_8B}"
TRAIN_FILE="${TRAIN_FILE:-$DEFAULT_TRAIN_DATA}"
VAL_FILE="${VAL_FILE:-$DEFAULT_VAL_DATA}"
TRAIN_STEPS="${TRAIN_STEPS:-100}"
EXP_NAME="${EXP_NAME:-DAPO-Qwen3-8b-${FRAMEWORK}-async-$(date +%m%d_%H%M)}"
PROJECT_NAME="${PROJECT_NAME:-$DEFAULT_PROJECT_NAME}"
CKPTS_DIR="${CKPTS_DIR:-/mnt2/jins_ckpt/${EXP_NAME}}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
TRAINER_GPUS="${TRAINER_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"

# SwanLab 配置（从环境变量或配置文件加载）
SWANLAB_USER="${SWANLAB_USER:-TrainingMaster}"

# 用户显式指定特性 (可选)
USER_FEATURES="${USER_FEATURES:-}"

# 最大 OOM 重试次数
MAX_OOM_RETRIES="${MAX_OOM_RETRIES:-2}"

# ==========================================
# 路径验证
# ==========================================
info "验证路径..."
validate_all_paths "$MODEL_PATH" "$TRAIN_FILE" "$VAL_FILE"
if [ $? -gt 0 ]; then
    fail "路径验证失败，请检查后重试"
    exit 1
fi

# ==========================================
# 环境准备
# ==========================================
cleanup_ray
load_proxy_config
setup_proxy

# 初始化 SwanLab 配置（从环境变量或配置文件加载）
init_swanlab_config

# 启动 Ray 集群（NPU 环境需显式声明 GPU 数）
TOTAL_GPUS=$((TRAINER_GPUS + ROLLOUT_GPUS))
start_ray_head $TOTAL_GPUS 8265 || exit 1

export VLLM_ASCEND_ENABLE_NZ=0
export HCCL_EXEC_TIMEOUT=60000
export HCCL_CONNECT_TIMEOUT=7200

# ==========================================
# 训练执行函数
# ==========================================

run_training() {
    local features="$1"
    local log_file="${2:-/verl/train.log}"

    # 清空日志
    : > "$log_file"

    # 生成训练脚本
    python3 "$SCRIPT_DIR/config_generator.py" \
        --framework "$FRAMEWORK" \
        --model "$MODEL_PATH" \
        --train-data "$TRAIN_FILE" \
        --val-data "$VAL_FILE" \
        --steps "$TRAIN_STEPS" \
        --exp-name "$EXP_NAME" \
        --project "$PROJECT_NAME" \
        --ckpt-dir "$CKPTS_DIR" \
        --lr "$LEARNING_RATE" \
        --trainer-gpus "$TRAINER_GPUS" \
        --rollout-gpus "$ROLLOUT_GPUS" \
        --feature $features \
        --output /tmp/run_verl_temp.sh

    # 执行训练 (后台)
    nohup bash /tmp/run_verl_temp.sh >> "$log_file" 2>&1 &
    local train_pid=$!

    echo "$train_pid" > /verl/train.pid

    # 通过 stdout 输出 PID（供调用方捕获）
    echo "$train_pid"
}

# ==========================================
# 输出 SwanLab 链接
# ==========================================

print_swanlab_links() {
    echo ""
    echo "=========================================="
    echo "训练已启动！"
    echo "=========================================="
    echo ""
    echo "SwanLab 监控:"
    echo "  Project: ${SWANLAB_HOST}/@${SWANLAB_USER}/${PROJECT_NAME}"
    echo "  Run:     ${SWANLAB_HOST}/@${SWANLAB_USER}/${PROJECT_NAME}/runs/${EXP_NAME}"
    echo ""
    echo "查看日志:"
    echo "  docker exec {container} tail -f /verl/train.log"
    echo ""
    echo "检查状态:"
    echo "  docker exec {container} grep 'Training Progress' /verl/train.log | tail -1"
    echo ""
    echo "=========================================="
    echo "训练将在后台静默运行，仅在错误时通知您。"
    echo "=========================================="
}

# ==========================================
# 错误监控函数
# ==========================================

monitor_for_errors() {
    local log_file="/verl/train.log"
    local pid_file="/verl/train.pid"
    
    while true; do
        sleep 30
        
        # 检查进程是否还在运行
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            if ! kill -0 $pid 2>/dev/null; then
                # 进程已退出，检查是否成功
                if grep -q "Training completed successfully" "$log_file" 2>/dev/null; then
                    echo "[INFO] 训练成功完成！"
                    echo "[INFO] Checkpoint: $CKPTS_DIR"
                    return 0
                else
                    # 训练失败
                    echo "[ERROR] 训练异常退出！"
                    echo "[ERROR] 查看日志: tail -50 $log_file"
                    tail -50 "$log_file"
                    return 1
                fi
            fi
        fi
        
        # 检查是否有 OOM
        if check_oom "$log_file"; then
            echo "[ERROR] 检测到 OOM 错误！"
            return 2  # OOM 特殊退出码
        fi
    done
}

# ==========================================
# OOM 自动重试主循环
# ==========================================

main() {
    local oom_retry=0
    
    while true; do
        # 设置 OOM 重试计数 (供 feature_manager 使用)
        export OOM_RETRY_COUNT=$oom_retry
        
        # 确定特性配置
        FEATURES=$(determine_features)
        
        # 显示启动信息
        info "=========================================="
        info "启动 Verl 异步 DAPO 训练"
        info "=========================================="
        info "框架: $FRAMEWORK"
        info "项目: $PROJECT_NAME / 实验: $EXP_NAME"
        info "步数: $TRAIN_STEPS"
        info "资源: trainer=$TRAINER_GPUS GPUs, rollout=$ROLLOUT_GPUS GPUs"
        info "特性: $FEATURES"
        info "=========================================="
        
        # 运行训练（通过 echo 捕获 PID）
        local train_pid
        train_pid=$(run_training "$FEATURES" "/verl/train.log")
        
        # 输出 SwanLab 链接
        print_swanlab_links
        
        # 监控错误
        monitor_for_errors
        local monitor_result=$?
        
        # 训练成功
        if [ $monitor_result -eq 0 ]; then
            return 0
        fi
        
        # OOM 错误
        if [ $monitor_result -eq 2 ]; then
            oom_retry=$((oom_retry + 1))
            
            if [ $oom_retry -le $MAX_OOM_RETRIES ]; then
                warn "[OOM] 尝试追加显存优化特性重试..."
                warn "[OOM] 重试 $oom_retry / $MAX_OOM_RETRIES"
                
                # 清理环境
                cleanup_ray
                sleep 10
            else
                fail "[OOM] 已达到最大重试次数 ($MAX_OOM_RETRIES)"
                fail "[OOM] 建议: 减小 batch size 或使用更小模型"
                return 1
            fi
        else
            # 其他错误
            return 1
        fi
    done
}

# ==========================================
# 入口
# ==========================================

main