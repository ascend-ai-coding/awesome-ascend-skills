#!/bin/bash
# common.sh - Verl 异步 DAPO 共享函数库
# 被训练脚本 source 使用

set -euo pipefail

# ==========================================
# 颜色定义 (必须最先定义，供后续函数使用)
# ==========================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ==========================================
# 打印函数
# ==========================================

info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; }

# ==========================================
# SwanLab Skill 集成
# ==========================================

# 查找 swanlab-setup skill 路径
_swanlab_setup_path() {
    local skill_dir="${HOME}/.claude/skills/swanlab-setup"
    # Windows 路径兼容
    if [ ! -d "$skill_dir" ]; then
        skill_dir="/root/.claude/skills/swanlab-setup"
    fi
    local functions_file="${skill_dir}/scripts/functions.sh"
    if [ -f "$functions_file" ]; then
        echo "$functions_file"
        return 0
    fi
    return 1
}

# 加载 SwanLab 函数（失败时打印警告但不阻断）
if _swanlab_path=$(_swanlab_setup_path) 2>/dev/null; then
    # shellcheck source=/dev/null
    source "$_swanlab_path"
    info "SwanLab 函数库已加载"
else
    # 提供哑桩函数，避免其他脚本因函数不存在而失败
    warn "未找到 swanlab-setup skill，SwanLab 相关功能不可用"
    init_swanlab_config() { :; }
    setup_swanlab_in_container() { :; }
    setup_swanlab_for_container() { :; }
fi

# ==========================================
# 默认配置
# ==========================================

DEFAULT_CONTAINER="jins"
DEFAULT_PROJECT_NAME="verl_async_dapo"
DEFAULT_MODEL_8B="/mnt/public/models/huggface_models/Qwen/Qwen3-8B"
DEFAULT_MODEL_06B="/mnt/public/models/huggface_models/Qwen/Qwen3-0.6B"
DEFAULT_TRAIN_DATA="/mnt2/wql/data/dapo-math-17k.parquet"
DEFAULT_VAL_DATA="/mnt2/wql/data/aime-2024.parquet"
DEFAULT_PROXY=""

# ==========================================
# 代理配置
# ==========================================

load_proxy_config() {
    # 同时检查大写和小写环境变量
    if [ -n "${http_proxy:-}" ]; then
        DEFAULT_PROXY="$http_proxy"
    elif [ -n "${HTTP_PROXY:-}" ]; then
        DEFAULT_PROXY="$HTTP_PROXY"
    elif [ -n "${https_proxy:-}" ]; then
        DEFAULT_PROXY="$https_proxy"
    elif [ -n "${HTTPS_PROXY:-}" ]; then
        DEFAULT_PROXY="$HTTPS_PROXY"
    fi
}

setup_proxy() {
    export http_proxy="${http_proxy:-$DEFAULT_PROXY}"
    export https_proxy="${https_proxy:-$DEFAULT_PROXY}"
}

# ==========================================
# 路径检查
# ==========================================

check_path() {
    local path="$1" desc="$2"
    if [ -z "$path" ]; then
        warn "${desc}: 未指定"
        return 1
    fi
    if [ -e "$path" ]; then
        local size
        size=$( [ -f "$path" ] && du -h "$path" | cut -f1 || du -sh "$path" | cut -f1 )
        ok "${desc}: ${path} (${size})"
        return 0
    else
        fail "${desc}: ${path} 不存在!"
        return 1
    fi
}

# ==========================================
# 模型路径解析
# ==========================================

get_default_model() {
    case "${MODEL_SIZE:-8B}" in
        0.6B|0.6) echo "$DEFAULT_MODEL_06B" ;;
        *)        echo "$DEFAULT_MODEL_8B" ;;
    esac
}

# ==========================================
# 验证所有路径
# ==========================================

validate_all_paths() {
    local model_path="${1:-$(get_default_model)}"
    local train_data="${2:-$DEFAULT_TRAIN_DATA}"
    local val_data="${3:-$DEFAULT_VAL_DATA}"
    local errors=0

    echo "--- 路径验证 ---"
    check_path "$model_path" "模型" || ((errors++))
    check_path "$train_data" "训练数据" || ((errors++))
    check_path "$val_data"   "验证数据" || ((errors++))

    export VALIDATED_MODEL_PATH="$model_path"
    export VALIDATED_TRAIN_DATA="$train_data"
    export VALIDATED_VAL_DATA="$val_data"

    return $errors
}

# ==========================================
# Ray 进程清理
# ==========================================

cleanup_ray() {
    info "清理 Ray 进程..."
    ray stop --force 2>/dev/null || true
    sleep 1
    pkill -9 ray 2>/dev/null || true
    rm -rf /tmp/ray
}

# ==========================================
# Ray 启动（NPU 环境需显式声明 GPU）
# ==========================================

start_ray_head() {
    local num_gpus="${1:-8}"
    local dashboard_port="${2:-8265}"
    
    info "启动 Ray Head 节点 (GPUs: $num_gpus, Dashboard: $dashboard_port)..."
    
    # NPU 环境下 Ray 无法自动检测 GPU，必须显式声明
    ray start --head \
        --num-gpus=$num_gpus \
        --dashboard-port=$dashboard_port \
        --port=6379 \
        2>&1 | grep -E "SUCC|INFO|ERROR|WARN" | tail -10
    
    # 等待 Ray 就绪
    sleep 3
    
    # 验证 Ray 状态
    if ray status 2>&1 | grep -q "Active:"; then
        ok "Ray 集群启动成功"
    else
        fail "Ray 集群启动失败"
        return 1
    fi
}

# ==========================================
# SwanLab 配置兼容层 (调用 swanlab-setup)
# ==========================================

# 初始化 SwanLab 配置（环境变量 > 配置文件 > 交互式输入）
init_swanlab_config() {
    if declare -f swanlab_init &>/dev/null; then
        swanlab_init
    else
        warn "swanlab_init 不可用"
    fi
}

# 在容器内配置 SwanLab
setup_swanlab_in_container() {
    if declare -f swanlab_setup &>/dev/null; then
        swanlab_setup
    else
        warn "swanlab_setup 不可用"
    fi
}

# 为指定容器配置 SwanLab
setup_swanlab_for_container() {
    local container="${1:-$DEFAULT_CONTAINER}"
    if declare -f swanlab_setup_for_container &>/dev/null; then
        swanlab_setup_for_container "$container"
    else
        warn "swanlab_setup_for_container 不可用"
    fi
}

# ==========================================
# 资源分配 (异步模式)
# ==========================================

auto_allocate_gpus() {
    local npu_count="${1:-8}"
    TRAINER_GPUS="${VERL_TRAINER_GPUS:-$((npu_count / 2))}"
    ROLLOUT_GPUS="${VERL_ROLLOUT_GPUS:-$((npu_count / 2))}"
}