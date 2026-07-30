#!/bin/bash
# functions.sh - SwanLab 共享函数库
# 被其他 skill 通过 source 调用

# 默认配置
SWANLAB_DEFAULT_HOST="http://10.143.2.129:8000"
SWANLAB_CONFIG_DIR="${HOME}/.verl"
SWANLAB_CONFIG_FILE="${SWANLAB_CONFIG_DIR}/swanlab.conf"

# 颜色
_sl_red='\033[0;31m'
_sl_green='\033[0;32m'
_sl_yellow='\033[1;33m'
_sl_blue='\033[0;34m'
_sl_nc='\033[0m'

_sl_info()  { echo -e "${_sl_blue}[SwanLab]${_sl_nc} $*"; }
_sl_ok()    { echo -e "${_sl_green}[SwanLab]${_sl_nc} $*"; }
_sl_warn()  { echo -e "${_sl_yellow}[SwanLab]${_sl_nc} $*"; }
_sl_fail()  { echo -e "${_sl_red}[SwanLab]${_sl_nc} $*"; }

# ==========================================
# 代理设置
# ==========================================

# 清除代理设置（SwanLab 连接需要）
swanlab_clear_proxy() {
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
    _sl_ok "已清除代理设置"
}

# ==========================================
# 配置加载
# ==========================================

# 从环境变量或配置文件加载
swanlab_load_config() {
    SWANLAB_HOST="${SWANLAB_HOST:-}"
    SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"

    if [ -z "$SWANLAB_HOST" ] || [ -z "$SWANLAB_API_KEY" ]; then
        if [ -f "$SWANLAB_CONFIG_FILE" ]; then
            # 安全地逐行读取，只导出已知变量
            while IFS='=' read -r key value; do
                [[ "$key" =~ ^#.*$ ]] && continue
                [[ -z "$key" ]] && continue
                value="${value%\"}" value="${value#\"}"
                value="${value%\'}" value="${value#\'}"
                case "$key" in
                    SWANLAB_HOST)   SWANLAB_HOST="$value" ;;
                    SWANLAB_API_KEY) SWANLAB_API_KEY="$value" ;;
                esac
            done < "$SWANLAB_CONFIG_FILE"
        fi
    fi

    export SWANLAB_HOST SWANLAB_API_KEY
}

# 保存配置到文件
swanlab_save_config() {
    local host="$1"
    local api_key="$2"

    mkdir -p "$SWANLAB_CONFIG_DIR"
    cat > "$SWANLAB_CONFIG_FILE" << EOF
# SwanLab 配置文件（由 swanlab-setup skill 自动生成）
SWANLAB_HOST="$host"
SWANLAB_API_KEY="$api_key"
EOF
    chmod 600 "$SWANLAB_CONFIG_FILE"
    _sl_ok "配置已保存到 $SWANLAB_CONFIG_FILE"
}

# 交互式获取配置
swanlab_prompt_config() {
    echo ""
    echo "--- SwanLab 配置 ---"
    echo "请提供以下信息（留空使用默认值）:"

    read -p "SwanLab 主机地址 [$SWANLAB_DEFAULT_HOST]: " input_host
    SWANLAB_HOST="${input_host:-$SWANLAB_DEFAULT_HOST}"

    read -p "SwanLab API Key: " input_key
    if [ -n "$input_key" ]; then
        SWANLAB_API_KEY="$input_key"
        swanlab_save_config "$SWANLAB_HOST" "$SWANLAB_API_KEY"
    fi
}

# 按优先级初始化配置：环境变量 > 配置文件 > 交互式输入
swanlab_init() {
    swanlab_load_config

    if [ -z "$SWANLAB_HOST" ] || [ -z "$SWANLAB_API_KEY" ]; then
        if [ -t 0 ]; then
            swanlab_prompt_config
        else
            SWANLAB_HOST="${SWANLAB_HOST:-$SWANLAB_DEFAULT_HOST}"
            if [ -z "$SWANLAB_API_KEY" ]; then
                _sl_warn "SwanLab API Key 未配置，请设置 SWANLAB_API_KEY 环境变量"
            fi
        fi
    fi

    export SWANLAB_HOST SWANLAB_API_KEY
}

# ==========================================
# 安装与登录
# ==========================================

# 在当前环境执行安装和登录
swanlab_login() {
    # 确保配置已加载
    swanlab_load_config

    # 安装
    if ! pip show swanlab &>/dev/null; then
        _sl_info "安装 SwanLab..."
        pip install swanlab -q
    fi

    # 登录
    if [ -n "$SWANLAB_API_KEY" ]; then
        _sl_info "登录 SwanLab (${SWANLAB_HOST:-$SWANLAB_DEFAULT_HOST})..."
        echo "$SWANLAB_API_KEY" | swanlab login --host "${SWANLAB_HOST:-$SWANLAB_DEFAULT_HOST}" 2>&1 \
            | grep -E "Login successfully|already logged" || _sl_ok "登录完成"
    else
        _sl_warn "SwanLab API Key 未设置，跳过登录"
    fi
}

# ==========================================
# 完整设置
# ==========================================

# 当前环境完整配置（clear_proxy + init + login）
swanlab_setup() {
    swanlab_clear_proxy
    swanlab_init
    swanlab_login
}

# 为指定容器配置 SwanLab
swanlab_setup_for_container() {
    local container="${1:?用法: swanlab_setup_for_container <container_name>}"

    swanlab_init

    local host="${SWANLAB_HOST:-$SWANLAB_DEFAULT_HOST}"
    local key="${SWANLAB_API_KEY:-}"

    _sl_info "为容器 $container 配置 SwanLab..."

    docker exec "$container" bash -c "
        # 清除代理
        unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
        pip install swanlab -q 2>/dev/null || true
        if [ -n '${key}' ]; then
            echo '${key}' | swanlab login --host '${host}'
        fi
    "
    _sl_ok "容器 $container SwanLab 配置完成"
}

# ==========================================
# 状态检查
# ==========================================

# 检查 SwanLab 连接状态
swanlab_check() {
    echo "=========================================="
    echo "SwanLab 状态检查"
    echo "=========================================="

    # 加载配置
    swanlab_load_config

    # 配置状态
    if [ -n "$SWANLAB_HOST" ]; then
        _sl_ok "Host: $SWANLAB_HOST"
    else
        _sl_fail "Host: 未配置"
    fi

    if [ -n "$SWANLAB_API_KEY" ]; then
        _sl_ok "API Key: 已配置 (${#SWANLAB_API_KEY} chars)"
    else
        _sl_fail "API Key: 未配置"
    fi

    # 配置文件
    if [ -f "$SWANLAB_CONFIG_FILE" ]; then
        _sl_ok "配置文件: $SWANLAB_CONFIG_FILE"
    else
        _sl_warn "配置文件不存在 (首次交互后会自动创建)"
    fi

    # 安装状态
    if pip show swanlab &>/dev/null; then
        local ver
        ver=$(pip show swanlab 2>/dev/null | grep Version | cut -d' ' -f2)
        _sl_ok "已安装: swanlab $ver"
    else
        _sl_warn "未安装: pip install swanlab"
    fi

    # 网络连通性
    if [ -n "$SWANLAB_HOST" ]; then
        if curl -sf -o /dev/null --connect-timeout 5 "$SWANLAB_HOST"; then
            _sl_ok "网络: 可连接"
        else
            _sl_fail "网络: 无法连接 $SWANLAB_HOST"
        fi
    fi

    echo "=========================================="
}