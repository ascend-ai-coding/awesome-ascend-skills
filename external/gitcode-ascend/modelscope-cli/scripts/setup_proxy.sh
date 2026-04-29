#!/bin/bash

# 代理环境自动检测与配置脚本
# 用于检测代理设置并交互式配置
#
# 使用方法:
#   bash setup_proxy.sh
#
# 功能:
#   1. 检测当前代理配置
#   2. 交互式配置 HTTP/HTTPS 代理
#   3. 写入 ~/.bashrc 持久化
#   4. 可选配置 pip 镜像源

set -e

SCRIPT_DIR=$(dirname "$0")

echo "========================================"
echo "代理环境配置"
echo "========================================"
echo ""

# ==================== Step 1: 检测当前配置 ====================

# 脱敏显示：隐藏代理地址中嵌入的用户名密码
mask_proxy() {
    echo "$1" | sed 's|//[^@]*@|//***@|' | cut -c1-60
}

echo "【Step 1】检测当前代理配置..."
echo ""

DETECTED_PROXY=""

if [ -n "$HTTPS_PROXY" ]; then
    DETECTED_PROXY="$HTTPS_PROXY"
    echo "  ✅ 检测到 HTTPS_PROXY: $(mask_proxy "$HTTPS_PROXY")"
elif [ -n "$https_proxy" ]; then
    DETECTED_PROXY="$https_proxy"
    echo "  ✅ 检测到 https_proxy: $(mask_proxy "$https_proxy")"
elif [ -n "$HTTP_PROXY" ]; then
    DETECTED_PROXY="$HTTP_PROXY"
    echo "  ✅ 检测到 HTTP_PROXY: $(mask_proxy "$HTTP_PROXY")"
elif [ -n "$http_proxy" ]; then
    DETECTED_PROXY="$http_proxy"
    echo "  ✅ 检测到 http_proxy: $(mask_proxy "$http_proxy")"
else
    echo "  ⚠️  未检测到代理配置"
fi

echo ""

# ==================== Step 2: 交互式配置 ====================

echo "【Step 2】配置代理"
echo ""
echo "  格式: http://[user:pass@]proxy-host:port"
echo "  示例: http://proxy.example.com:8080"
echo "  示例: http://user:pass@proxy.example.com:8080"
echo ""

if [ -n "$DETECTED_PROXY" ]; then
    read -p "  当前代理为 $DETECTED_PROXY，是否保留？(yes/no): " KEEP_PROXY
    if [ "$KEEP_PROXY" = "yes" ]; then
        PROXY_VAL="$DETECTED_PROXY"
    else
        read -p "  请输入新的代理地址 (留空跳过): " PROXY_VAL
    fi
else
    read -p "  请输入代理地址 (留空跳过): " PROXY_VAL
fi

# ==================== Step 3: 写入环境变量 ====================

if [ -n "$PROXY_VAL" ]; then
    echo ""
    echo "  配置代理: $PROXY_VAL"

    export HTTP_PROXY="$PROXY_VAL"
    export HTTPS_PROXY="$PROXY_VAL"
    export http_proxy="$PROXY_VAL"
    export https_proxy="$PROXY_VAL"

    # 持久化到 ~/.bashrc
    BASHRC="$HOME/.bashrc"
    MARKER="# >>> modelscope-cli proxy >>>"

    # 移除旧配置
    if grep -q "$MARKER" "$BASHRC" 2>/dev/null; then
        sed -i "/$MARKER/,/# <<< modelscope-cli proxy <<</d" "$BASHRC"
    fi

    # 写入新配置
    cat >> "$BASHRC" << PROXYEOF

$MARKER
export HTTP_PROXY="$PROXY_VAL"
export HTTPS_PROXY="$PROXY_VAL"
export http_proxy="$PROXY_VAL"
export https_proxy="$PROXY_VAL"
# <<< modelscope-cli proxy <<<
PROXYEOF

    echo "  ✅ 已写入 $BASHRC"
else
    echo ""
    echo "  跳过代理配置"
fi

# ==================== Step 4: 可选 pip 镜像源 ====================

echo ""
read -p "是否配置 pip 镜像源？(yes/no): " SETUP_PIP
if [ "$SETUP_PIP" = "yes" ]; then
    echo ""
    echo "  常用镜像源："
    echo "    1. 清华: https://pypi.tuna.tsinghua.edu.cn/simple"
    echo "    2. 阿里: https://mirrors.aliyun.com/pypi/simple"
    echo "    3. 跳过"
    echo ""
    read -p "  请选择 (1-3): " PIP_CHOICE

    PIP_MIRROR=""
    case "$PIP_CHOICE" in
        1) PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple" ;;
        2) PIP_MIRROR="https://mirrors.aliyun.com/pypi/simple" ;;
        *)  echo "  跳过 pip 配置" ;;
    esac

    if [ -n "$PIP_MIRROR" ]; then
        pip config set global.index-url "$PIP_MIRROR" 2>/dev/null || \
            echo "  ⚠️  pip 配置失败，可手动执行: pip config set global.index-url $PIP_MIRROR"
        echo "  ✅ pip 镜像源已配置: $PIP_MIRROR"
    fi
fi

# ==================== Step 5: 验证 ====================

echo ""
echo "========================================"
echo "配置完成"
echo "========================================"
echo ""

if [ -n "$PROXY_VAL" ]; then
    echo "  代理: $PROXY_VAL"
fi

echo ""
echo "💡 下一步："
echo "   1. 重新加载环境变量: source ~/.bashrc"
echo "   2. 运行前置检查验证: bash $SCRIPT_DIR/run_preflight_check.sh"
echo ""

read -p "是否立即重新加载环境变量？(yes/no): " RELOAD_CONFIRM
if [ "$RELOAD_CONFIRM" = "yes" ]; then
    source ~/.bashrc
    echo "  ✅ 环境变量已加载"
fi

exit 0
