#!/bin/bash

# ModelScope 下载前置检查脚本
# 在执行下载前检测环境和网络配置
#
# 使用方法:
#   bash run_preflight_check.sh
#
# 检查项:
#   - Python 环境（>= 3.7）
#   - ModelScope CLI 安装
#   - 代理配置
#   - 网络连接
#   - SSL 证书
#   - 磁盘空间
#   - ModelScope 缓存

set -e

echo "========================================"
echo "ModelScope 下载前置检查"
echo "========================================"
echo ""

PASS=0
FAIL=0
WARN=0

check_pass() {
    echo "  ✅ $1"
    PASS=$((PASS + 1))
}

check_fail() {
    echo "  ❌ $1"
    echo "     → $2"
    FAIL=$((FAIL + 1))
}

check_warn() {
    echo "  ⚠️  $1"
    echo "     → $2"
    WARN=$((WARN + 1))
}

# 1. 检查 Python（版本 >= 3.7）
echo "【1. Python 环境】"
if command -v python3 &> /dev/null; then
    PY_VER=$(python3 --version 2>&1)
    
    # 获取主版本号和次版本号
    PY_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)" 2>/dev/null || echo "0")
    PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
    
    # 检查版本 >= 3.7
    if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 7 ]; then
        check_pass "Python 已安装: $PY_VER (符合要求 >= 3.7)"
    elif [ "$PY_MAJOR" -eq 2 ]; then
        check_fail "Python 版本过低: $PY_VER" "需要 Python 3.7+，请升级 Python"
    elif [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 7 ]; then
        check_fail "Python 版本过低: $PY_VER (需要 >= 3.7)" "升级 Python: yum install python3.9 或 apt install python3.9"
    else
        check_warn "Python 版本未知: $PY_VER" "请确认版本 >= 3.7"
    fi
else
    check_fail "Python 未安装" "安装 Python 3.7+: yum install python3 或 apt install python3"
fi

# 2. 检查 ModelScope
echo ""
echo "【2. ModelScope CLI】"
if python3 -c "import modelscope" 2>/dev/null; then
    MS_VER=$(python3 -c "
try:
    import modelscope
    print(modelscope.__version__)
except AttributeError:
    print('unknown')
except Exception:
    print('error')
" 2>/dev/null)
    
    if [ "$MS_VER" = "unknown" ] || [ "$MS_VER" = "error" ] || [ -z "$MS_VER" ]; then
        check_pass "ModelScope 已安装（版本未知）"
    else
        check_pass "ModelScope 已安装: v${MS_VER}"
    fi
else
    check_fail "ModelScope 未安装" "pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple"
fi

# 3. 检查代理配置
echo ""
echo "【3. 代理配置】"
if [ -n "$HTTPS_PROXY" ]; then
    PROXY_VAL="$HTTPS_PROXY"
elif [ -n "$https_proxy" ]; then
    PROXY_VAL="$https_proxy"
elif [ -n "$HTTP_PROXY" ]; then
    PROXY_VAL="$HTTP_PROXY"
elif [ -n "$http_proxy" ]; then
    PROXY_VAL="$http_proxy"
else
    PROXY_VAL=""
fi

if [ -n "$PROXY_VAL" ]; then
    # 截断显示，避免太长
    PROXY_DISPLAY="${PROXY_VAL:0:50}"
    if [ ${#PROXY_VAL} -gt 50 ]; then
        PROXY_DISPLAY="${PROXY_DISPLAY}..."
    fi
    check_pass "代理已配置: $PROXY_DISPLAY"
else
    check_warn "代理未配置" "如需代理: export HTTPS_PROXY=http://proxy-host:port/ 或运行 bash scripts/setup_proxy.sh"
fi

# 4. 检查网络连接
echo ""
echo "【4. 网络连接】"
echo -n "  测试连接 ModelScope... "

# 先获取 HTTP 状态码，再判断
HTTP_CODE=$(timeout 15 curl -k -s -o /dev/null -w "%{http_code}" 'https://www.modelscope.cn' 2>/dev/null || echo "000")

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "301" ] || [ "$HTTP_CODE" = "302" ]; then
    echo ""
    check_pass "ModelScope 可访问 (HTTP $HTTP_CODE)"
elif [ "$HTTP_CODE" = "000" ]; then
    echo ""
    check_fail "无法连接 ModelScope (连接超时)" "检查代理配置或网络，运行 run_network_diagnose.sh 详细诊断"
else
    echo ""
    check_fail "ModelScope 返回异常 (HTTP $HTTP_CODE)" "检查网络或代理配置"
fi

# 5. 检查 SSL 证书
echo ""
echo "【5. SSL 证书】"

# 使用 heredoc 避免引号转义问题
SSL_CHECK=$(python3 << 'PYEOF' 2>/dev/null || echo "FAIL"
import ssl
import socket

try:
    ctx = ssl.create_default_context()
    with socket.create_connection(('www.modelscope.cn', 443), timeout=10) as s:
        with ctx.wrap_socket(s, server_hostname='www.modelscope.cn'):
            pass
    print('OK')
except ssl.SSLCertVerificationError:
    print('CERT_FAIL')
except socket.timeout:
    print('TIMEOUT')
except Exception as e:
    print('FAIL:' + str(e)[:40])
PYEOF
)

if [ "$SSL_CHECK" = "OK" ]; then
    check_pass "SSL 证书验证通过"
elif [ "$SSL_CHECK" = "CERT_FAIL" ]; then
    check_warn "SSL 证书验证失败（可能为自签名证书）" "安装 CA 证书 或 禁用 SSL 验证（见 SKILL.md 常见问题）"
elif [ "$SSL_CHECK" = "TIMEOUT" ]; then
    check_warn "SSL 连接超时" "检查网络连接"
else
    check_warn "SSL 检测异常" "可能需要安装 SSL 证书或配置代理"
fi

# 6. 检查磁盘空间
echo ""
echo "【6. 磁盘空间】"
if command -v df &> /dev/null; then
    # 获取当前目录所在分区的可用空间（KB）
    AVAILABLE_KB=$(df -k . 2>/dev/null | tail -1 | awk '{print $4}')
    
    # 检查是否为空或非数字
    if [ -z "$AVAILABLE_KB" ] || ! [[ "$AVAILABLE_KB" =~ ^[0-9]+$ ]]; then
        check_warn "无法解析磁盘空间" "手动检查: df -h"
    else
        # 转换为 GB
        AVAILABLE_GB=$((AVAILABLE_KB / 1024 / 1024))
        
        if [ "$AVAILABLE_GB" -ge 100 ]; then
            check_pass "磁盘空间充足: ${AVAILABLE_GB}GB 可用"
        elif [ "$AVAILABLE_GB" -ge 50 ]; then
            check_warn "磁盘空间尚可: ${AVAILABLE_GB}GB 可用" "建议预留 100GB+ 用于大模型下载"
        elif [ "$AVAILABLE_GB" -ge 10 ]; then
            check_warn "磁盘空间有限: ${AVAILABLE_GB}GB 可用" "大模型下载可能失败，建议清理空间"
        else
            check_fail "磁盘空间不足: ${AVAILABLE_GB}GB 可用" "清理空间或使用 --local_dir 指定其他目录"
        fi
    fi
else
    check_warn "无法检测磁盘空间 (df 命令不存在)" "手动检查: df -h"
fi

# 7. 检查 ModelScope 缓存目录
echo ""
echo "【7. ModelScope 缓存】"
CACHE_DIR="${MODELSCOPE_CACHE:-$HOME/.cache/modelscope}"

if [ -d "$CACHE_DIR" ]; then
    if command -v du &> /dev/null; then
        CACHE_SIZE=$(du -sh "$CACHE_DIR" 2>/dev/null | awk '{print $1}' || echo "unknown")
        check_pass "缓存目录: $CACHE_DIR (大小: $CACHE_SIZE)"
    else
        check_pass "缓存目录: $CACHE_DIR"
    fi
else
    check_pass "缓存目录将创建于: $CACHE_DIR"
fi

# ==================== 汇总 ====================
echo ""
echo "========================================"
echo "检查结果汇总"
echo "========================================"
echo "  ✅ 通过: $PASS"
echo "  ⚠️  警告: $WARN"
echo "  ❌ 失败: $FAIL"
echo ""

# 退出码和后续操作提示
if [ $FAIL -gt 0 ]; then
    echo "❌ 存在 $FAIL 项检查失败，请修复后重试"
    echo ""
    echo "💡 建议："
    echo "   1. 运行详细诊断: bash run_network_diagnose.sh"
    echo "   2. 查看文档: cat SKILL.md | grep -A 10 '常见问题'"
    echo "   3. 跳过检查强制执行: SKIP_PREFLIGHT=1 bash run_ms_model_download.sh"
    exit 1
elif [ $WARN -gt 0 ]; then
    echo "⚠️  存在 $WARN 项警告"
    echo ""
    read -p "是否继续下载？(yes/no): " CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        echo "❌ 用户取消"
        exit 1
    fi
    echo "✅ 继续执行..."
else
    echo "✅ 所有检查通过，可以开始下载"
fi

echo ""
echo "💡 下一步："
echo "   bash run_ms_model_download.sh"
echo ""
echo "   或使用循环重试（推荐大模型）："
echo "   bash ms_loop.sh run_ms_model_download.sh"

exit 0
