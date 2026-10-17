#!/usr/bin/env bash
# =============================================================================
# msaicerr 环境自检
# =============================================================================
# 检查 msaicerr 工具的运行前置条件：python 版本、CANN 环境变量、msaicerr.py
# 位置，以及（可选）待分析故障信息目录的完整性与路径关系。
#
# 用法：
#   bash scripts/preflight.sh                    # 仅检查工具与环境
#   bash scripts/preflight.sh <故障信息目录>      # 追加检查故障信息与路径关系
# =============================================================================

set -uo pipefail

REPORT_PATH="${1:-}"
fail=0
warn=0

pass()  { printf '  [PASS] %s\n' "$1"; }
fail_() { printf '  [FAIL] %s\n' "$1"; fail=$((fail + 1)); }
warn_() { printf '  [WARN] %s\n' "$1"; warn=$((warn + 1)); }

echo "=== msaicerr 环境自检 ==="
echo ""

# --- 1. python 版本（要求 3.7.5+） -----------------------------------------
echo "[1/5] python 版本"
if command -v python3 >/dev/null 2>&1; then
    py_ver=$(python3 -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null)
    if python3 -c 'import sys; sys.exit(0 if sys.version_info[:3] >= (3, 7, 5) else 1)' 2>/dev/null; then
        pass "python3 ${py_ver}（要求 >= 3.7.5）"
    else
        fail_ "python3 ${py_ver} 低于要求的 3.7.5"
    fi
else
    fail_ "未找到 python3，msaicerr 依赖 python3.7.5 及以上版本"
fi

# --- 2. CANN 环境变量 -------------------------------------------------------
echo "[2/5] CANN 环境变量"
install_path=""
for var in ASCEND_TOOLKIT_HOME ASCEND_HOME_PATH ASCEND_OPP_PATH; do
    val="${!var:-}"
    if [ -n "$val" ]; then
        pass "$var=$val"
        [ -z "$install_path" ] && [ -d "$val" ] && install_path="$val"
    fi
done
if [ -z "$install_path" ]; then
    warn_ "未检测到 CANN 环境变量，请先执行 source \${INSTALL_DIR}/set_env.sh"
fi

# --- 3. msaicerr.py 位置 ----------------------------------------------------
echo "[3/5] msaicerr.py"
msaicerr_dir=""
for base in "$install_path" /usr/local/Ascend/cann /usr/local/Ascend/ascend-toolkit/latest; do
    [ -z "$base" ] && continue
    if [ -f "$base/tools/msaicerr/msaicerr.py" ]; then
        msaicerr_dir="$base/tools/msaicerr"
        break
    fi
done
if [ -n "$msaicerr_dir" ]; then
    pass "找到 $msaicerr_dir/msaicerr.py"
    echo "         使用前请执行：cd $msaicerr_dir"
else
    warn_ "未在常见路径下找到 msaicerr.py，请确认已安装 Toolkit 包，并手动定位 \${install_path}/tools/msaicerr"
fi

# --- 4. 故障信息目录完整性 --------------------------------------------------
echo "[4/5] 故障信息目录"
if [ -z "$REPORT_PATH" ]; then
    echo "         未传入故障信息目录，跳过（用法：bash scripts/preflight.sh <目录>）"
elif [ ! -d "$REPORT_PATH" ]; then
    fail_ "目录不存在：$REPORT_PATH"
else
    pass "目录存在：$REPORT_PATH"

    # dump 文件
    if find "$REPORT_PATH" -type d -name data-dump 2>/dev/null | grep -q .; then
        dump_cnt=$(find "$REPORT_PATH" -path '*data-dump*' -type f 2>/dev/null | wc -l | tr -d ' ')
        if [ "$dump_cnt" -gt 0 ]; then
            pass "data-dump 下存在 $dump_cnt 个文件"
        else
            fail_ "data-dump 目录为空，无法提取 AI Core Error 信息"
        fi
    else
        fail_ "未找到 data-dump 目录（预期 dfx/data-dump），无法提取 AI Core Error 信息"
    fi

    # 算子编译产物
    o_cnt=$(find "$REPORT_PATH" -type f -name '*.o' 2>/dev/null | wc -l | tr -d ' ')
    json_cnt=$(find "$REPORT_PATH" -type f -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
    if [ "$o_cnt" -gt 0 ] && [ "$json_cnt" -gt 0 ]; then
        pass "存在算子编译信息（*.o: ${o_cnt}，*.json: ${json_cnt}）"
    else
        fail_ "缺少异常算子编译信息（*.o: ${o_cnt}，*.json: ${json_cnt}），需重新采集"
    fi

    # cann 日志
    if [ -d "$REPORT_PATH/dfx/log/host/cann" ]; then
        log_cnt=$(find "$REPORT_PATH/dfx/log/host/cann" -type f 2>/dev/null | wc -l | tr -d ' ')
        if [ "$log_cnt" -gt 0 ]; then
            pass "dfx/log/host/cann 下存在 $log_cnt 个日志文件"
        else
            fail_ "dfx/log/host/cann 为空，无法提取 AI Core Error 信息"
        fi
    else
        fail_ "未找到 dfx/log/host/cann 目录，无法提取 AI Core Error 信息"
    fi
fi

# --- 5. 路径关系（执行目录/输出目录 不能在 -p 目录内） ----------------------
echo "[5/5] 路径关系"
if [ -z "$REPORT_PATH" ] || [ ! -d "$REPORT_PATH" ]; then
    echo "         无故障信息目录，跳过"
else
    report_abs=$(cd "$REPORT_PATH" 2>/dev/null && pwd -P)
    cwd_abs=$(pwd -P)
    case "$cwd_abs/" in
        "$report_abs"/*)
            fail_ "当前目录位于 -p 指定目录内（${cwd_abs}），会导致解析卡住或失败；请切换到 \${install_path}/tools/msaicerr 执行" ;;
        *)
            pass "当前目录不在 -p 指定目录内" ;;
    esac
    echo "         提醒：-out 指定目录同样不能是 -p 目录或其子目录"
fi

echo ""
echo "=== 结果：$fail 项失败，$warn 项告警 ==="
if [ "$fail" -gt 0 ]; then
    echo "存在失败项，请先修复后再执行 msaicerr。"
    exit 1
fi
echo "前置条件检查通过。"
exit 0
