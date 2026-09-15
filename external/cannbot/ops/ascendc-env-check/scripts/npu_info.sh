#!/bin/bash
# NPU 设备信息查询脚本
# 用途：查询 NPU 设备列表、状态、资源使用情况
#
# 设计原则：不再直接解析 npu-smi info 表格输出。
# 所有 npu-smi 调用通过 _npu_info.py 统一封装，该模块：
#   - 优先使用 npu-smi info -t <type> -i <id> 的 key:value 格式
#   - 使用 npu-smi info -m 获取设备映射
#   - 永不直接解析 npu-smi info 主表格
#   - 格式不匹配时输出明确告警
#
# npu-smi 不可用时回退到 asys

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WARNINGS=0
USE_ASYS=false
ASYS_CMD=""

_find_asys() {
    if [ -n "$ASCEND_HOME_PATH" ] && [ -x "$ASCEND_HOME_PATH/tools/ascend_system_advisor/asys/asys" ]; then
        echo "$ASCEND_HOME_PATH/tools/ascend_system_advisor/asys/asys"
    elif command -v asys &> /dev/null; then
        which asys
    fi
}

# 从 asys 表格行 "| key | value |" 提取第二列（纯 bash 参数展开，不使用 awk）
_asys_table_value() {
    # $1 = 表格原始输出, $2 = key（行首匹配）
    local line val
    line=$(echo "$1" | grep "| $2" | head -1)
    [ -z "$line" ] && return 0
    val="${line#*|}"      # 去掉第一列（key）
    val="${val#*\|}"      # 去掉第二列前的分隔符
    val="${val%%|*}"      # 去掉第三列及之后
    echo "${val}" | tr -s ' ' | sed 's/^ //;s/ $//'
}

_detect_npu_via_npu_smi() {
    command -v npu-smi &> /dev/null || return 1
    local output
    output=$(npu-smi info -m 2>&1) || return 1
    # Validate mapping table format
    echo "$output" | grep -qE '^\s+NPU ID' || return 1
    echo "$output" | tail -n +2 | grep -qE '^\s+[0-9]+' || return 1
}

_detect_npu_via_asys() {
    ASYS_CMD=$(_find_asys)
    [ -z "$ASYS_CMD" ] && return 1
    local output
    output=$("$ASYS_CMD" health 2>&1) || return 1
    echo "$output" | grep -q "Device ID:"
}

echo "================================"
echo "NPU 设备信息查询"
echo "================================"
echo ""

# [1/4] 工具检测
echo -e "${BLUE}[1/4] 检查设备检测工具...${NC}"
if _detect_npu_via_npu_smi; then
    npu_smi_version=$(npu-smi -v 2>/dev/null | head -1 || echo "未知")
    echo -e "${GREEN}✓ npu-smi 可用 (版本: $npu_smi_version)${NC}"
elif _detect_npu_via_asys; then
    USE_ASYS=true
    echo -e "${YELLOW}⚠ npu-smi 不可用，回退到 asys${NC}"
    echo -e "${GREEN}✓ asys 可用 ($ASYS_CMD)${NC}"
else
    echo -e "${RED}✗ npu-smi 和 asys 均不可用${NC}"
    echo "  可能原因："
    echo "    1. 未安装 CANN"
    echo "    2. 未 source CANN 环境变量"
    echo "    3. 当前环境不支持 NPU"
    exit 1
fi

echo ""

# [2/4] 设备列表和型号（通过 _npu_info.py 结构化查询）
if [ "$USE_ASYS" = false ]; then
    echo -e "${BLUE}[2/4] 设备列表（结构化查询）...${NC}"

    if [ -f "$SCRIPT_DIR/_npu_info.py" ]; then
        JSON_OUTPUT=$(python3 "$SCRIPT_DIR/_npu_info.py" --json 2>&1) || {
            echo -e "${RED}✗ _npu_info.py 执行失败${NC}"
            echo "$JSON_OUTPUT"
            exit 1
        }

        NPU_COUNT=$(echo "$JSON_OUTPUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['npu_count'])")
        if [ "$NPU_COUNT" -eq 0 ]; then
            echo -e "${RED}✗ 未检测到 NPU 设备${NC}"
            exit 1
        fi

        echo -e "${GREEN}✓ 检测到 $NPU_COUNT 个 NPU 设备${NC}"
        echo "$JSON_OUTPUT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for dev in data['devices']:
    chip = dev.get('chip_name') or 'unknown'
    health = dev.get('health') or 'unknown'
    print(f\"  NPU {dev['npu_id']}: {chip} | {health}\")
"

        # 输出解析警告
        WARNINGS_JSON=$(echo "$JSON_OUTPUT" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('warnings',[])))")
        if [ "$WARNINGS_JSON" -gt 0 ]; then
            WARNINGS=$((WARNINGS + WARNINGS_JSON))
            echo ""
            echo -e "${YELLOW}解析警告:${NC}"
            echo "$JSON_OUTPUT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for w in data.get('warnings', []):
    print(f\"  ! {w}\")
"
        fi
    else
        echo -e "${YELLOW}⚠ _npu_info.py 不存在，回退到直接调用${NC}"
        # 极简 fallback：只用 npu-smi info -m 列出设备（不做列解析）
        npu-smi info -m 2>/dev/null || echo "  (无法获取设备列表)"
    fi
else
    echo -e "${BLUE}[2/4] 设备列表（asys）...${NC}"
    # npu-smi 不可用回退 asys：设备清单 + 健康
    device_count=$("$ASYS_CMD" health 2>/dev/null | grep -c "Device ID:" || true)
    echo -e "${GREEN}✓ 检测到 $device_count 个设备${NC}"
    "$ASYS_CMD" health 2>/dev/null | grep -E "Device ID" | sed 's/^ *|/  /;s/ *|$//'
    # 芯片型号：asys info -r=hardware 的 Chip Info（npu-smi Chip Name 作为 short-soc-version 不可信，issue #587）
    # set -e 保护：asys 子命令失败时取空值降级，不让脚本静默退出
    hw_output=$("$ASYS_CMD" info -r=hardware 2>/dev/null || true)
    chip_info=$(_asys_table_value "$hw_output" "Chip Info")
    if [ -n "$chip_info" ]; then
        echo -e "${GREEN}  芯片型号: ${chip_info}${NC}"
    else
        echo -e "${YELLOW}  ⚠ 无法获取芯片型号（asys info -r=hardware 失败）${NC}"
    fi
fi

echo ""

# [3/4] 设备资源使用
echo -e "${BLUE}[3/4] 设备资源使用...${NC}"
if [ "$USE_ASYS" = false ] && [ -f "$SCRIPT_DIR/_npu_info.py" ]; then
    echo "$JSON_OUTPUT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for dev in data['devices']:
    temp = dev.get('temperature') or '?'
    power = dev.get('power') or '?'
    mem = dev.get('memory', {})
    usage = dev.get('usage', {})
    hbm_cap = mem.get('hbm_capacity_mb') or '?'
    hbm_use = mem.get('hbm_usage_rate') or '?'
    aicore = usage.get('aicore') or '?'
    print(f\"  NPU {dev['npu_id']}: 温度={temp}°C 功耗={power}W HBM={hbm_use}%({hbm_cap}MB) AICore={aicore}%\")
"
else
    # asys 模式：逐卡 asys info -r=status -d=N 取监控/HBM（纯 bash 解析，不使用 awk）
    # set -e 保护：health/status 失败时取空值降级
    dev_ids=$("$ASYS_CMD" health 2>/dev/null | grep -oE "Device ID: *[0-9]+" | grep -oE "[0-9]+" || true)
    if [ -n "$dev_ids" ]; then
        for dev_id in $dev_ids; do
            status_output=$("$ASYS_CMD" info -r=status -d="$dev_id" 2>/dev/null || true)
            temp=$(_asys_table_value "$status_output" "Temperature (C)")
            power=$(_asys_table_value "$status_output" "Power (W)")
            hbm_total=$(_asys_table_value "$status_output" "HBM Total")
            hbm_used=$(_asys_table_value "$status_output" "HBM Used")
            aicore=$(_asys_table_value "$status_output" "AI Core Usage")
            echo "  NPU $dev_id: 温度=${temp:-?}°C 功耗=${power:-?}W HBM=${hbm_used:-?}/${hbm_total:-?}MB AICore=${aicore:-?}%"
        done
    else
        echo -e "${YELLOW}⚠ 结构化查询不可用${NC}"
    fi
fi

echo ""

# [4/4] 进程信息
echo -e "${BLUE}[4/4] 进程信息...${NC}"
if [ "$USE_ASYS" = false ] && [ -f "$SCRIPT_DIR/_npu_info.py" ]; then
    # 优先使用结构化数据判断是否有活跃进程
    echo "$JSON_OUTPUT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
has_process = False
for dev in data['devices']:
    usage = dev.get('usage', {})
    # 判断是否有活跃计算：AICore、NPU Utilization、HBM 使用率任一 > 阈值
    aicore = int(usage.get('aicore') or 0)
    npu_util = int(usage.get('npu_util') or 0)
    hbm = int(usage.get('hbm_usage') or 0)
    if aicore > 1 or npu_util > 1 or hbm > 10:
        has_process = True
        print(f\"  NPU {dev['npu_id']}: 有活跃进程 (AICore={aicore}%, NPU={npu_util}%, HBM={hbm}%)\")
    else:
        print(f\"  NPU {dev['npu_id']}: 无活跃进程\")
if not has_process:
    print('  (所有 NPU 空闲)')
"
elif [ "$USE_ASYS" = false ]; then
    # Fallback：简单检测 npu-smi info 输出中的关键词（非列解析）
    npu-smi info 2>/dev/null | grep -q "No running" && \
        echo "  无运行中的进程" || \
        echo "  有进程运行（详见 npu-smi info）"
else
    echo -e "${YELLOW}⚠ asys 模式不展示进程详情${NC}"
fi

echo ""
echo "================================"
if [ $WARNINGS -gt 0 ]; then
    echo -e "${YELLOW}查询完成（$WARNINGS 个警告）${NC}"
else
    echo "查询完成"
fi
echo "================================"
