#!/bin/bash
# NPU 设备信息查询脚本
# 用途：查询 NPU 设备列表、状态、资源使用情况
# 优先使用 npu-smi，不可用时回退到 asys

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

_detect_npu_via_npu_smi() {
    command -v npu-smi &> /dev/null || return 1
    local output
    output=$(npu-smi info 2>&1) || return 1
    echo "$output" | grep -qE '(OK|Warning|Alarm|Critical|UNKNOWN)' && echo "$output" | grep -qE '^\|\s*[0-9]+\s+'
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

# [1/4] 工具是否能执行
echo -e "${BLUE}[1/4] 检查设备检测工具...${NC}"
if _detect_npu_via_npu_smi; then
    npu_smi_version=$(npu-smi -v 2>/dev/null | head -1 || echo "未知")
    echo -e "${GREEN}✓ npu-smi 可用 (版本: $npu_smi_version)${NC}"
elif _detect_npu_via_asys; then
    USE_ASYS=true
    echo -e "${YELLOW}⚠ npu-smi 不可用或未检测到设备，回退到 asys${NC}"
    echo -e "${GREEN}✓ asys 可用 ($ASYS_CMD)${NC}"
else
    echo -e "${RED}✗ npu-smi 和 asys 均不可用或未检测到设备${NC}"
    echo "  可能原因："
    echo "    1. 未安装 CANN"
    echo "    2. 未 source CANN 环境变量"
    echo "    3. 当前环境不支持 NPU（模拟环境）"
    exit 1
fi

echo ""

# [2/4] 芯片型号
echo -e "${BLUE}[2/4] 芯片型号...${NC}"
if [ "$USE_ASYS" = false ]; then
    device_line=$(npu-smi info 2>/dev/null | grep -E '^\|\s*[0-9]+\s+.*\b(OK|Warning|Alarm|Critical|UNKNOWN)\b' | head -1)
    chip_name=$(echo "$device_line" | awk -F'|' '{gsub(/^ +| +$/, "", $2); print $2}' | awk '{print $2}')
    echo -e "${GREEN}✓ Ascend $chip_name${NC}"
else
    chip_name=$("$ASYS_CMD" info -r status 2>/dev/null | grep "Chip Name" | awk -F'|' '{gsub(/^ +| +$/, "", $3); print $3}' || echo "未知")
    echo -e "${GREEN}✓ $chip_name${NC}"
fi

echo ""

# [3/4] 有多少个 device
echo -e "${BLUE}[3/4] 设备数量...${NC}"
if [ "$USE_ASYS" = false ]; then
    device_count=$(npu-smi info 2>/dev/null | grep -cE '^\|\s*[0-9]+\s+.*\b(OK|Warning|Alarm|Critical|UNKNOWN)\b' || echo "0")
    echo -e "${GREEN}✓ 检测到 $device_count 个设备${NC}"
else
    device_count=$("$ASYS_CMD" health 2>/dev/null | grep -c "Device ID:" || echo "0")
    echo -e "${GREEN}✓ 检测到 $device_count 个设备${NC}"
fi

echo ""

# [4/4] 可用的 device 列表
echo -e "${BLUE}[4/4] 可用设备列表...${NC}"
if [ "$USE_ASYS" = false ]; then
    device_lines=$(npu-smi info 2>/dev/null | grep -E '^\|\s*[0-9]+\s+.*\b(OK|Warning|Alarm|Critical|UNKNOWN)\b' || true)
    if [ -z "$device_lines" ]; then
        echo -e "${RED}✗ 未检测到可用设备${NC}"
        WARNINGS=$((WARNINGS + 1))
    else
        echo "$device_lines" | while IFS= read -r line; do
            devid=$(echo "$line" | awk -F'|' '{gsub(/^ +| +$/, "", $2); print $2}' | awk '{print $1}')
            health=$(echo "$line" | awk -F'|' '{gsub(/^ +| +$/, "", $3); print $3}')
            case "$health" in
                OK|Warning)
                    echo -e "${GREEN}✓ Device $devid: $health (可用)${NC}"
                    ;;
                *)
                    echo -e "${RED}✗ Device $devid: $health (不可用)${NC}"
                    ;;
            esac
        done
    fi
else
    health_output=$("$ASYS_CMD" health 2>&1)
    device_lines=$(echo "$health_output" | grep "Device ID:")
    if [ -z "$device_lines" ]; then
        echo -e "${RED}✗ 未检测到可用设备${NC}"
        WARNINGS=$((WARNINGS + 1))
    else
        echo "$device_lines" | while IFS= read -r line; do
            devid=$(echo "$line" | awk -F'|' '{gsub(/^ +| +$/, "", $2); print $2}' | sed 's/Device ID:[[:space:]]*//')
            status=$(echo "$line" | awk -F'|' '{gsub(/^ +| +$/, "", $3); print $3}')
            case "$status" in
                Healthy|Warning)
                    echo -e "${GREEN}✓ Device $devid: $status (可用)${NC}"
                    ;;
                *)
                    echo -e "${RED}✗ Device $devid: $status (不可用)${NC}"
                    ;;
            esac
        done
    fi
fi

echo ""
echo "================================"
echo "查询完成"
echo "================================"
