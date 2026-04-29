#!/bin/bash

# =================================================================
# 脚本名称: run_report_param.sh
# 功能: 根据模型权重文件大小和精度标识，统计模型参数量
# =================================================================

# 精度标识说明:
# 精度类型              每参数字节数    说明
# FP32                   4.0             32位浮点
# BF16 / FP16           2.0             16位浮点
# W8A8Z / W8A8       1.0             8位权重，8位激活
# FP8 / INT8            1.0             8位浮点/整数
# W4A8 / INT4 / Q4    0.5             4位权重，8位激活

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 校验输入
if [ -z "$1" ]; then
    echo -e "${YELLOW}使用方法: $0 <模型目录路径>${NC}"
    echo ""
    echo "示例:"
    echo "  $0 ./Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp"
    echo "  $0 /mnt/data/models"
    exit 1
fi

BASE_DIR="$1"

# 检查目录是否存在
if [ ! -d "$BASE_DIR" ]; then
    echo -e "${RED}错误: 目录不存在: $BASE_DIR${NC}"
    exit 1
fi

# 根据文件名推测精度 (Bytes per Parameter)
get_precision() {
    local FILE_NAME="$1"
    local UPPER_NAME=$(echo "$FILE_NAME" | tr '[:lower:]' '[:upper:]')

    local BPP=2.0
    local PRECISION="BF16/FP16 (Default)"

    case "$UPPER_NAME" in
        *FP32*)
            BPP=4.0
            PRECISION="FP32"
            ;;
        *BF16*|*FP16*)
            BPP=2.0
            PRECISION="BF16/FP16"
            ;;
        *W8A8Z*|*W8A8*)
            BPP=1.0
            PRECISION="W8A8Z/W8A8"
            ;;
        *FP8*|*INT8*|*Q8*)
            BPP=1.0
            PRECISION="FP8/INT8"
            ;;
        *W4A8*|*Q4*|*Q4_8*|*4BIT*|*INT4*)
            BPP=0.5
            PRECISION="W4A8/Q4"
            ;;
        *)
            BPP=2.0
            PRECISION="BF16/FP16 (Default)"
            ;;
    esac

    echo "$BPP|$PRECISION"
}

# 分析模型目录
analyze_model() {
    local MODEL_DIR="$1"
    local MODEL_NAME=$(basename "$MODEL_DIR")

    echo -e "${CYAN}========================================${NC}"
    echo -e "${GREEN}模型: ${MODEL_NAME}${NC}"
    echo -e "${CYAN}========================================${NC}"

    # 查找权重文件
    local WEIGHTS=$(find "$MODEL_DIR" -type f \( -name "*-of-*.safetensors" -o -name "model*.safetensors" -o -name "pytorch_model*.bin" -o -name "*.gguf" \) 2>/dev/null)

    if [ -z "$WEIGHTS" ]; then
        echo -e "${RED}警告: 未找到权重文件${NC}"
        echo ""
        return 0
    fi

    local TOTAL_SIZE_BYTES=0
    local FILE_COUNT=0
    local BPP=2.0
    local PRECISION="BF16/FP16 (Default)"

    # 遍历文件并统计
    while IFS= read -r file; do
        if [[ "$OSTYPE" == "darwin"* ]]; then
            local file_size=$(stat -f%z "$file")
        else
            local file_size=$(stat -c%s "$file")
        fi

        TOTAL_SIZE_BYTES=$((TOTAL_SIZE_BYTES + file_size))
        FILE_COUNT=$((FILE_COUNT + 1))

        # 从第一个文件获取精度信息
        if [ $FILE_COUNT -eq 1 ]; then
            local precision_info=$(get_precision "$(basename "$file")")
            BPP=$(echo "$precision_info" | cut -d'|' -f1)
            PRECISION=$(echo "$precision_info" | cut -d'|' -f2)
        fi
    done <<< "$WEIGHTS"

    # 转换为 GB
    local SIZE_GB=$(awk "BEGIN {printf \"%.2f\", $TOTAL_SIZE_BYTES / 1024 / 1024 / 1024}")
    local SIZE_TB=$(awk "BEGIN {printf \"%.4f\", $TOTAL_SIZE_BYTES / 1024 / 1024 / 1024 / 1024}")

    # 计算参数量 (B)
    local PARAMS_B=$(awk "BEGIN {printf \"%.2f\", $TOTAL_SIZE_BYTES / 1024 / 1024 / 1024 / $BPP}")

    # 判断参数量范围
    local PARAMS_RANGE=""
    if awk -v b="$PARAMS_B" 'BEGIN { exit !(b < 1) }'; then
        PARAMS_RANGE="<1B"
    elif awk -v b="$PARAMS_B" 'BEGIN { exit !(b < 7) }'; then
        PARAMS_RANGE="1-7B"
    elif awk -v b="$PARAMS_B" 'BEGIN { exit !(b < 13) }'; then
        PARAMS_RANGE="7-13B"
    elif awk -v b="$PARAMS_B" 'BEGIN { exit !(b < 34) }'; then
        PARAMS_RANGE="13-34B"
    elif awk -v b="$PARAMS_B" 'BEGIN { exit !(b < 70) }'; then
        PARAMS_RANGE="34-70B"
    else
        PARAMS_RANGE=">70B"
    fi

    # 输出报告
    echo -e "权重文件数量: ${FILE_COUNT}"
    echo -e "模型总大小: ${YELLOW}${SIZE_GB} GB${NC} (${SIZE_TB} TB)"
    echo -e "数据精度: ${PRECISION} (每参数 ${BPP} 字节)"
    echo -e "推测参数量: ${GREEN}${PARAMS_B} B${NC} (${PARAMS_RANGE})"
    echo ""

    # 输出到全局变量用于汇总
    echo "${MODEL_NAME}|${SIZE_GB}|${PARAMS_B}|${PRECISION}" >> "$TEMP_FILE"
}

# 主程序
echo -e "${CYAN}"
echo "========================================"
echo "ModelScope 模型参数量统计报告"
echo "========================================"
echo -e "${NC}"
echo ""

# 清空临时文件
TEMP_FILE=$(mktemp /tmp/param_report_XXXXXX.txt)
> "$TEMP_FILE"

# 确保异常退出时清理临时文件
trap 'rm -f "$TEMP_FILE"' EXIT INT TERM

# 遍历模型目录
MODEL_COUNT=0
for dir in "$BASE_DIR"/*/; do
    if [ -d "$dir" ]; then
        analyze_model "$dir"
        MODEL_COUNT=$((MODEL_COUNT + 1))
    fi
done

# 如果只有一个模型目录，直接分析
if [ $MODEL_COUNT -eq 1 ]; then
    rm -f $TEMP_FILE
    echo -e "${GREEN}========================================${NC}"
    echo "报告完成"
    echo -e "${GREEN}========================================${NC}"
    exit 0
fi

# 或者是单个文件/目录结构不是子目录
if [ $MODEL_COUNT -eq 0 ]; then
    analyze_model "$BASE_DIR"
    rm -f $TEMP_FILE
    echo -e "${GREEN}========================================${NC}"
    echo "报告完成"
    echo -e "${GREEN}========================================${NC}"
    exit 0
fi

# 生成汇总报告
echo -e "${CYAN}"
echo "========================================"
echo "汇总报告"
echo "========================================"
echo -e "${NC}"
printf "%-50s %12s %12s %20s\n" "模型名称" "大小(GB)" "参数量(B)" "精度"
echo -e "${CYAN}--------------------------------------------------------------------------------${NC}"

TOTAL_SIZE_GB=0
while IFS='|' read -r name size params precision; do
    TOTAL_SIZE_GB=$(awk "BEGIN {printf \"%.2f\", $TOTAL_SIZE_GB + $size}")
    printf "%-50s %12s %12s %20s\n" "$name" "$size" "$params" "$precision"
done < $TEMP_FILE

echo -e "${CYAN}--------------------------------------------------------------------------------${NC}"
printf "%-50s %12s %12s %20s\n" "总计" "$TOTAL_SIZE_GB" "-" "-"

# 清理临时文件
rm -f $TEMP_FILE

echo ""
echo -e "${GREEN}========================================${NC}"
echo "报告完成"
echo -e "${GREEN}========================================${NC}"
