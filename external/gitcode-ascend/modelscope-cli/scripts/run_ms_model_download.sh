#!/bin/bash

# ModelScope 批量模型下载脚本
# 支持批量下载多个模型，并可选过滤文件类型
#
# 推荐：对于 Ascend NPU 部署，建议优先从 Eco-Tech 组织下载
# 经过量化和优化的模型，这些模型已针对 Ascend 平台进行了适配。
#
# 使用方法:
#   1. 编辑 MODELS 数组，添加需要下载的模型 ID
#   2. 执行脚本: bash run_ms_model_download.sh
#
# 前置检查:
#   脚本会自动运行 run_preflight_check.sh 检测环境配置
#   如需跳过检查，设置环境变量: SKIP_PREFLIGHT=1

set -e

# 配置：需要下载的模型列表
# 推荐：优先使用 Eco-Tech 组织的量化模型（针对 Ascend NPU 优化）
# 提示：请访问 https://modelscope.cn/models 搜索并复制最新模型 ID
MODELS=(
  # Qwen3.5-2B-Base (创建时间: 2026-02-28)
  Qwen/Qwen3.5-2B-Base
)

# 配置：文件过滤选项
ALLOW_PATTERNS=""  # 需要包含的文件模式（glob 格式，留空表示包含所有）
EXCLUDE="*.onnx *.onnx_data"  # 需要排除的文件模式

# 配置：下载目标目录
DIR="./models"

# ==================== 前置检查 ====================

if [ -z "$SKIP_PREFLIGHT" ]; then
    SCRIPT_DIR=$(dirname "$0")
    PREFLIGHT_SCRIPT="${SCRIPT_DIR}/run_preflight_check.sh"
    
    if [ -f "$PREFLIGHT_SCRIPT" ]; then
        echo "========================================"
        echo "执行前置检查..."
        echo "========================================"
        echo ""
        
        if ! bash "$PREFLIGHT_SCRIPT"; then
            echo ""
            echo "❌ 前置检查失败，请修复后重试"
            echo "   或设置 SKIP_PREFLIGHT=1 跳过检查"
            exit 1
        fi
        echo ""
    else
        echo "⚠️  未找到前置检查脚本: $PREFLIGHT_SCRIPT"
        echo "   建议手动执行: bash scripts/run_preflight_check.sh"
        echo ""
    fi
else
    echo "⚠️  已跳过前置检查 (SKIP_PREFLIGHT=1)"
    echo ""
fi

# ==================== 下载前确认 ====================

echo "========================================"
echo "ModelScope 模型批量下载 - 下载前确认"
echo "========================================"
echo ""
echo "⚠️  请确认以下下载配置信息："
echo ""
echo "【下载目标目录】"
echo "  ${DIR}"
echo ""

# 显示磁盘空间
if command -v df &> /dev/null; then
    DISK_INFO=$(df -h "$DIR" 2>/dev/null | tail -1)
    echo "【磁盘空间】"
    echo "  $DISK_INFO"
    echo ""
fi

echo "【文件过滤配置】"
echo "  排除: ${EXCLUDE}"
echo ""
echo "【待下载模型列表】"

TOTAL_MODELS=${#MODELS[@]}
if [ $TOTAL_MODELS -eq 0 ]; then
    echo ""
    echo "  ⚠️  模型列表为空！请编辑脚本添加模型 ID"
    echo "     编辑命令: vim $0"
    echo ""
    exit 1
fi

for i in "${!MODELS[@]}"; do
    MODEL_ID="${MODELS[$i]}"
    echo "  [$((i+1))] ${MODEL_ID}"
done

echo ""
echo "========================================"
echo ""
echo "⚠️  风险提示："
echo ""
echo "  - 大模型下载可能耗时数小时，请确保网络稳定"
echo "  - 建议使用循环重试: bash scripts/ms_loop.sh $0"
echo "  - 中断后可继续下载，会自动跳过已下载文件"
echo ""

# 请求用户确认
echo ""
echo "✅ 自动确认，开始下载..."
echo ""

# ==================== 开始下载 ====================

SUCCESS_COUNT=0
FAILED_COUNT=0
FAILED_MODELS=()

START_TIME=$(date +%s)

# 遍历模型列表进行下载
for MODEL_ID in "${MODELS[@]}"; do
    LOCAL_DIR=${DIR}/${MODEL_ID}

    # 创建目标目录（如果不存在）
    if [ ! -d "${LOCAL_DIR}" ]; then
        mkdir -p ${LOCAL_DIR}
    fi

    # 构建下载命令
    cmd="modelscope download --model ${MODEL_ID} --local_dir ${LOCAL_DIR} --exclude '${EXCLUDE}'"

    # 显示执行的命令
    echo -e "\n========================================"
    echo "正在下载: ${MODEL_ID}"
    echo "目标目录: ${LOCAL_DIR}"
    echo "========================================"
    echo -e "\t>${cmd}"
    echo ""

    # 执行下载命令
    set +e
    eval ${cmd}
    EXIT_CODE=$?
    set -e

    # 检查下载结果
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ ${MODEL_ID} 下载完成"
        
        # 显示下载后的目录大小
        if command -v du &> /dev/null; then
            DOWNLOADED_SIZE=$(du -sh ${LOCAL_DIR} 2>/dev/null | awk '{print $1}')
            echo "   下载大小: ${DOWNLOADED_SIZE}"
        fi
        
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "❌ ${MODEL_ID} 下载失败，退出码: $EXIT_CODE"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_MODELS+=("${MODEL_ID}")
    fi
done

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
ELAPSED_MINUTES=$((ELAPSED_TIME / 60))
ELAPSED_SECONDS=$((ELAPSED_TIME % 60))

# ==================== 下载后确认 ====================

echo ""
echo "========================================"
echo "ModelScope 模型批量下载 - 下载完成"
echo "========================================"
echo ""
echo "【下载结果摘要】"
echo "  总计模型数量: ${TOTAL_MODELS}"
echo "  下载成功: ${SUCCESS_COUNT}"
echo "  下载失败: ${FAILED_COUNT}"
echo "  耗时: ${ELAPSED_MINUTES}分${ELAPSED_SECONDS}秒"
echo ""

if [ ${FAILED_COUNT} -gt 0 ]; then
    echo "【失败模型列表】"
    for failed_model in "${FAILED_MODELS[@]}"; do
        echo "  ❌ ${failed_model}"
    done
    echo ""
    echo "💡 建议："
    echo "   - 使用循环重试继续下载: bash scripts/ms_loop.sh $0"
    echo "   - 检查网络配置: bash scripts/run_network_diagnose.sh"
    echo ""
fi

# 显示磁盘使用情况
echo "【磁盘使用情况】"
du -sh ${DIR}/* 2>/dev/null | tail -10 || echo "  (无法获取磁盘使用信息)"
echo ""

echo "【总下载大小】"
TOTAL_SIZE=$(du -sh ${DIR} 2>/dev/null | awk '{print $1}')
echo "  ${TOTAL_SIZE}"
echo ""

echo "========================================"
echo ""

echo "✅ 下载完成，脚本退出"
