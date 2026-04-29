#!/bin/bash

# ModelScope 批量数据集下载脚本
# 支持批量下载多个数据集，并可选过滤文件类型

set -e

# 详见: https://modelscope.cn/docs/models/download
# 配置：需要下载的数据集列表
DATASETS=(
  WorldVQA/WorldVQA
)

# 配置：文件过滤选项
ALLOW_PATTERNS=""  # 需要包含的文件模式（glob 格式，留空表示包含所有）
EXCLUDE="*.onnx *.onnx_data"  # 需要排除的文件模式

# 配置：下载目标目录
DIR="./datasets"

# ==================== 下载前确认 ====================

echo "========================================"
echo "ModelScope 数据集批量下载 - 下载前确认"
echo "========================================"
echo ""
echo "⚠️  请确认以下下载配置信息："
echo ""
echo "【下载目标目录】"
echo "  ${DIR}"
echo ""
echo "【文件过滤配置】"
echo "  排除: ${EXCLUDE}"
echo ""
echo "【待下载数据集列表】"
for i in "${!DATASETS[@]}"; do
    echo "  [$((i+1))] ${DATASETS[$i]}"
done
echo ""
echo "========================================"
echo ""

# 请求用户确认
echo ""
echo "✅ 自动确认，开始下载..."
echo ""

# ==================== 开始下载 ====================

SUCCESS_COUNT=0
FAILED_COUNT=0
FAILED_DATASETS=()

# 遍历数据集列表进行下载
for DATASET in "${DATASETS[@]}"; do
    LOCAL_DIR=${DIR}/${DATASET}

    # 创建目标目录（如果不存在）
    if [ ! -d "${LOCAL_DIR}" ]; then
        mkdir -p ${LOCAL_DIR}
    fi

    # 构建下载命令
    cmd="modelscope download --dataset ${DATASET} --local_dir ${LOCAL_DIR} --exclude '${EXCLUDE}'"

    # 显示执行的命令
    echo -e "\n========================================"
    echo "正在下载: ${DATASET}"
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
        echo "✅ ${DATASET} 下载完成"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "❌ ${DATASET} 下载失败，退出码: $EXIT_CODE"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_DATASETS+=("${DATASET}")
    fi
done

# ==================== 下载后确认 ====================

echo ""
echo "========================================"
echo "ModelScope 数据集批量下载 - 下载完成"
echo "========================================"
echo ""
echo "【下载结果摘要】"
echo "  总计数据集数量: ${#DATASETS[@]}"
echo "  下载成功: ${SUCCESS_COUNT}"
echo "  下载失败: ${FAILED_COUNT}"
echo ""

if [ ${FAILED_COUNT} -gt 0 ]; then
    echo "【失败数据集列表】"
    for failed_dataset in "${FAILED_DATASETS[@]}"; do
        echo "  ❌ ${failed_dataset}"
    done
    echo ""
fi

# 显示磁盘使用情况
echo "【磁盘使用情况】"
du -sh ${DIR}/* 2>/dev/null | tail -5 || echo "  (无法获取磁盘使用信息)"
echo ""

echo "========================================"
echo ""

echo ""
echo "✅ 自动确认，脚本退出"
