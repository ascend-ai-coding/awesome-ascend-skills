#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================
# add_example 算子调用示例执行脚本

set -e

# ============================================================================
# 配置
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_MODE="eager"  # 默认 aclnn 调用示例
CLEAN_BUILD=false

# ============================================================================
# 帮助信息
# ============================================================================
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "算子调用示例：演示 aclnn / 图模式 (GE IR) 调用方式"
    echo ""
    echo "Options:"
    echo "  --eager          运行 aclnn 调用示例（默认）"
    echo "  --graph          运行图模式 (GE IR) 调用示例"
    echo "  --clean          清理构建目录后退出"
    echo "  -h, --help       显示帮助信息"
    echo ""
    echo "前置条件："
    echo "  - 算子包已编译并安装（bash build.sh --soc=<soc> && ./build/custom_opp_ubuntu_aarch64.run）"
    echo "  - NPU 设备可用"
    echo ""
    echo "Examples:"
    echo "  # 运行 aclnn 调用示例（默认）"
    echo "  $0"
    echo ""
    echo "  # 运行图模式调用示例"
    echo "  $0 --graph"
    echo ""
    echo "  # 清理构建目录"
    echo "  $0 --clean"
}

# ============================================================================
# 解析参数
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --eager)
            EXAMPLE_MODE="eager"
            shift
            ;;
        --graph)
            EXAMPLE_MODE="graph"
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 根据模式设置独立的构建目录
if [ "$EXAMPLE_MODE" = "graph" ]; then
    BUILD_DIR="${SCRIPT_DIR}/build-geir"
else
    BUILD_DIR="${SCRIPT_DIR}/build-eager"
fi

# ============================================================================
# 清理构建目录
# ============================================================================
if [ "$CLEAN_BUILD" = true ]; then
    echo "清理构建目录..."
    rm -rf "${SCRIPT_DIR}/build-eager" "${SCRIPT_DIR}/build-geir"
    echo "清理完成"
    exit 0
fi

# ============================================================================
# 显示配置信息
# ============================================================================
echo "========================================"
echo "add_example 算子调用示例"
echo "========================================"
if [ "$EXAMPLE_MODE" = "graph" ]; then
    echo "模式: 图模式 (GE IR)"
else
    echo "模式: aclnn (eager)"
fi
echo "工作目录: ${SCRIPT_DIR}"
echo "========================================"
echo ""

# ============================================================================
# 检查依赖
# ============================================================================
echo "检查依赖..."

if ! command -v cmake &> /dev/null; then
    echo "错误: 未找到 cmake"
    exit 1
fi

if ! command -v g++ &> /dev/null; then
    echo "错误: 未找到 g++"
    exit 1
fi

echo "依赖检查完成"
echo ""

# ============================================================================
# 设置环境变量
# ============================================================================
echo "设置环境变量..."

if [ -z "$ASCEND_HOME_PATH" ]; then
    echo "警告: 未设置 ASCEND_HOME_PATH 环境变量"
    echo "使用默认路径: /usr/local/Ascend/cann"
    export ASCEND_HOME_PATH=/usr/local/Ascend/cann
fi

export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}

# 添加自定义算子库路径（查找已安装的 vendor 目录）
VENDOR_LIB_DIR=$(find "${ASCEND_HOME_PATH}/opp/vendors" -path "*/op_api/lib" -name "lib" -type d 2>/dev/null | grep -v "^${ASCEND_HOME_PATH}/opp/vendors/vendors" | head -1)
if [ -n "${VENDOR_LIB_DIR}" ]; then
    export LD_LIBRARY_PATH=${VENDOR_LIB_DIR}:${LD_LIBRARY_PATH}
    echo "VENDOR_LIB_DIR: ${VENDOR_LIB_DIR}"
fi

echo "ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}"

echo "环境变量设置完成"
echo ""

# ============================================================================
# 创建构建目录
# ============================================================================
echo "创建构建目录: ${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

# ============================================================================
# CMake 配置
# ============================================================================
echo ""
echo "CMake 配置..."
cd "${BUILD_DIR}"

if [ "$EXAMPLE_MODE" = "graph" ]; then
    # cmake 不支持 -f 指定 CMakeLists 文件名，需临时替换后恢复
    _ORIG_CMAKE="${SCRIPT_DIR}/CMakeLists.txt"
    _BACKUP="${SCRIPT_DIR}/.CMakeLists_aclnn.bak"
    mv "${_ORIG_CMAKE}" "${_BACKUP}"
    cp "${SCRIPT_DIR}/CMakeLists_geir.txt" "${_ORIG_CMAKE}"
    trap 'mv "${_BACKUP}" "${_ORIG_CMAKE}" 2>/dev/null' EXIT
    cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE "${SCRIPT_DIR}"
    # 立即恢复，不依赖 trap（trap 作为兜底）
    mv "${_BACKUP}" "${_ORIG_CMAKE}"
    trap - EXIT
else
    cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE "${SCRIPT_DIR}"
fi

# ============================================================================
# 编译
# ============================================================================
echo ""
echo "编译调用示例..."
make -j$(nproc)

echo "编译成功"
echo ""

# ============================================================================
# 执行调用示例
# ============================================================================
echo "========================================"
echo "执行调用示例"
echo "========================================"

if [ "$EXAMPLE_MODE" = "graph" ]; then
    echo ""
    echo ">>> 运行图模式 (GE IR) 调用示例 <<<"
    echo ""
    ./test_geir_add_example 0
else
    echo ""
    echo ">>> 运行 aclnn 调用示例 <<<"
    echo ""
    cd bin
    ./test_aclnn_add_example
fi

EXAMPLE_RESULT=$?

# ============================================================================
# 输出结果
# ============================================================================
echo ""
echo "========================================"
if [ $EXAMPLE_RESULT -eq 0 ]; then
    echo "执行结果: PASS"
else
    echo "执行结果: FAIL"
fi
echo "========================================"
echo ""

exit $EXAMPLE_RESULT
