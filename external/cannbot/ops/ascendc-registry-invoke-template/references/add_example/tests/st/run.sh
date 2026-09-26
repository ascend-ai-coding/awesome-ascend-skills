#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================
# add_example 算子 ST 测试执行脚本

set -e

# ============================================================================
# 配置
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USE_MOCK=""  # 默认使用 Real 模式（NPU）
BUILD_DIR=""  # 根据 USE_MOCK 动态设置
CASE_FILE=""
USE_TORCH=""  # 是否使用 PyTorch 测试

# ============================================================================
# 帮助信息
# ============================================================================
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --mock          使用 Mock 模式（CPU golden 验证，无需 NPU，仅 C++ 测试）"
    echo "  --real          使用 Real 模式（NPU 执行，默认）"
    echo "  --torch         使用 PyTorch 接入测试（仅支持 Real 模式）"
    echo "  --case <file>   执行指定的测试用例 YAML 文件（仅 C++ 测试）"
    echo "  --help          显示帮助信息"
    echo ""
    echo "Examples:"
    echo "  # C++ Real 模式批量测试（默认，需要 NPU）"
    echo "  $0"
    echo ""
    echo "  # C++ Mock 模式批量测试（无 NPU 环境）"
    echo "  $0 --mock"
    echo ""
    echo "  # PyTorch 测试（需要 NPU）"
    echo "  $0 --torch"
    echo ""
    echo "  # 执行单个测试用例（C++ Real 模式）"
    echo "  $0 --case testcases/case_002_mixed_values.yaml"
}

# ============================================================================
# 解析参数
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --mock)
            USE_MOCK="-DUSE_MOCK=ON"
            shift
            ;;
        --real)
            USE_MOCK=""
            shift
            ;;
        --torch)
            USE_TORCH="1"
            shift
            ;;
        --case)
            CASE_FILE="$2"
            shift 2
            ;;
        --help|-h)
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

# ============================================================================
# PyTorch 测试路径（仅支持 Real 模式）
# ============================================================================
if [ -n "$USE_TORCH" ]; then
    if [ -n "$USE_MOCK" ]; then
        echo "错误: PyTorch 测试不支持 Mock 模式"
        echo "说明: PyTorch 测试需要完整的 CANN/NPU 环境，请使用 C++ Mock 模式进行无 NPU 验证"
        echo "      或使用 $0 --mock 执行 C++ Mock 测试"
        exit 1
    fi

    TORCH_DIR="${SCRIPT_DIR}/torch"

    if [ ! -d "$TORCH_DIR" ]; then
        echo "错误: PyTorch 测试目录不存在: $TORCH_DIR"
        exit 1
    fi

    echo "========================================"
    echo "add_example 算子 ST 测试 (PyTorch)"
    echo "========================================"
    echo "模式: Real (NPU 执行)"
    echo "工作目录: ${TORCH_DIR}"
    echo "========================================"
    echo ""

    # 检查依赖
    echo "检查依赖..."
    if ! python3 -c "import torch" 2>/dev/null; then
        echo "错误: 未找到 PyTorch，请先安装: pip install torch"
        exit 1
    fi
    echo "PyTorch 版本: $(python3 -c "import torch; print(torch.__version__)")"

    if ! python3 -c "import torch_npu" 2>/dev/null; then
        echo "错误: 未找到 torch_npu，请先安装"
        exit 1
    fi
    if [ -z "$ASCEND_HOME_PATH" ]; then
        echo "警告: 未设置 ASCEND_HOME_PATH 环境变量"
        echo "建议设置: source set_env.sh (in CANN install directory)"
    fi
    echo "依赖检查完成"
    echo ""

    # 编译
    echo "========================================"
    echo "编译 PyTorch 适配层"
    echo "========================================"

    TORCH_BUILD_DIR="${TORCH_DIR}/build"
    mkdir -p "$TORCH_BUILD_DIR"
    cd "$TORCH_BUILD_DIR"

    echo "CMake 配置..."
    TORCH_CMAKE_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null)
    if [ -n "$TORCH_CMAKE_PATH" ]; then
        cmake .. -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PATH"
    else
        cmake ..
    fi

    if [ $? -ne 0 ]; then
        echo "错误: CMake 配置失败"
        exit 1
    fi

    echo ""
    echo "编译..."
    make -j$(nproc)

    if [ $? -ne 0 ]; then
        echo "错误: 编译失败"
        exit 1
    fi

    echo "编译成功"
    echo ""

    # 执行测试
    echo "========================================"
    echo "执行测试"
    echo "========================================"

    cd "$TORCH_DIR"
    python3 test.py --lib "${TORCH_BUILD_DIR}/libtorch_adapter.so"

    TEST_RESULT=$?

    echo ""
    echo "========================================"
    if [ $TEST_RESULT -eq 0 ]; then
        echo "测试结果: PASS ✓"
    else
        echo "测试结果: FAIL ✗"
    fi
    echo "========================================"
    echo ""

    exit $TEST_RESULT
fi

# ============================================================================
# C++ 原生测试路径
# ============================================================================
# 根据模式设置独立的构建目录，避免 mock/real 缓存冲突
if [ -n "$USE_MOCK" ]; then
    BUILD_DIR="${SCRIPT_DIR}/build-mock"
else
    BUILD_DIR="${SCRIPT_DIR}/build-real"
fi

# ============================================================================
# 显示配置信息
# ============================================================================
echo "========================================"
echo "add_example 算子 ST 测试 (C++)"
echo "========================================"
if [ -n "$USE_MOCK" ]; then
    echo "模式: Mock (CPU golden 验证)"
else
    echo "模式: Real (NPU 执行，默认)"
fi
echo "工作目录: ${SCRIPT_DIR}"
echo "========================================"
echo ""

# ============================================================================
# 检查依赖
# ============================================================================
echo "检查依赖..."

# 检查 cmake
if ! command -v cmake &> /dev/null; then
    echo "错误: 未找到 cmake"
    exit 1
fi

# 检查 g++
if ! command -v g++ &> /dev/null; then
    echo "错误: 未找到 g++"
    exit 1
fi

# 检查 AscendCL（Real 模式需要）
if [ -z "$USE_MOCK" ]; then
    if [ -z "$ASCEND_HOME_PATH" ]; then
        echo "警告: 未设置 ASCEND_HOME_PATH 环境变量"
        echo "建议设置: source set_env.sh (in CANN install directory)"
    fi
fi

echo "依赖检查完成"
echo ""

# ============================================================================
# 设置环境变量
# ============================================================================
echo "设置环境变量..."

# 设置 LD_LIBRARY_PATH（基于 ASCEND_HOME_PATH 动态查找）
CUSTOM_OP_LIB_DIR=""
for candidate_dir in \
    "${ASCEND_HOME_PATH}/opp/vendors/add_example_custom/op_api/lib" \
    "/usr/local/Ascend/opp/vendors/add_example_custom/op_api/lib" \
    "${HOME}/Ascend/opp/vendors/add_example_custom/op_api/lib"; do
    if [ -d "$candidate_dir" ]; then
        CUSTOM_OP_LIB_DIR="$candidate_dir"
        break
    fi
done
if [ -n "$CUSTOM_OP_LIB_DIR" ]; then
    export LD_LIBRARY_PATH=${CUSTOM_OP_LIB_DIR}:${LD_LIBRARY_PATH}
    echo "LD_LIBRARY_PATH: ${CUSTOM_OP_LIB_DIR}"
else
    echo "警告: 未找到 add_example_custom op_api lib 目录"
fi

# 设置日志级别（可选：DEBUG）
# export ASCEND_SLOG_PRINT_TO_STDOUT=1
# export ASCEND_GLOBAL_LOG_LEVEL=0

echo "环境变量设置完成"
echo ""

# ============================================================================
# 编译并安装算子（可选）
# ============================================================================
COMPILE_OPERATOR=${COMPILE_OPERATOR:-"false"}

if [ "$COMPILE_OPERATOR" == "true" ]; then
    echo "========================================"
    echo "编译并安装算子"
    echo "========================================"

    OPERATOR_DIR="${SCRIPT_DIR}/../.."
    cd "${OPERATOR_DIR}"

    # 清理旧构建
    if [ -d "build" ]; then
        echo "清理旧构建..."
        rm -rf build
    fi

    # 创建构建目录
    mkdir -p build
    cd build

    # 编译（包含 kernel binary）
    echo "编译算子..."
    bash ../build.sh --pkg --soc=ascend910b -j$(nproc)

    if [ $? -ne 0 ]; then
        echo "错误: 算子编译失败"
        exit 1
    fi

    # 安装
    echo "安装算子..."
    ./custom_opp_ubuntu_aarch64.run

    if [ $? -ne 0 ]; then
        echo "错误: 算子安装失败"
        exit 1
    fi

    echo "算子编译并安装成功"
    cd "${SCRIPT_DIR}"
    echo ""
fi

# ============================================================================
# 创建构建目录
# ============================================================================
echo "创建构建目录..."
mkdir -p "${BUILD_DIR}"

# ============================================================================
# CMake 配置
# ============================================================================
echo ""
echo "CMake 配置..."
cd "${BUILD_DIR}"

if [ -n "$USE_MOCK" ]; then
    cmake .. -DUSE_MOCK=ON
else
    cmake ..
fi

if [ $? -ne 0 ]; then
    echo "错误: CMake 配置失败"
    exit 1
fi

# ============================================================================
# 编译
# ============================================================================
echo ""
echo "编译测试程序..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "错误: 编译失败"
    exit 1
fi

echo "编译成功"
echo ""

# ============================================================================
# 执行测试
# ============================================================================
echo "========================================"
echo "执行测试"
echo "========================================"

if [ -n "$CASE_FILE" ]; then
    # 执行单个测试用例
    if [ ! -f "${SCRIPT_DIR}/${CASE_FILE}" ]; then
        echo "错误: 测试用例文件不存在: ${CASE_FILE}"
        exit 1
    fi

    echo "执行测试用例: ${CASE_FILE}"
    ./test_aclnn_add_example "${SCRIPT_DIR}/${CASE_FILE}"
else
    # 执行内置测试用例
    echo "执行内置测试用例..."
    ./test_aclnn_add_example
fi

TEST_RESULT=$?

# ============================================================================
# 输出结果
# ============================================================================
echo ""
echo "========================================"
if [ $TEST_RESULT -eq 0 ]; then
    echo "测试结果: PASS ✓"
else
    echo "测试结果: FAIL ✗"
fi
echo "========================================"
echo ""

exit $TEST_RESULT
