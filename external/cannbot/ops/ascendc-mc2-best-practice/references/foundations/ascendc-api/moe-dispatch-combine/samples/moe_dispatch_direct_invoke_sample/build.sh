# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


#!/bin/bash
set -e

# ============================================================================
# Ascend C MoE Dispatch 单算子直调工程编译脚本
#
# 用法：bash build.sh
#
# 前置条件：
#   export ASCEND_HOME_PATH=/path/to/CANN
#
# 产物：
#   build/lib/libascendc_kernels.so  — Kernel 共享库
#   build/test_moe_dispatch          — 多 rank 测试可执行文件
# ============================================================================

echo "Setting up environment..."
[ -n "${ASCEND_HOME_PATH:-}" ] || { echo "ERROR: ASCEND_HOME_PATH 未设置，请先配置 CANN 环境" >&2; exit 1; }
source "${ASCEND_HOME_PATH}/set_env.sh" || { echo "ERROR: set_env.sh 执行失败" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building in: $SCRIPT_DIR/build"
rm -rf build
mkdir -p build
cd build

echo "Running CMake..."
cmake ..

echo "Compiling..."
make -j$(nproc)

if [ -f "lib/libascendc_kernels.so" ] && [ -f "test_moe_dispatch" ]; then
	echo ""
	echo "Build successful!"
	echo "  Kernel library: $(pwd)/lib/libascendc_kernels.so"
	echo "  Test binary:    $(pwd)/test_moe_dispatch"
	echo ""
	ls -lh lib/libascendc_kernels.so test_moe_dispatch
else
	echo "Build failed!"
	exit 1
fi