#!/usr/bin/env bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
set -euo pipefail  # 开启严格模式：出错立即退出、未定义变量报错、管道失败报错

# ===================== 1. 检查并设置 TILE_FWK_DEVICE_ID =====================
if [ -z "${TILE_FWK_DEVICE_ID:-}" ]; then  # 修复：兼容变量未定义的情况
    echo "❌ TILE_FWK_DEVICE_ID 变量未设置（值为空）, 设置device 0......"
    export TILE_FWK_DEVICE_ID=0
    echo "✅ TILE_FWK_DEVICE_ID 已设置为：${TILE_FWK_DEVICE_ID}"
else
    echo "✅ TILE_FWK_DEVICE_ID 已存在：${TILE_FWK_DEVICE_ID}"
fi

# ===================== 2. 检查并设置 PTO_TILE_LIB_CODE_PATH =====================
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)  # 获取脚本所在绝对目录
DEFAULT_PTO_PATH="${SCRIPT_DIR}/../../../../../pto_isa/pto-isa" # 代码仓的前一级目录

if [ -z "${PTO_TILE_LIB_CODE_PATH:-}" ]; then
    echo "❌ PTO_TILE_LIB_CODE_PATH 变量未设置（值为空）, 尝试设置为默认路径......"
    export PTO_TILE_LIB_CODE_PATH="${DEFAULT_PTO_PATH}"
else
    echo "✅ PTO_TILE_LIB_CODE_PATH 已存在：${PTO_TILE_LIB_CODE_PATH}"
    # 检查现有路径是否存在，不存在则重置为默认
    if [ ! -d "${PTO_TILE_LIB_CODE_PATH}" ]; then
        echo "❌ 现有 PTO_TILE_LIB_CODE_PATH 目录不存在，重置为默认路径......"
        export PTO_TILE_LIB_CODE_PATH="${DEFAULT_PTO_PATH}"
    fi
fi

# 检查目录是否存在，不存在则克隆源码
if [ -d "${PTO_TILE_LIB_CODE_PATH}" ]; then
    echo "✅ 目录存在：${PTO_TILE_LIB_CODE_PATH}"
else
    echo "❌ 目录不存在：${PTO_TILE_LIB_CODE_PATH}, 尝试拉取源码......"
    if ! command -v git &> /dev/null; then
        echo "❌ 错误：系统未安装git，请先执行 yum install git / apt install git 安装！"
        exit 1  # 安装git前退出，避免后续无效操作
    fi
    mkdir -p "${SCRIPT_DIR}/pto_isa" || { echo "❌ 创建目录失败！"; exit 1; }
    git clone https://gitcode.com/cann/pto-isa.git "${DEFAULT_PTO_PATH}" || {
        echo "❌ git clone 失败（网络/权限问题），请手动克隆！"
        exit 1
    }
    echo "✅ git clone 成功，PTO_TILE_LIB_CODE_PATH 设置为：${DEFAULT_PTO_PATH}"
    export PTO_TILE_LIB_CODE_PATH="${DEFAULT_PTO_PATH}"
fi

# 最终验证变量是否生效（脚本内）
echo -e "\n✅ 最终环境变量设置结果："
echo "TILE_FWK_DEVICE_ID = ${TILE_FWK_DEVICE_ID}"
echo "PTO_TILE_LIB_CODE_PATH = ${PTO_TILE_LIB_CODE_PATH}"
