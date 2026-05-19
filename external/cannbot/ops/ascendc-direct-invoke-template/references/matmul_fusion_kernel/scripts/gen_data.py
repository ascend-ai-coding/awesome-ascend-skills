#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


# ============================================================
# 测试数据生成脚本 — mxfp8 + eltwise 融合算子（主样板：mxfp8_matmul_div）
#
# 当前基线数学公式: Output = matmul(A, B) / D
#
# [USER] 新融合算子需按算子语义修改三处：
#   1. OUTPUT_DTYPE    — 必须与 Host Launcher 中 OutputType 一致（float/half/bf16）
#   2. gen_fusion_inputs() — 写出第二路输入 bin（Relu 等无第二路输入可返回空 dict）
#   3. compute_golden()    — 实现融合 golden（如 Mul: a*b*x、Add: a*b+c、Relu: max(a*b,0) 等）
#
# [SAMPLE] 以下文件名与 dtype 是 Div 样例的硬编码，新算子按需改名：
#   input/input_a.bin      — A 矩阵 (mxfp8 E4M3)          [PATTERN]
#   input/input_b.bin      — B 矩阵 (mxfp8 E4M3)          [PATTERN]
#   input/input_scaleA.bin — A 的 per-block scale (e8m0) [PATTERN]
#   input/input_scaleB.bin — B 的 per-block scale (e8m0) [PATTERN]
#   input/input_d.bin      — [SAMPLE] Div 除数 [M,N] f32  ← 按算子改名
#   output/cpu_output.bin  — Golden 输出                 [PATTERN]
# ============================================================


import math
import os
import sys

import numpy as np
import torch
from en_dtypes import float8_e8m0
from ml_dtypes import float8_e4m3fn

# ============================================================
# [MODIFY] 输出数据类型 — 必须与 Host Launcher 中 OutputType 一致
# Host: using OutputType = float → np.float32
# ============================================================
OUTPUT_DTYPE = np.float32
# D 除数类型（与 Host DivisorType 一致）
DIVISOR_DTYPE = np.float32


def write_output(path, tensor):
    """按 OUTPUT_DTYPE 序列化输出张量。自动处理 bf16 等特殊类型。"""
    if OUTPUT_DTYPE == np.float32:
        tensor.to(torch.float32).numpy().tofile(path)
    elif OUTPUT_DTYPE == np.float16:
        tensor.to(torch.float16).numpy().tofile(path)
    elif OUTPUT_DTYPE == "bfloat16":
        tensor.to(torch.bfloat16).view(torch.uint16).numpy().tofile(path)
    else:
        tensor.numpy().astype(OUTPUT_DTYPE).tofile(path)


def write_artifacts(base_dir, a_fp8, b_fp8, a_scale, b_scale, fusion_inputs, out):
    """写入所有输入和输出文件。"""
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # matmul 基础输入
    a_fp8.view(np.uint8).tofile(os.path.join(input_dir, "input_a.bin"))
    b_fp8.view(np.uint8).tofile(os.path.join(input_dir, "input_b.bin"))
    a_scale.tofile(os.path.join(input_dir, "input_scaleA.bin"))
    b_scale.tofile(os.path.join(input_dir, "input_scaleB.bin"))

    # [EXTEND] 融合算子的额外输入（D 除数）
    for filename, data in fusion_inputs.items():
        data.tofile(os.path.join(input_dir, filename))

    # golden 输出
    write_output(os.path.join(output_dir, "cpu_output.bin"), out)


def gen_matmul_inputs(m, k, n):
    """生成 mxfp8 matmul 的基础输入。"""
    a_ori = np.random.uniform(1, 8, (m, k)).astype(float8_e4m3fn)
    b_ori = np.random.uniform(1, 8, (n, k)).astype(float8_e4m3fn)
    a_scale = np.random.uniform(1, 8, size=(m, math.ceil(k / 64), 2)).astype(float8_e8m0)
    b_scale = np.random.uniform(1, 8, size=(n, math.ceil(k / 64), 2)).astype(float8_e8m0)

    # 反量化计算 matmul golden
    a_scale_reshape = a_scale.reshape(m, -1)
    a_scale_broadcast = np.repeat(a_scale_reshape, 32, axis=-1)[..., :k]
    b_ori_transpose = np.swapaxes(b_ori, -1, -2)
    b_scale_reshape = b_scale.reshape(n, -1)
    b_scale_broadcast = np.repeat(b_scale_reshape, 32, axis=-1)[..., :k]
    b_scale_broadcast_transpose = np.swapaxes(b_scale_broadcast, -1, -2)

    a_dequant = a_ori.astype(np.float32) * a_scale_broadcast.astype(np.float32)
    b_dequant = b_ori_transpose.astype(np.float32) * b_scale_broadcast_transpose.astype(np.float32)

    a_cpu = torch.from_numpy(a_dequant)
    b_cpu = torch.from_numpy(b_dequant)
    matmul_out = torch.matmul(a_cpu, b_cpu).to(torch.float32)

    return a_ori, b_ori, a_scale, b_scale, matmul_out


def gen_fusion_inputs(m, n):
    """
    [EXTEND] 生成 D 除数张量 [M, N]。
    使用正值避免除零问题。
    """
    # D 除数：正值范围 [0.5, 5.0]，避免除零和极小值
    d = np.random.uniform(0.5, 5.0, size=(m, n)).astype(DIVISOR_DTYPE)
    return {"input_d.bin": d}, torch.from_numpy(d)


def compute_golden(matmul_out, fusion_tensors):
    """
    [EXTEND] 计算融合算子的 golden 输出: Output = matmul(A, B) / D
    """
    d_tensor = fusion_tensors  # torch.Tensor [M, N]
    # Div 在 float 精度下执行
    golden = matmul_out.to(torch.float32) / d_tensor.to(torch.float32)
    return golden.to(torch.float32)


def gen_golden_data(m, k, n):
    """主生成函数。"""
    a_ori, b_ori, a_scale, b_scale, matmul_out = gen_matmul_inputs(m, k, n)
    fusion_inputs, fusion_tensors = gen_fusion_inputs(m, n)
    out = compute_golden(matmul_out, fusion_tensors)

    current_dir = os.getcwd()
    write_artifacts(current_dir, a_ori, b_ori, a_scale, b_scale, fusion_inputs, out)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.normcase(os.path.abspath(script_dir)) != os.path.normcase(os.path.abspath(current_dir)):
        write_artifacts(script_dir, a_ori, b_ori, a_scale, b_scale, fusion_inputs, out)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 gen_data.py m k n")
        sys.exit(1)

    m = int(sys.argv[1])
    k = int(sys.argv[2])
    n = int(sys.argv[3])

    gen_golden_data(m, k, n)
