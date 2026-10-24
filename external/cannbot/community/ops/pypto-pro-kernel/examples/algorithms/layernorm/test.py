# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""四问：

1. 解决什么：512×200 FP32 逐行 LayerNorm（无仿射，有偏/总体方差），
   16 个 AIV 核按行均分（每核 32 行），eps=1e-5 编译期常量。
2. 物理访问是什么：每核一次性把自己的 32×200 行块 load 进 UB，`fillpad(EXPAND)`
   垫到 32×256（4 个 64-lane 寄存器块，第 4 块只有 8 个真实 lane），逐行在寄存器里
   算 mean/var/normalize，`store` 时用 `set_validshape` 只写回真实的 200 列。
3. 为何这样同步：两个 tile 各一个单槽 `make_tile_group`，`auto_mutex=True` 按
   mutex_ids 推 load → fillpad → vf 计算 → store 之间的顺序，细节见 kernel.py 顶部。
4. 如何独立验证：`reference.py` 用 float64 中间精度直接按数学定义算 mean/var/normalize，
   不复现 kernel 的分块/掩码路径；容差推导见下方 RTOL/ATOL 注释。
"""

import os

import pytest
import torch

from .kernel import COLS, NUM_CORES, ROWS, layernorm_kernel
from .reference import layernorm_reference

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

# 容差依据：kernel 侧每行的求和树是「4 个 64-lane 块内归约（硬件二叉树，结果确定）
# + 3 次标量相加」，分组方式与 numpy 的求和顺序不同，不追求逐位相同。
# 单次 FP32 加法的相对舍入 ~= eps(1.19e-7)；树深度 log2(256) = 8，
# sum -> mean -> var -> std -> div 链条上保守估计误差不超过几十倍 eps，
# 即量级 1e-5～1e-6。RTOL=1e-4 留出数倍安全边际；ATOL=1e-4 兜底归一化结果
# 接近 0 的行（相对误差在那里没有意义）。这是从链路深度推出来的初始值，
# 不是失败后反向放宽的值。
RTOL = 1e-4
ATOL = 1e-4


def _require_a5(device):
    # `torch.npu` 这个命名空间是 import torch_npu 的副作用，不 import 就没有这个属性
    # （见 examples/conftest.py 的 device fixture，这里内联同一检查，
    # 让 test.py 不依赖样例库自己的 conftest 也能独立跑）。
    pytest.importorskip("torch_npu")
    try:
        torch.npu.set_device(device)
    except (RuntimeError, AttributeError) as exc:
        pytest.skip(f"NPU unavailable: {exc}")
    name = torch.npu.get_device_name()
    if "Ascend950" not in name:
        pytest.skip(f"Current device is {name}, not A5 (Ascend950). Skip.")


def test_layernorm_multicore():
    device = ST_DEVICE
    _require_a5(device)
    torch.manual_seed(0)
    x = torch.randn(ROWS, COLS, device=device, dtype=torch.float32)
    out = torch.empty(ROWS, COLS, device=device, dtype=torch.float32)

    layernorm_kernel[None, NUM_CORES](x, out)
    torch.npu.synchronize()

    ref = layernorm_reference(x.cpu().numpy())
    torch.testing.assert_close(out.cpu(), torch.from_numpy(ref), rtol=RTOL, atol=ATOL)
