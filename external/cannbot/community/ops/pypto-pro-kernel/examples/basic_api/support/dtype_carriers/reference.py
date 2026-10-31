# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""22 个 `DT_*` 的位宽分组、host 载体表、以及哪些根本进不了搬运通道。

**每一格都是 2026-09-27 在 A5 上试出来的**，不是从签名或文档推的。三个证据源：

1. 安装源码 `runtime/jit.py` 的 `_PL_DTYPE_TO_TORCH`——只有 11 条映射，**其余 11 个 dtype 的
   host 张量 dtype 压根不检查**（`jit.py:2326` 是 `if expected is not None and ...`）。
2. `ir/op/block_ops.py` 的 `_A5_LOAD_COMBOS` / `_A5_STORE_COMBOS`——搬运通道支持的
   (位置, 排布, dtype) 组合表。
3. 板上逐个走一趟 GM→Vec→GM 直通。

第 1、2 条是内部符号，**不在测试里 import**（[skill constraints](../../../../SKILL.md#constraints-and-workspace-policy)：用到未公开
路径要有版本守卫）。它们只用来解释「为什么」，测试断言的是第 3 条的行为。
"""

# 按位宽分组。22 = 6 + 7 + 4 + 3 + 2。
BITS = {
    4: ["DT_FP4", "DT_FP4E1M2", "DT_FP4E2M1", "DT_HF4", "DT_INT4", "DT_UINT4"],
    8: [
        "DT_BOOL",
        "DT_FP8E4M3FN",
        "DT_FP8E5M2",
        "DT_FP8E8M0",
        "DT_HF8",
        "DT_INT8",
        "DT_UINT8",
    ],
    16: ["DT_BF16", "DT_FP16", "DT_INT16", "DT_UINT16"],
    32: ["DT_FP32", "DT_INT32", "DT_UINT32"],
    64: ["DT_INT64", "DT_UINT64"],
}

# 进不了搬运通道的 5 个：`_A5_LOAD_COMBOS` 里根本没出现，
# 板上报 `load: unsupported layout/dtype combination src=GM(ND,DT_X) -> dst=Vec(ND,DT_X)`。
#
# 其中 4 个是 4 位的：**硬件没有 int4 这类 4 位整数元素类型**，A5 的 4 位元素类型只有
# `FP4E2M1` / `FP4E1M2`（与 ascriptor 的 a5 dtype 表一致）。它们在枚举里是占位。
# `DT_BOOL` 是 8 位，有 `pl.reinterpret` 退路。
NOT_IN_DATA_PATH = ["DT_BOOL", "DT_FP4", "DT_HF4", "DT_INT4", "DT_UINT4"]

# 参与搬运的 17 个。GM(ND)→Vec(ND) 共 69 条组合 = 2² + 6² + 4² + 3² + 2²
# ——**同位宽任意配对，不跨位宽**。所以 GM 声明 INT8、Tile 声明 HF8 是合法的。
IN_DATA_PATH_COUNT = 17
ND_COMBO_COUNT = 69

# `_PL_DTYPE_TO_TORCH` 的 11 条：这些 dtype 的 host 张量 dtype **被严格检查**。
TORCH_DTYPE_ENFORCED = {
    "DT_FP16": "float16",
    "DT_BF16": "bfloat16",
    "DT_FP32": "float32",
    "DT_INT8": "int8",
    "DT_INT16": "int16",
    "DT_INT32": "int32",
    "DT_INT64": "int64",
    "DT_UINT8": "uint8",
    "DT_UINT64": "uint64",
    "DT_BOOL": "bool",
    "DT_HF8": "uint8",  # ← torch 没有 hif8，映射到裸字节
}

# 另外 11 个没有映射 → **host dtype 不检查**。这是本例最锋利的坑：
# 传错位宽的张量会跑通、不报错、数据错（见 test 里的 silent 用例）。
TORCH_DTYPE_UNCHECKED = [
    "DT_FP4",
    "DT_FP4E1M2",
    "DT_FP4E2M1",
    "DT_HF4",
    "DT_INT4",
    "DT_UINT4",
    "DT_FP8E4M3FN",
    "DT_FP8E5M2",
    "DT_FP8E8M0",
    "DT_UINT16",
    "DT_UINT32",
]

# 实测能走通的载体。`host_axis` 是 host 张量末轴相对逻辑元素数的比例。
#   1   —— 一样（整字节 dtype）
#   0.5 —— 一半（4 bit：两个元素一个字节）
WORKING_CARRIERS = [
    # (DT 名, torch dtype 名, host 末轴比例, 末轴是否必须 DYNAMIC, 备注)
    ("DT_FP32", "float32", 1, False, "映射表里有，严格检查"),
    ("DT_HF8", "uint8", 1, False, "torch 无 hif8，映射到 uint8 裸字节"),
    ("DT_FP8E4M3FN", "float8_e4m3fn", 1, False, "没映射，但 torch 有同名 dtype"),
    (
        "DT_FP8E8M0",
        "uint8",
        1,
        False,
        "torch 的 float8_e8m0fnu 在卡上建不出张量，用 uint8",
    ),
    (
        "DT_UINT32",
        "int32",
        1,
        False,
        "没映射，所以 int32 也收；uint32 在卡上建不出张量",
    ),
    (
        "DT_UINT64",
        "uint64",
        1,
        False,
        "映射表里有，只收 uint64；**张量要在 CPU 建好再拷**",
    ),
    (
        "DT_FP4E2M1",
        "uint8",
        0.5,
        True,
        "4 bit：此项演示 DYNAMIC，host 给字节数；固定注解也支持",
    ),
]

# torch_npu 在卡上建不出张量的三个 dtype（`zeros_like` / `zeros(device=)` 都报
# `ZerosLikeKernelNpuOpApi.cpp:26 ... error code is 161002`）。
# 办法：在 CPU 上建好再 `.to(device)`。这是 torch_npu 的限制，不是 pypto 的。
NO_DEVICE_SIDE_CREATION = ["uint32", "uint64", "float8_e8m0fnu"]
