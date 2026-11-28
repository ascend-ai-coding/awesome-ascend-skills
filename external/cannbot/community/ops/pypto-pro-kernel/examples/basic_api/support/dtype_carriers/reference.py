# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""导出 dtype 的位宽、当前运行时接受的 host 存储载体及数据通道集合。

存储载体可用于绑定参数，不代表该 dtype 的所有数据搬运或计算路径均受支持。
正向验证使用逐字节拷贝，不进行数值格式转换。
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
# The tested A5 transfer table admits FP4E2M1/FP4E1M2 for four-bit elements,
# not the four legacy names below. Packed integer payloads are a separate representation.
# `DT_BOOL` 是 8 位，有 `pl.reinterpret` 退路。
NOT_IN_DATA_PATH = ["DT_BOOL", "DT_FP4", "DT_HF4", "DT_INT4", "DT_UINT4"]

# 参与搬运的 17 个。GM(ND)→Vec(ND) 共 69 条组合 = 2² + 6² + 4² + 3² + 2²
# ——**同位宽任意配对，不跨位宽**。所以 GM 声明 INT8、Tile 声明 HF8 是合法的。
IN_DATA_PATH_COUNT = 17
ND_COMBO_COUNT = 69

# 运行时按 torch 提供的这些存储类型注册载体；多个成员表示同一参数允许的别名。
TORCH_DTYPE_ENFORCED = {
    "DT_FP16": ("float16",),
    "DT_BF16": ("bfloat16",),
    "DT_FP32": ("float32",),
    "DT_INT8": ("int8",),
    "DT_INT16": ("int16", "uint16"),
    "DT_INT32": ("int32",),
    "DT_INT64": ("int64",),
    "DT_UINT8": ("uint8",),
    "DT_UINT16": ("uint16", "int16"),
    "DT_UINT32": ("uint32", "int32"),
    "DT_UINT64": ("uint64",),
    "DT_BOOL": ("bool",),
    "DT_FP4": ("uint8",),
    "DT_FP4E2M1": ("uint8", "float4_e2m1fn_x2"),
    "DT_FP4E1M2": ("uint8",),
    "DT_HF4": ("uint8",),
    "DT_INT4": ("uint8",),
    "DT_UINT4": ("uint8",),
    "DT_FP8E4M3FN": ("uint8", "float8_e4m3fn"),
    "DT_FP8E5M2": ("uint8", "float8_e5m2"),
    "DT_FP8E8M0": ("uint8", "float8_e8m0fnu"),
    "DT_HF8": ("uint8",),
}
TORCH_DTYPE_UNCHECKED = []

# 实测能走通的载体。`host_axis` 是 host 张量末轴相对逻辑元素数的比例。
#   1   —— 一样（整字节 dtype）
#   0.5 —— 一半（4 bit：两个元素一个字节）
WORKING_CARRIERS = [
    # (DT 名, torch dtype 名, host 末轴比例, 末轴是否必须 DYNAMIC, 备注)
    ("DT_FP32", "float32", 1, False, "映射表里有，严格检查"),
    ("DT_HF8", "uint8", 1, False, "torch 无 hif8，映射到 uint8 裸字节"),
    ("DT_FP8E4M3FN", "float8_e4m3fn", 1, False, "接受对应 torch FP8 或 uint8 存储载体"),
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
        "接受 int32/uint32 同位宽载体；在 CPU 创建后搬到设备",
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
