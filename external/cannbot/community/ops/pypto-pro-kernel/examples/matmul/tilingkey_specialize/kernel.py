# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# ruff: noqa: F821
"""写法：语法糖（`pl.gemm`）。编译期特化：`tiling_key` 把几个选择折成 IR 里的常量。

`Dtype` 和 `TileBlock` **不是形参**——它们在函数体里当自由变量用，由 parser 在解析时
折成常量。所以这个文件顶上必须有 `# ruff: noqa: F821`：对 linter 来说这两个名字未定义。
"""

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField

M, N, K = 512, 512, 128

# 两个 block 尺寸都要能整除 M 和 N。512 对 64 和 128 都行。
TILE_BLOCKS = (64, 128)


class MatmulKey:
    """key 的声明。`values` 里写的是**真正要用的数**，不是 0/1 编号。

    `bits` 是这个字段在打包后的 64 位 key 里占几位，够放下 `values` 的**下标**即可——
    打包用的是下标而不是值本身（`TileBlock=128` 打出来是 1，不是 128），和 AscendC 的
    模板 tiling key 对齐。
    """

    Dtype = TilingKeyField(bits=1, values=[0, 1])  # 0: FP16   1: BF16
    TileBlock = TilingKeyField(bits=1, values=[64, 128])  # M/N 方向的块大小

    def is_valid(self, key):
        """非法组合在这里拒。`key` 是按声明顺序排的具体值元组。

        **这个函数不能碰 `self` 以外什么都行的那个 `self`**：schema 用
        `object.__new__(cls)` 造实例、**不走 `__init__`**，所以任何 `__init__` 里设的
        属性在这里都不存在，读了就是 `AttributeError`。它必须是 `key` 的纯谓词。

        这个 kernel 其实没有真正非法的组合，下面这条是**人为**加的，用来演示接口：
        BF16 配 64 块不给编。
        """
        dtype, tile_block = key
        return not (dtype == 1 and tile_block == 64)


@pl.jit(auto_mutex=True, tiling_key=MatmulKey, compile_timeout=300)
def matmul_keyed(
    a_ptr: pl.Ptr[pl.DT_UINT8],
    b_ptr: pl.Ptr[pl.DT_UINT8],
    out_ptr: pl.Ptr[pl.DT_UINT8],
):
    """形参只能是 `pl.Ptr[pl.DT_UINT8]`——dtype 是 key 决定的，绑定前写不出
    `pl.Tensor[..., io_dtype]`。类型由 `pl.make_tensor` 在体内补上。
    """
    if Dtype == 0:
        io_dtype = pl.DT_FP16
    else:
        io_dtype = pl.DT_BF16
    tile = TileBlock  # key 的值可以直接当数用，parser 会折成常量

    a = pl.make_tensor(a_ptr, [M, K], dtype=io_dtype)
    b = pl.make_tensor(b_ptr, [K, N], dtype=io_dtype)
    out = pl.make_tensor(out_ptr, [M, N], dtype=pl.DT_FP32)

    a_mat = pl.make_tile_group(
        type=pl.TileType(
            shape=[tile, K], dtype=io_dtype, target_memory=pl.MemorySpace.Mat
        ),
        addrs=0,
        mutex_ids="auto",
        depth=2,
    )
    b_mat = pl.make_tile_group(
        type=pl.TileType(
            shape=[K, tile], dtype=io_dtype, target_memory=pl.MemorySpace.Mat
        ),
        addrs=tile * K * 4,
        mutex_ids="auto",
        depth=2,
    )
    a_left = pl.make_tile_group(
        type=pl.TileType(
            shape=[tile, K], dtype=io_dtype, target_memory=pl.MemorySpace.Left
        ),
        addrs=0,
        mutex_ids="auto",
        depth=2,
    )
    b_right = pl.make_tile_group(
        type=pl.TileType(
            shape=[K, tile], dtype=io_dtype, target_memory=pl.MemorySpace.Right
        ),
        addrs=0,
        mutex_ids="auto",
        depth=2,
    )
    acc = pl.make_tile_group(
        type=pl.TileType(
            shape=[tile, tile], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc
        ),
        addrs=0,
        mutex_ids="auto",
        depth=2,
    )

    with pl.section_cube():
        for m_idx in pl.range(0, M // tile):
            for n_idx in pl.range(0, N // tile):
                cur_a = a_mat.next()
                cur_b = b_mat.next()
                cur_acc = acc.next()
                pl.load_tile(cur_a, a, [m_idx, 0])
                pl.load_tile(cur_b, b, [0, n_idx])
                pl.gemm(cur_acc, cur_a, cur_b, l0a=a_left, l0b=b_right)
                pl.store_tile(out, cur_acc, [m_idx, n_idx])


class NeedsInitKey:
    """反例：`is_valid` 读了 `__init__` 设的属性。

    `TilingKeySchema(NeedsInitKey)` **构造得过去**——它只是把 `is_valid` 绑到一个
    `object.__new__` 出来的空实例上，不调用。等到真的调它（`enumerate_valid()` 或
    `validate_concrete()`）才 `AttributeError`。只被测试用到。
    """

    Dtype = TilingKeyField(bits=1, values=[0, 1])

    def __init__(self):
        self.allowed = {0}

    def is_valid(self, key):
        return key[0] in self.allowed
