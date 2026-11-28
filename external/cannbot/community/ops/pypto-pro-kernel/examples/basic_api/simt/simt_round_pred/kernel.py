# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。五种取整、三个数值谓词，以及 `cast` / `bitcast` / `popcount` / `mul_hi` / `fmod`。

**SIMT 函数可以直接收 GM 张量**——不需要先 `make_tile` 再搬。这一例用的就是这个最简形状：
一个 `section_vector` 里一条启动，没有 tile、没有同步。

```python
@pl.vector_function(mode="simt", max_threads=N)
def body(src: pl.Tensor[[1, N], FP32], out: pl.Tensor[[1, N], FP32]):
    tid = pl.simt.linear_thread_idx()
    out[0, tid] = pl.simt.floor(src[0, tid])

@pl.jit()
def kernel(src: ..., out: ...):
    with pl.section_vector():
        body[N](src, out)
```

**五种取整的区别全在 `.5` 和负数上**，所以输入里专门放了 `±0.5 / ±1.5 / ±2.5 / ±1.7`。
只用 `1.2` 这种值的话五个算子有三个给出相同结果，判据分不开它们。

**`cast` 与 `bitcast` 是两件事**：前者做数值转换（`1.5 → 1`），后者原样重解释位模式
（`1.5 → 1069547520`）。和 GM 侧 `pl.make_tensor(dtype=)` 与 numpy 的 `view` / `astype`
是同一个区别（见 ``datamove/strided_view`` §五）。

`popcount` 要 **UINT32 或 UINT64**（返回 INT32）；`mul_hi` 的结果保持输入 dtype；
`fmod` 两个操作数都要 FP32，商向零截断、结果取被除数的符号。

第一手来源：`pypto/python/tests/st/pypto_pro/frontend/simt/scalar/` 的 15 个文件。
"""

import pypto_pro.language as pl

FP32 = pl.DT_FP32
I32 = pl.DT_INT32
U32 = pl.DT_UINT32
BOOL = pl.DT_BOOL

THREADS = 32

# 取整输出：五种取整 + fmod
ROWS_F = 6
(ROW_RINT, ROW_ROUND, ROW_FLOOR, ROW_CEIL, ROW_TRUNC, ROW_FMOD) = range(ROWS_F)

# 谓词输出（BOOL）
ROWS_B = 3
(ROW_ISNAN, ROW_ISINF, ROW_ISFINITE) = range(ROWS_B)

# 整数输出
ROWS_I = 4
(ROW_CAST, ROW_BITCAST, ROW_POPCOUNT, ROW_MUL_HI) = range(ROWS_I)

FMOD_DIVISOR = 2.0


@pl.vector_function(mode="simt", max_threads=THREADS)
def round_and_predicates(
    f: pl.Tensor[[1, THREADS], FP32],
    u: pl.Tensor[[1, THREADS], U32],
    i: pl.Tensor[[1, THREADS], I32],
    out_f: pl.Tensor[[ROWS_F, THREADS], FP32],
    out_b: pl.Tensor[[ROWS_B, THREADS], BOOL],
    out_i: pl.Tensor[[ROWS_I, THREADS], I32],
):
    tid = pl.simt.linear_thread_idx()
    fv = f[0, tid]

    out_f[ROW_RINT, tid] = pl.simt.rint(fv)
    out_f[ROW_ROUND, tid] = pl.simt.round(fv)
    out_f[ROW_FLOOR, tid] = pl.simt.floor(fv)
    out_f[ROW_CEIL, tid] = pl.simt.ceil(fv)
    out_f[ROW_TRUNC, tid] = pl.simt.trunc(fv)
    out_f[ROW_FMOD, tid] = pl.simt.fmod(fv, FMOD_DIVISOR)

    out_b[ROW_ISNAN, tid] = pl.simt.isnan(fv)
    out_b[ROW_ISINF, tid] = pl.simt.isinf(fv)
    out_b[ROW_ISFINITE, tid] = pl.simt.isfinite(fv)

    out_i[ROW_CAST, tid] = pl.simt.cast(fv, I32)
    out_i[ROW_BITCAST, tid] = pl.simt.bitcast(fv, I32)
    # ST 里也先 cast 一遍再交给 popcount——照抄，不省这一步。
    out_i[ROW_POPCOUNT, tid] = pl.simt.popcount(pl.simt.cast(u[0, tid], U32))
    out_i[ROW_MUL_HI, tid] = pl.simt.mul_hi(i[0, tid], i[0, tid])


@pl.jit()
def simt_round_and_predicates(
    f: pl.Tensor[[1, THREADS], FP32],
    u: pl.Tensor[[1, THREADS], U32],
    i: pl.Tensor[[1, THREADS], I32],
    out_f: pl.Tensor[[ROWS_F, THREADS], FP32],
    out_b: pl.Tensor[[ROWS_B, THREADS], BOOL],
    out_i: pl.Tensor[[ROWS_I, THREADS], I32],
):
    """**没有 tile，没有同步。** GM 张量直接进 SIMT 函数，这是最小形状。"""
    with pl.section_vector():
        round_and_predicates[THREADS](f, u, i, out_f, out_b, out_i)
