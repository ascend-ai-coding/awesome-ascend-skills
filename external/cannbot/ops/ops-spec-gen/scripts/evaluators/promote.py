# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Numpy-style dtype promotion table.

覆盖 dtype：
  * 标准浮点 fp16 / bf16 / fp32 / fp64
  * 窄浮点（Atlas A3 / 950）fp4_e2m1 / fp4_e1m2 / fp8_e4m3fn / fp8_e5m2 / fp8_e8m0 / hf8 → 走 fp16 stand-in
  * 整数 int4 / int8 / int16 / int32 / int64 + uint1 / uint4 / uint8 / uint16 / uint32 / uint64 + bool
  * 复数 complex32 / complex64 / complex128

三个独立 rank 表（避免 fp / complex 混排引起阅读混乱）：
  * _FLOAT_FP_RANK  — 真实浮点 + 窄浮点
  * _COMPLEX_RANK   — 复数；隐含 "fp32 / fp64 的 complex 包装"
  * _INT_RANK       — 整数（含 bool / int4 / uint / int 各宽度）

Promotion 规则：
  * 同 dtype 直接返回（含 int4+int4 → int4 / fp8+fp8 → fp8）
  * Complex 吸收一切 → complex
  * 异型窄浮点（fp8_e4m3fn + fp8_e5m2 / fp4 + fp8 / hf8 + fp8）→ fp16
  * narrow + 任意非 narrow → 至少升 fp16
  * int + float → 按 _INT_REQUIRED_FP_RANK 抬到能装下 int 量级的 float（int32+fp16 必升 fp64）
  * 同 rank signed/unsigned int → 升一档 signed
  * bool + int4 / bool + uint4 → int4 / uint4
"""

from __future__ import annotations

from .types import DslError


_FLOAT_FP_RANK = {
    # 窄浮点（rank 0）：FP4 / FP8 系列。numpy 都没有原生类型，stage 8 走 fp16 stand-in。
    # 排名上比 fp16/bf16 窄一档；任意 narrow + 非 narrow → 至少升 fp16；
    # narrow1 + narrow2（不同名）→ fp16（混合窄浮点不安全，统一升 fp16）。
    "float4_e2m1": 0,    # 4-bit float (1 符号 + 2 指数 + 1 尾数)；Atlas 950 极致量化
    "float4_e1m2": 0,    # 4-bit float (1 符号 + 1 指数 + 2 尾数)；动态范围更小但精度略高
    "float8_e4m3fn": 0,    # 8-bit float (1 符号 + 4 指数 + 3 尾数，无 Inf)；Atlas A3 训练 / 推理
    "float8_e5m2": 0,    # 8-bit float (1 符号 + 5 指数 + 2 尾数)；动态范围更大
    "float8_e8m0": 0,    # 8-bit float (全指数，0 尾数)；OCP MXFP8 缩放因子专用
    "hifloat8": 0,    # Huawei Float 8 自研变体；Atlas 950
    "float16": 1,
    "bfloat16": 1,
    "float32": 2,
    "float64": 3,
}
_COMPLEX_RANK = {
    "complex32": 1,   # 位宽类似 fp16+fp16；PyTorch 实验性，ascend 暂未广泛部署
    "complex64": 2,   # 位宽类似 fp32+fp32
    "complex128": 3,   # 位宽类似 fp64+fp64
}
_INT_RANK = {
    "bool": 0,
    "uint1": 0,                      # 1-bit unsigned int (0/1)；与 bool 同 rank（值域同最窄）
    "int4": 0, "uint4": 0,         # int4 与 bool 同 rank（值域最窄；[-8,7] / [0,15]）
    "int8": 1, "uint8": 1,
    "int16": 2, "uint16": 2,
    "int32": 3, "uint32": 3,
    "int64": 4, "uint64": 4,
}
# Minimum float rank that can hold the magnitude of an int dtype without precision loss.
_INT_REQUIRED_FP_RANK = {
    "bool": 1,
    "uint1": 1,                      # uint1 ⊆ {0,1}，fp16 mantissa=10 远超
    "int4": 1, "uint4": 1,         # int4 在 fp16 mantissa=10 内
    "int8": 1, "uint8": 1,
    "int16": 2, "uint16": 2,
    "int32": 3, "uint32": 3,
    "int64": 3, "uint64": 3,
}
# Same-rank signed/unsigned integer pair → next wider signed int. 显式表，避免依赖 dict 顺序。
# bool / uint1 / int4 / uint4 均在 rank 0；理论上不会在 spec 里混用，但补全防止 fall-through。
_INT_SIGNED_PAIR_PROMOTE = {
    frozenset(("bool", "uint1")): "uint1",
    frozenset(("bool", "int4")): "int4",
    frozenset(("bool", "uint4")): "uint4",
    frozenset(("uint1", "int4")): "int4",
    frozenset(("uint1", "uint4")): "uint4",
    frozenset(("int4", "uint4")): "int8",
    frozenset(("int8", "uint8")): "int16",
    frozenset(("int16", "uint16")): "int32",
    frozenset(("int32", "uint32")): "int64",
    frozenset(("int64", "uint64")): "int64",   # 没有更宽的；numpy 行为
}
_ALL_DTYPES = set(_FLOAT_FP_RANK) | set(_COMPLEX_RANK) | set(_INT_RANK)

# 窄浮点集合（rank=0；混型必升 fp16）。模块级常量，避免 promote_pair 调用时重建。
_NARROW_FLOATS = frozenset({d for d, r in _FLOAT_FP_RANK.items() if r == 0})


def _is_float(d: str) -> bool:
    return d in _FLOAT_FP_RANK


def _is_complex(d: str) -> bool:
    return d in _COMPLEX_RANK


def _is_int(d: str) -> bool:
    return d in _INT_RANK


def _fp_or_complex_rank(d: str) -> int:
    """取 fp 或 complex 的宽度排名（以 fp64=3 为宽上限）。"""
    if d in _FLOAT_FP_RANK:
        return _FLOAT_FP_RANK[d]
    if d in _COMPLEX_RANK:
        return _COMPLEX_RANK[d]
    return 0


def _promote_complex(a, b):
    rank = max(_fp_or_complex_rank(a), _fp_or_complex_rank(b))
    return "complex128" if rank >= 3 else "complex64"


def _promote_with_float(a, b):
    fa = _FLOAT_FP_RANK.get(a, 0)
    fb = _FLOAT_FP_RANK.get(b, 0)
    if _is_int(a):
        fa = max(fa, _INT_REQUIRED_FP_RANK[a])
    if _is_int(b):
        fb = max(fb, _INT_REQUIRED_FP_RANK[b])
    if a in _NARROW_FLOATS or b in _NARROW_FLOATS:
        fa = max(fa, 1)
        fb = max(fb, 1)
    target_rank = max(fa, fb)
    if target_rank <= 1:
        if "bfloat16" in (a, b) and "float16" not in (a, b):
            return "bfloat16"
        return "float16"
    if target_rank == 2:
        return "float32"
    return "float64"


def _promote_int_pair(a, b):
    ra, rb = _INT_RANK[a], _INT_RANK[b]
    if ra > rb:
        return a
    if rb > ra:
        return b
    pair = frozenset((a, b))
    if pair in _INT_SIGNED_PAIR_PROMOTE:
        return _INT_SIGNED_PAIR_PROMOTE[pair]
    return "int64"


def promote_pair(a: str, b: str) -> str:
    """Promote two dtypes to a single dtype per numpy-ish rules."""
    if a not in _ALL_DTYPES:
        raise DslError("dsl_eval_error", f"未知 dtype: {a!r}")
    if b not in _ALL_DTYPES:
        raise DslError("dsl_eval_error", f"未知 dtype: {b!r}")
    if a == b:
        return a
    if _is_complex(a) or _is_complex(b):
        return _promote_complex(a, b)
    if {a, b} == {"float16", "bfloat16"}:
        return "float32"
    if a in _NARROW_FLOATS and b in _NARROW_FLOATS:
        return "float16"
    if _is_float(a) or _is_float(b):
        return _promote_with_float(a, b)
    return _promote_int_pair(a, b)


def promote_many(dtypes: list[str]) -> str:
    if not dtypes:
        raise DslError("dsl_eval_error", "promote(...) 至少需要 1 个参数")
    out = dtypes[0]
    for d in dtypes[1:]:
        out = promote_pair(out, d)
    return out


# ---------- additional v1+ dtype helpers -----------------------------------
# 5 个工具函数 —— 给 DtypeSolver 的对应 handler 使用。

# upcast_for_accum: fp16/bf16 → fp32；fp32/fp64/int/complex/narrow 保持不变
def upcast_for_accum(d: str) -> str:
    if d not in _ALL_DTYPES:
        raise DslError("dsl_eval_error", f"未知 dtype: {d!r}")
    if d in ("float16", "bfloat16"):
        return "float32"
    return d


# downcast_to_input: 用于"累加在 fp32, 输出回 fp16/bf16" 的语义。返回 target_dtype
def downcast_to_input(accum_dtype: str, target_dtype: str) -> str:
    if target_dtype not in _ALL_DTYPES:
        raise DslError("dsl_eval_error", f"未知 dtype: {target_dtype!r}")
    if accum_dtype not in _ALL_DTYPES:
        raise DslError("dsl_eval_error", f"未知 dtype: {accum_dtype!r}")
    return target_dtype


# widen_to: 显式拓宽到至少 <target>。若 input 已比 target 宽，保留 input
def widen_to(d: str, target: str) -> str:
    if d not in _ALL_DTYPES:
        raise DslError("dsl_eval_error", f"未知 dtype: {d!r}")
    if target not in _ALL_DTYPES:
        raise DslError("dsl_eval_error", f"未知 widen_to 目标 dtype: {target!r}")
    return promote_pair(d, target)


# complex_of: float ↔ complex（fp16→c32 / fp32→c64 / fp64→c128）
# 注意：bfloat16 没有对应的 complex 变体（complex 的实部仅有 fp16/fp32/fp64 三档），
# 此处显式不收 bf16，调用方拿到 DslError 后应自行决定升 fp32 再走 complex_of，
# 而不是被静默映射到 complex32（mantissa 7 bit vs fp16 的 10 bit 差异会丢精度）。
_FLOAT_TO_COMPLEX = {
    "float16": "complex32",
    "float32": "complex64",
    "float64": "complex128",
}


def complex_of(d: str) -> str:
    if d in _COMPLEX_RANK:
        return d
    if d in _FLOAT_TO_COMPLEX:
        return _FLOAT_TO_COMPLEX[d]
    if d == "bfloat16":
        raise DslError(
            "dsl_eval_error",
            "complex_of 不接受 bfloat16（complex 的实部没有 bf16 变体）；"
            "请先把 bf16 提升到 fp32 再走 complex_of，或在 spec 中显式声明 complex64",
        )
    raise DslError("dsl_eval_error",
                   f"complex_of 仅接受 float16/float32/float64/complex 输入，得到 {d!r}")


# real_of: complex → 对应实部 float
_COMPLEX_TO_REAL = {
    "complex32": "float16",
    "complex64": "float32",
    "complex128": "float64",
}


def real_of(d: str) -> str:
    if d in _FLOAT_FP_RANK:
        return d
    if d in _COMPLEX_TO_REAL:
        return _COMPLEX_TO_REAL[d]
    raise DslError("dsl_eval_error",
                   f"real_of 仅接受 float/complex 输入，得到 {d!r}")
