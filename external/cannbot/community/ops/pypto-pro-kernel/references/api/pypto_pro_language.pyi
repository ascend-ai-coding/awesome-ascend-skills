# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Generated agent reference for import pypto_pro.language as pl.
# Read top-level names as pl.<name>; Vf is pl.Vf (often imported as vf).
# Snapshot of source declarations; Any and missing annotations are retained.
# This file is for lookup, not installed IDE stubs or proof of device support.
# Source SHA256 (ordered paths and contents): d65bee02d1858d28dd776058f7964e7c196802527709cdfbc5682728bca380f9
# Regenerate: python scripts/check.py api --write-pyi
# Verify against the selected source: python scripts/check.py api --check-pyi
# Select a different source with --pypto-root before the api subcommand.
# Contracts: ../index.md, constraints.md, memory-allocation.md, vf.md, simt.md,
# gemm.md, datamove.md, crosscore-sync.md, fixpipe.md and support.md.
# Enum member names come from Python bindings; numeric encodings are omitted.

from collections.abc import Callable
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pypto.pypto_impl import ir as _ir_core
from pypto.pypto_impl.ir import Call
from pypto.pypto_impl.ir import DataType
from pypto.pypto_impl.ir import Expr
from pypto.pypto_impl.ir import Span
from pypto_pro.language.typing.direction import TensorDirection
from pypto_pro.language.typing.shape import _ShapePolicy
from pypto_pro.language.typing.tensor import EllipsisType
from pypto_pro.runtime.pipeline.config import PipelineConfig
from typing import Any
from typing import List
from typing import Optional
from typing import Union
import ast
import enum

# Source: python/pypto_pro/language/_api.py
Tile = Any
TileGroup = Any
DType = Any
Offset = Union[List[int], int]

# Source: python/pypto_pro/language/parser/decorator.py
def inline(fn: Callable) -> Callable: ...
def vector_function(
    fn: Callable | None = None,
    *,
    mode: str | None = None,
    max_threads: int | None = None,
) -> Callable: ...

# Source: python/pypto_pro/language/__init__.py
from pypto_pro.language import parser as parser

# Source: python/pypto_pro/language/typing/tensor.py
class Tensor:
    def __init__(
        self,
        shape: Sequence[int | _ShapePolicy | EllipsisType] | None = None,
        dtype: DataType | None = None,
        expr: Expr | TensorDirection | None = None,
        layout: TensorLayout | None = None,
        memref: MemRef | None = None,
        _annotation_only: bool = False,
    ): ...
    @classmethod
    def __class_getitem__(cls, item: tuple) -> "Tensor": ...
    def __repr__(self) -> str: ...
    def unwrap(self) -> Expr: ...

# Source: python/pypto_pro/language/typing/ptr.py
class Ptr:
    def __init__(
        self,
        dtype: DataType | None = None,
        expr: Expr | None = None,
        _annotation_only: bool = False,
    ): ...
    def __repr__(self) -> str: ...
    @classmethod
    def __class_getitem__(cls, dtype: DataType) -> "Ptr": ...
    def unwrap(self) -> Expr: ...

# Source: python/pypto_pro/language/typing/scalar.py
class Scalar:
    def __init__(self, expr: Expr | None = None): ...
    def __repr__(self) -> str: ...
    def unwrap(self) -> Expr: ...

# Source: python/pypto_pro/language/typing/shape.py
DYNAMIC = _ShapePolicy("DYNAMIC")
STATIC = _ShapePolicy("STATIC")

# Source: python/pypto_pro/ir/op/system_ops.py
class system:
    MAX_EVENT_ID = 15
    MAX_FLAG_EVENT_ID = 7
    MAX_MUTEX_ID = 31
    @staticmethod
    def sync_src(
        *,
        set_pipe: PipeType,
        wait_pipe: PipeType,
        event_id: int | Expr,
        span: Span | None = None,
    ) -> Call: ...
    @staticmethod
    def sync_dst(
        *,
        set_pipe: PipeType,
        wait_pipe: PipeType,
        event_id: int | Expr,
        span: Span | None = None,
    ) -> Call: ...
    @staticmethod
    def bar_m(*, span: Span | None = None) -> Call: ...
    @staticmethod
    def bar_mte1(*, span: Span | None = None) -> Call: ...
    @staticmethod
    def bar_mte2(*, span: Span | None = None) -> Call: ...
    @staticmethod
    def bar_mte3(*, span: Span | None = None) -> Call: ...
    @staticmethod
    def bar_fix(*, span: Span | None = None) -> Call: ...
    @staticmethod
    def bar_all(*, span: Span | None = None) -> Call: ...
    @staticmethod
    def set_cross_core(
        *,
        pipe: PipeType,
        event_id: int | Expr,
        sync_mode: CrossCoreSyncMode = CrossCoreSyncMode.INTRA_BLOCK,
        span: Span | None = None,
    ) -> Call: ...
    @staticmethod
    def wait_cross_core(
        *,
        pipe: PipeType,
        event_id: int | Expr,
        sync_mode: CrossCoreSyncMode = CrossCoreSyncMode.INTRA_BLOCK,
        span: Span | None = None,
    ) -> Call: ...
    @staticmethod
    def sync_all(
        *, core_type: SyncCoreType = SyncCoreType.MIX, span: Span | None = None
    ) -> Call: ...
    @staticmethod
    def dcci(
        target: Expr,
        offset: int | Expr | tuple[int | Expr, ...] | list[int | Expr] | None = None,
        *,
        cache_line: CacheLine = CacheLine.ENTIRE_DATA_CACHE,
        dst: DcciDst = DcciDst.AUTO,
        span: Span | None = None,
    ) -> Call: ...
    @staticmethod
    def mutex_lock(
        *, pipe: PipeType, mutex_id: int | Expr, span: Span | None = None
    ) -> Call: ...
    @staticmethod
    def mutex_unlock(
        *, pipe: PipeType, mutex_id: int | Expr, span: Span | None = None
    ) -> Call: ...
    @staticmethod
    def set_mm_layout_transform(*, enabled: bool, span: Span | None = None) -> Call: ...

# Source: python/pypto_pro/language/_simt_api.py
class simt:
    @staticmethod
    def thread_idx() -> Any: ...
    @staticmethod
    def block_dim() -> Any: ...
    @staticmethod
    def block_idx() -> Any: ...
    @staticmethod
    def grid_dim() -> Any: ...
    @staticmethod
    def linear_thread_idx() -> Scalar: ...
    @staticmethod
    def warp_size() -> Scalar: ...
    @staticmethod
    def syncthreads() -> None: ...
    @staticmethod
    def threadfence_block() -> None: ...
    @staticmethod
    def threadfence() -> None: ...
    @staticmethod
    def cast(
        value: Scalar, dtype: DType, *, mode: RoundMode = RoundMode.CAST_NONE
    ) -> Scalar: ...
    @staticmethod
    def bitcast(value: Scalar, dtype: DType) -> Scalar: ...
    @staticmethod
    def abs(value: Scalar) -> Scalar: ...
    @staticmethod
    def min(lhs: Scalar, rhs: Scalar) -> Scalar: ...
    @staticmethod
    def max(lhs: Scalar, rhs: Scalar) -> Scalar: ...
    @staticmethod
    def sqrt(value: Scalar) -> Scalar: ...
    @staticmethod
    def rsqrt(value: Scalar) -> Scalar: ...
    @staticmethod
    def exp(value: Scalar) -> Scalar: ...
    @staticmethod
    def exp2(value: Scalar) -> Scalar: ...
    @staticmethod
    def log(value: Scalar) -> Scalar: ...
    @staticmethod
    def log2(value: Scalar) -> Scalar: ...
    @staticmethod
    def log1p(value: Scalar) -> Scalar: ...
    @staticmethod
    def sin(value: Scalar) -> Scalar: ...
    @staticmethod
    def cos(value: Scalar) -> Scalar: ...
    @staticmethod
    def tanh(value: Scalar) -> Scalar: ...
    @staticmethod
    def rint(value: Scalar) -> Scalar: ...
    @staticmethod
    def round(value: Scalar) -> Scalar: ...
    @staticmethod
    def floor(value: Scalar) -> Scalar: ...
    @staticmethod
    def ceil(value: Scalar) -> Scalar: ...
    @staticmethod
    def trunc(value: Scalar) -> Scalar: ...
    @staticmethod
    def isnan(value: Scalar) -> Scalar: ...
    @staticmethod
    def isinf(value: Scalar) -> Scalar: ...
    @staticmethod
    def isfinite(value: Scalar) -> Scalar: ...
    @staticmethod
    def popcount(value: Scalar) -> Scalar: ...
    @staticmethod
    def mul_hi(lhs: Scalar, rhs: Scalar) -> Scalar: ...
    @staticmethod
    def fmod(lhs: Scalar, rhs: Scalar) -> Scalar: ...
    @staticmethod
    def fma(lhs: Scalar, rhs: Scalar, addend: Scalar) -> Scalar: ...
    @staticmethod
    def atomic_add(target: Scalar, value: Scalar) -> Scalar | None: ...
    @staticmethod
    def atomic_sub(target: Scalar, value: Scalar) -> Scalar: ...
    @staticmethod
    def atomic_exch(target: Scalar, value: Scalar) -> Scalar: ...
    @staticmethod
    def atomic_max(target: Scalar, value: Scalar) -> Scalar | None: ...
    @staticmethod
    def atomic_min(target: Scalar, value: Scalar) -> Scalar | None: ...
    @staticmethod
    def atomic_inc(target: Scalar, limit: Scalar) -> Scalar: ...
    @staticmethod
    def atomic_dec(target: Scalar, limit: Scalar) -> Scalar: ...
    @staticmethod
    def atomic_cas(target: Scalar, compare: Scalar, value: Scalar) -> Scalar: ...
    @staticmethod
    def atomic_and(target: Scalar, value: Scalar) -> Scalar: ...
    @staticmethod
    def atomic_or(target: Scalar, value: Scalar) -> Scalar: ...
    @staticmethod
    def atomic_xor(target: Scalar, value: Scalar) -> Scalar: ...

# Source: python/pypto_pro/language/__init__.py
class mutex:
    @staticmethod
    def mutex_lock(
        *, pipe: PipeType, mutex_id: int | Expr, span: Span | None = None
    ) -> Call: ...
    @staticmethod
    def mutex_unlock(
        *, pipe: PipeType, mutex_id: int | Expr, span: Span | None = None
    ) -> Call: ...

# Source: python/pypto_pro/ir/op/block_ops.py
@dataclass
class TileType:
    shape: "Sequence[int] | _ir_core.MakeTuple"
    dtype: DataType
    target_memory: MemorySpace = MemorySpace.Vec
    valid_shape: Optional[Sequence[int]] = None
    layout: Optional[TensorLayout] = None
    fractal: Optional[int] = None
    pad: Optional[int] = None
    compact: Optional[int] = None
    def __post_init__(self): ...

# Source: python/pypto_pro/language/_api.py
def make_tile(tile_type: Any, /, *, addr: int) -> Tile: ...

# Source: python/pypto_pro/ir/op/ptr_ops.py
def make_tensor(
    ptr: Expr,
    shape: Sequence[int | Expr] | _ir_core.MakeTuple,
    stride: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    dtype: DataType | None = None,
    span: Span | None = None,
) -> Call: ...
def make_ptr(
    ptr: Expr, dtype: DataType | None = None, span: Span | None = None
) -> Call: ...
def addptr(ptr: Expr, offset: int | Expr, span: Span | None = None) -> Call: ...

# Source: python/pypto_pro/language/__init__.py
def struct(*args: Any, **kwargs: Any) -> Any: ...
def make_tuple(**kwargs: Any) -> Any: ...
def struct_array(size: int, *args: Any, **kwargs: Any) -> Any: ...

# Source: python/src/bindings/ir/ir.cpp
class FunctionType:
    Opaque: FunctionType
    Orchestration: FunctionType
    InCore: FunctionType
    Helper: FunctionType
    SimtVF: FunctionType
    SimtCallee: FunctionType

# Source: python/pypto/pypto_impl/ir.pyi
class MemRef(Expr):
    memory_space: MemorySpace
    addr: Expr
    size: int
    def __init__(
        self,
        memory_space: MemorySpace,
        addr: Expr | int,
        size: int,
        span: Optional[Span] = None,
    ) -> None: ...
    @staticmethod
    def same_allocation(a: MemRef, b: MemRef) -> bool: ...
    @staticmethod
    def may_alias(a: MemRef, b: MemRef) -> bool: ...

# Source: python/src/bindings/ir/ir.cpp
class MemorySpace:
    DDR: MemorySpace
    Vec: MemorySpace
    Mat: MemorySpace
    Left: MemorySpace
    Right: MemorySpace
    Scaling: MemorySpace
    Acc: MemorySpace
    Bias: MemorySpace
    ScaleLeft: MemorySpace
    ScaleRight: MemorySpace

class TilePad:
    null: TilePad
    zero: TilePad
    max: TilePad
    min: TilePad

class PipeType:
    MTE1: PipeType
    MTE2: PipeType
    MTE3: PipeType
    M: PipeType
    V: PipeType
    S: PipeType
    FIX: PipeType
    ALL: PipeType

class TensorLayout:
    ND: TensorLayout
    DN: TensorLayout
    NZ: TensorLayout
    ZN: TensorLayout
    NN: TensorLayout
    ZZ: TensorLayout

# Source: python/pypto_pro/language/__init__.py
ND = TensorLayout.ND
DN = TensorLayout.DN
NZ = TensorLayout.NZ
ZN = TensorLayout.ZN
NN = TensorLayout.NN
ZZ = TensorLayout.ZZ

# Source: python/pypto_pro/language/typing/direction.py
Input = TensorDirection.INPUT
Output = TensorDirection.OUTPUT

# Source: python/src/bindings/ir/ir.cpp
class ReluPreMode:
    NormalRelu: ReluPreMode

class AtomicType:
    AtomicNone: AtomicType
    AtomicAdd: AtomicType

class STPhase:
    Unspecified: STPhase
    Partial: STPhase
    Final: STPhase

class AccPhase:
    Unspecified: AccPhase
    Partial: AccPhase
    Final: AccPhase

class AccToVecMode:
    SingleModeVec0: AccToVecMode
    SingleModeVec1: AccToVecMode
    DualModeSplitM: AccToVecMode
    DualModeSplitN: AccToVecMode

class RoundMode:
    CAST_NONE: RoundMode
    CAST_RINT: RoundMode
    CAST_ROUND: RoundMode
    CAST_FLOOR: RoundMode
    CAST_CEIL: RoundMode
    CAST_TRUNC: RoundMode
    CAST_ODD: RoundMode

class QuantMode:
    SYM: QuantMode
    ASYM: QuantMode

class CrossCoreSyncMode:
    INTER_BLOCK: CrossCoreSyncMode
    INTER_SUBBLOCK: CrossCoreSyncMode
    INTRA_BLOCK: CrossCoreSyncMode
    UNICAST_BLOCK: CrossCoreSyncMode

class SyncCoreType:
    AIV_ONLY: SyncCoreType
    AIC_ONLY: SyncCoreType
    MIX: SyncCoreType

class CacheLine:
    SINGLE_CACHE_LINE: CacheLine
    ENTIRE_DATA_CACHE: CacheLine

class DcciDst:
    AUTO: DcciDst
    CACHELINE_OUT: DcciDst
    CACHELINE_UB: DcciDst
    CACHELINE_ALL: DcciDst
    CACHELINE_ATOMIC: DcciDst

# Source: python/pypto_pro/language/__init__.py
DT_FP4 = DataType.FP4
DT_FP8E4M3FN = DataType.FP8E4M3FN
DT_FP8E5M2 = DataType.FP8E5M2
DT_FP8E8M0 = DataType.FP8E8M0
DT_FP4E2M1 = DataType.FP4E2M1
DT_FP4E1M2 = DataType.FP4E1M2
DT_FP16 = DataType.FP16
DT_FP32 = DataType.FP32
DT_BF16 = DataType.BF16
DT_HF4 = DataType.HF4
DT_HF8 = DataType.HF8
DT_INT4 = DataType.INT4
DT_INT8 = DataType.INT8
DT_INT16 = DataType.INT16
DT_INT32 = DataType.INT32
DT_INT64 = DataType.INT64
DT_UINT4 = DataType.UINT4
DT_UINT8 = DataType.UINT8
DT_UINT16 = DataType.UINT16
DT_UINT32 = DataType.UINT32
DT_UINT64 = DataType.UINT64
DT_BOOL = DataType.BOOL

# Source: python/pypto_pro/language/_api.py
def load(
    dst_tile: Tile,
    src_tensor: Tensor,
    /,
    offsets: Offset,
    *,
    order: Optional[List[int]] = None,
) -> None: ...
def load_tile(
    dst_tile: Tile,
    src_tensor: Tensor,
    /,
    tile_offsets: Offset,
    *,
    order: Optional[List[int]] = None,
) -> None: ...
def store(
    dst_tensor: Tensor,
    src_tile: Tile,
    /,
    offsets: Offset,
    *,
    relu_pre_mode: Optional[ReluPreMode] = None,
    scale: Optional[Union[float, Scalar, Tile]] = None,
    quant_offset: Optional[int] = None,
    order: Optional[List[int]] = None,
    atomic: AtomicType = AtomicType.AtomicNone,
    phase: Optional[STPhase] = None,
) -> None: ...
def store_tile(
    dst_tensor: Tensor,
    src_tile: Tile,
    /,
    tile_offsets: Offset,
    *,
    relu_pre_mode: Optional[ReluPreMode] = None,
    scale: Optional[Union[float, Scalar, Tile]] = None,
    quant_offset: Optional[int] = None,
    order: Optional[List[int]] = None,
    atomic: AtomicType = AtomicType.AtomicNone,
    phase: Optional[STPhase] = None,
) -> None: ...
def init_output(
    tensor: Tensor, *, offset: int = 0, size: int, value: Union[int, float, Scalar] = 0
) -> None: ...
def move(
    dst_tile: Tile,
    src_tile: Tile,
    /,
    offset: Optional[Offset] = None,
    *,
    acc_to_vec_mode: Optional[AccToVecMode] = None,
    relu_pre_mode: Optional[ReluPreMode] = None,
    scale: Optional[Union[float, Scalar, Tile]] = None,
    quant_offset: Optional[int] = None,
    phase: Optional[STPhase] = None,
) -> None: ...
def insert(
    dst_tile: Tile,
    src_tile: Tile,
    /,
    offset: List[int],
    *,
    relu_pre_mode: Optional[ReluPreMode] = None,
    scale: Optional[Union[float, Scalar, Tile]] = None,
    quant_offset: Optional[int] = None,
    phase: Optional[STPhase] = None,
) -> None: ...
def ssbuf_store(struct_var: Any, offset: int, /) -> None: ...
def ssbuf_load(struct_var: Any, offset: int, /) -> None: ...
def add(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None: ...
def adds(out: Tile, lhs: Tile, rhs: Scalar) -> None: ...
def sub(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None: ...
def subs(out: Tile, lhs: Tile, rhs: Scalar) -> None: ...
def mul(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None: ...
def muls(out: Tile, lhs: Tile, rhs: Scalar) -> None: ...
def div(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None: ...
def maximum(
    out: Tile, lhs: Tile, rhs: Union[Tile, Scalar], *, dim: Optional[int] = None
) -> None: ...
def minimum(
    out: Tile, lhs: Tile, rhs: Union[Tile, Scalar], *, dim: Optional[int] = None
) -> None: ...
def and_(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None: ...
def shl(out: Tile, lhs: Tile, rhs: Tile) -> None: ...
def shr(out: Tile, lhs: Tile, rhs: Tile) -> None: ...
def xor(out: Tile, lhs: Tile, rhs: Tile, tmp: Tile) -> None: ...
def expands(out: Tile, scalar: Scalar) -> None: ...
def neg(out: Tile, src: Tile) -> None: ...
def abs(out: Tile, src: Tile) -> None: ...
def exp(out: Tile, src: Tile) -> None: ...
def log(out: Tile, src: Tile) -> None: ...
def sqrt(out: Tile, src: Tile) -> None: ...
def rsqrt(out: Tile, src: Tile) -> None: ...
def recip(out: Tile, src: Tile) -> None: ...
def relu(out: Tile, src: Tile) -> None: ...
def fillpad(
    out: Tile, src: Tile, *, mode: FillPadMode = FillPadMode.NORMAL
) -> None: ...

# Source: python/pypto_pro/ir/op/block_ops.py
class FillPadMode(enum.Enum):
    NORMAL = 0
    EXPAND = 1
    INPLACE = 2

# Source: python/pypto_pro/language/_api.py
def cast(out: Tile, src: Tile, *, mode: RoundMode = RoundMode.CAST_ROUND) -> None: ...
def add_relu_cast(
    out: Tile,
    lhs: Tile,
    rhs: Tile,
    *,
    target_type: DType,
    mode: RoundMode = RoundMode.CAST_ROUND,
) -> None: ...
def sub_relu_cast(
    out: Tile,
    lhs: Tile,
    rhs: Tile,
    *,
    target_type: DType,
    mode: RoundMode = RoundMode.CAST_ROUND,
) -> None: ...
def mul_cast(
    out: Tile,
    lhs: Tile,
    rhs: Tile,
    *,
    target_type: DType,
    mode: RoundMode = RoundMode.CAST_ROUND,
) -> None: ...
def eq(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None: ...
def ne(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None: ...
def lt(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None: ...
def le(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None: ...
def gt(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None: ...
def ge(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None: ...
def select(
    out: Tile, mask: Tile, lhs: Tile, rhs: Union[Tile, Scalar], tmp: Tile
) -> None: ...
def add_relu(out: Tile, lhs: Tile, rhs: Tile) -> None: ...
def sub_relu(out: Tile, lhs: Tile, rhs: Tile) -> None: ...
def mul_add_dst(out: Tile, lhs: Tile, rhs: Tile) -> None: ...
def fused_mul_add(out: Tile, lhs: Tile, rhs: Tile) -> None: ...
def fused_mul_add_relu(out: Tile, a: Tile, b: Tile) -> None: ...
def axpy(out: Tile, src: Tile, alpha: Scalar, /) -> None: ...
def partadd(out: Tile, src0: Tile, src1: Tile) -> None: ...
def partmax(out: Tile, src0: Tile, src1: Tile) -> None: ...
def partmin(out: Tile, src0: Tile, src1: Tile) -> None: ...
def partmul(out: Tile, src0: Tile, src1: Tile) -> None: ...
def matmul(
    dst_tile: Tile,
    lhs_tile: Tile,
    rhs_tile: Tile,
    bias_tile: Optional[Tile] = None,
    /,
    *,
    phase: Optional[AccPhase] = None,
) -> None: ...
def matmul_acc(
    dst_tile: Tile,
    acc_tile: Tile,
    lhs_tile: Tile,
    rhs_tile: Tile,
    /,
    *,
    phase: Optional[AccPhase] = None,
) -> None: ...
def matmul_mx(
    dst_tile: Tile,
    lhs_tile: Tile,
    rhs_tile: Tile,
    /,
    scale_a: Tile,
    scale_b: Tile,
    *,
    phase: Optional[AccPhase] = None,
) -> None: ...
def matmul_mx_acc(
    dst_tile: Tile,
    acc_tile: Tile,
    lhs_tile: Tile,
    rhs_tile: Tile,
    /,
    scale_a: Tile,
    scale_b: Tile,
    *,
    phase: Optional[AccPhase] = None,
) -> None: ...
def gemm(
    acc: Tile,
    l1a: Tile,
    l1b: Tile,
    /,
    *,
    l0a: Union[Tile, TileGroup],
    l0b: Union[Tile, TileGroup],
    m: Optional[Scalar] = None,
    n: Optional[Scalar] = None,
    k: Optional[Scalar] = None,
    init: bool = True,
    splitn: Optional[int] = None,
    splitk: Optional[int] = None,
    trans_a: bool = False,
    trans_b: bool = False,
    bias: Optional[Tile] = None,
) -> None: ...
def gemm_mx(
    acc: Tile,
    l1a: Tile,
    l1b: Tile,
    l1a_scale: Tile,
    l1b_scale: Tile,
    /,
    *,
    l0a: Union[Tile, TileGroup],
    l0b: Union[Tile, TileGroup],
    scale_a: Union[Tile, TileGroup],
    scale_b: Union[Tile, TileGroup],
    m: Optional[Scalar] = None,
    n: Optional[Scalar] = None,
    k: Optional[Scalar] = None,
    init: bool = True,
    trans_a: bool = False,
    trans_b: bool = False,
) -> None: ...
def transpose(out: Tile, src: Tile) -> None: ...
def sum(out: Tile, src: Tile, tmp: Tile, *, dim: int = 0) -> None: ...
def argmax(out: Tile, src: Tile, tmp: Tile, *, dim: int = 0) -> None: ...
def argmin(out: Tile, src: Tile, tmp: Tile, *, dim: int = 0) -> None: ...
def expand_max(out: Tile, src: Tile, scalar: Tile, *, dim: int = 0) -> None: ...
def expand_min(out: Tile, src: Tile, scalar: Tile, *, dim: int = 0) -> None: ...
def expand_mul(out: Tile, src: Tile, scalar: Tile, *, dim: int = 0) -> None: ...
def expand_sub(out: Tile, src: Tile, scalar: Tile, *, dim: int = 0) -> None: ...
def expand_div(out: Tile, src: Tile, scalar: Tile, *, dim: int = 0) -> None: ...
def gather(
    out: Tile, src: Tile, idx: Tile, tmp: Tile, *, cmp_mode: int = 0, offset: int = 0
) -> None: ...
def gatherb(out: Tile, src: Tile, offsets: Tile) -> None: ...
def gathermask(out: Tile, src: Tile, *, pattern_mode: int) -> None: ...
def scatter(out: Tile, src: Tile, idx: Tile) -> None: ...
def mrgsort(dst: Tile, src: Tile, *, block_len: int) -> None: ...
def mrgsort2(
    src0: Tile, src1: Tile, dst: Tile, tmp: Tile, *args, exhausted: bool = False
) -> None: ...
def sort32(dst: Tile, src: Tile, idx: Tile, tmp: Optional[Tile] = None) -> None: ...
def histogram(dst: Tile, src: Tile, idx: Tile, *, is_msb: bool) -> None: ...
def quant(
    out: Tile,
    src: Tile,
    scale: Tile,
    *,
    mode: QuantMode = QuantMode.SYM,
    offset: Optional[Tile] = None,
) -> None: ...
def dequant(out: Tile, src: Tile, scale: Tile, offset: Tile) -> None: ...
def getval(container: "Tile | Tensor", offset: int) -> Scalar: ...
def setval(container: "Tile | Tensor", offset: int, value: Scalar) -> None: ...
def reinterpret(
    tile: "Tile | TileGroup",
    /,
    *,
    dtype: DType = None,
    shape: List[int] = None,
    layout: Optional[TensorLayout] = None,
) -> "Tile | TileGroup": ...
def set_validshape(tile: "Tile | TileGroup", /, shape: List[int]) -> None: ...
def set_mask_count() -> None: ...
def set_mask_norm() -> None: ...
def set_vec_mask(mask_high: int, mask_low: int) -> None: ...
def set_ctrl_spr(start_bit: int, end_bit: int, value: int, /) -> None: ...
def set_saturation_flag(*, mode: SaturationFlagMode, enable: bool) -> None: ...
def get_ctrl_spr(start_bit: int, end_bit: int, /) -> int: ...
def get_spr() -> int: ...
def get_saturation_flag(*, mode: SaturationFlagMode) -> bool: ...
def reset_ctrl_spr(start_bit: int, end_bit: int, /) -> None: ...
def reset_mask() -> None: ...
def fill_index(out: Tile, start: Scalar) -> None: ...

# Source: python/pypto_pro/language/_vf_api.py
class Vf:
    @staticmethod
    def create_mask(
        pattern: Optional[MaskPattern] = None, dtype: Optional[DType] = None
    ): ...
    @staticmethod
    def update_mask(scalar, dtype: Optional[DType] = None): ...
    @staticmethod
    def full(
        src,
        preg,
        dtype: Optional[DType] = None,
        mode: Optional[MergeMode] = None,
        pos: Optional[DuplicatePos] = None,
    ): ...
    @staticmethod
    def load_align(
        tile,
        offset=None,
        dist: Optional[LoadDist] = None,
        dtype: Optional[DType] = None,
        post_update: bool = False,
        block_stride=None,
        repeat_stride=None,
        data_copy_mode: Optional[DataCopyMode] = None,
    ): ...
    @staticmethod
    def store_align(
        tile,
        src,
        preg,
        offset,
        dist: Optional[StoreDist] = None,
        data_copy_mode: Optional[DataCopyMode] = None,
        block_stride=None,
        repeat_stride=None,
        post_update: bool = False,
    ): ...
    @staticmethod
    def store_unalign(tile, src, align_reg, stride=None, post_update: bool = False): ...
    @staticmethod
    def store_unalign_post(tile, align_reg, stride, post_update: bool = False): ...
    @staticmethod
    def squeeze_store_unalign(tile, src, align_reg): ...
    @staticmethod
    def squeeze_store_unalign_post(tile, align_reg): ...
    @staticmethod
    def unalign_reg_for_store(): ...
    @staticmethod
    def mem_bar(mode: Optional[MemBarMode] = None): ...
    @staticmethod
    def max(src0, src1, preg, mode: Optional[MergeMode] = None): ...
    @staticmethod
    def add(src0, src1, preg, mode: Optional[MergeMode] = None): ...
    @staticmethod
    def sub(src0, src1, preg, mode: Optional[MergeMode] = None): ...
    @staticmethod
    def mul(src0, src1, preg, mode: Optional[MergeMode] = None): ...
    @staticmethod
    def div(
        src0,
        src1,
        preg,
        mode: Optional[MergeMode] = None,
        precision: Optional[bool] = None,
    ): ...
    @staticmethod
    def muls(src, scalar, preg, mode: Optional[MergeMode] = None): ...
    @staticmethod
    def mul_add_dst(src0, src1, preg, mode: Optional[MergeMode] = None): ...
    @staticmethod
    def and_(src0, src1, preg, mode: Optional[MergeMode] = None): ...
    @staticmethod
    def or_(src0, src1, preg, mode: Optional[MergeMode] = None): ...
    @staticmethod
    def xor(
        src0,
        src1,
        preg,
        mode: Optional[MergeMode] = None,
        dtype: Optional[DType] = None,
    ): ...
    @staticmethod
    def reduce_sum(
        src, preg, datablock: bool = False, merge_mode: Optional[MergeMode] = None
    ): ...
    @staticmethod
    def reduce_max(
        src, preg, datablock: bool = False, merge_mode: Optional[MergeMode] = None
    ): ...
    @staticmethod
    def reduce_min(
        src, preg, datablock: bool = False, merge_mode: Optional[MergeMode] = None
    ): ...
    @staticmethod
    def ln(
        src, preg, mode: Optional[MergeMode] = None, precision: Optional[bool] = None
    ): ...
    @staticmethod
    def exp_sub(src0, src1, preg, layout: Optional[CastLayout] = None): ...
    @staticmethod
    def astype(
        src,
        preg,
        dtype: DType,
        layout: Optional[CastLayout] = None,
        round_mode: Optional[VFRoundMode] = None,
        saturate: Optional[SaturateMode] = None,
        mode: Optional[MergeMode] = None,
    ): ...
    @staticmethod
    def de_interleave(src0, src1, dtype: Optional[DType] = None): ...
    @staticmethod
    def select(src0, src1, preg): ...
    @staticmethod
    def shift_left(src, shift, preg, mode: Optional[MergeMode] = None): ...
    @staticmethod
    def shift_right(src, shift, preg, mode: Optional[MergeMode] = None): ...
    @staticmethod
    def histograms(
        src,
        preg,
        bin_type: Optional[BinType] = None,
        hist_type: Optional[HistType] = None,
    ): ...
    @staticmethod
    def eq(src0, src1, preg): ...
    @staticmethod
    def ne(src0, src1, preg): ...
    @staticmethod
    def lt(src0, src1, preg): ...
    @staticmethod
    def gt(src0, src1, preg): ...
    @staticmethod
    def le(src0, src1, preg): ...
    @staticmethod
    def ge(src0, src1, preg): ...
    @staticmethod
    def squeeze(
        src,
        preg,
        gather_mode: Optional[SqueezeMode] = None,
        *,
        dtype: Optional[DType] = None,
    ): ...
    @staticmethod
    def arange(start, dtype: DType, index_order: Optional[IndexOrder] = None): ...
    @staticmethod
    def gather(src, index, preg, data_copy_mode: Optional[DataCopyMode] = None): ...
    @staticmethod
    def clear_spr(): ...
    @staticmethod
    def log(
        src, preg, mode: Optional[MergeMode] = None, precision: Optional[bool] = None
    ): ...
    @staticmethod
    def min(src0, src1, preg, mode: Optional[MergeMode] = None): ...
    @staticmethod
    def exp(
        src, preg, mode: Optional[MergeMode] = None, precision: Optional[bool] = None
    ): ...
    @staticmethod
    def abs(src, preg, mode: Optional[MergeMode] = None): ...
    @staticmethod
    def not_(src, preg, mode: Optional[MergeMode] = None): ...
    @staticmethod
    def sqrt(
        src, preg, mode: Optional[MergeMode] = None, precision: Optional[bool] = None
    ): ...
    @staticmethod
    def relu(src, preg, mode: Optional[MergeMode] = None): ...
    @staticmethod
    def neg(src, preg, mode: Optional[MergeMode] = None): ...
    @staticmethod
    def adds(src, scalar, preg, mode: Optional[MergeMode] = None): ...
    @staticmethod
    def mins(src, scalar, preg, mode: Optional[MergeMode] = None): ...
    @staticmethod
    def maxs(src, scalar, preg, mode: Optional[MergeMode] = None): ...
    @staticmethod
    def leaky_relu(src, scalar, preg, mode: Optional[MergeMode] = None): ...
    @staticmethod
    def interleave(src0, src1, dtype: Optional[DType] = None): ...
    @staticmethod
    def pair_reduce_sum(src, preg, mode: Optional[MergeMode] = None): ...
    @staticmethod
    def abs_sub(src0, src1, preg, mode: Optional[MergeMode] = None): ...
    @staticmethod
    def axpy(src, scalar, preg, mode: Optional[MergeMode] = None): ...
    @staticmethod
    def bit_cast(src, *, dtype: DType): ...
    @staticmethod
    def mul_dst_add(src0, src1, preg, mode: Optional[MergeMode] = None): ...
    @staticmethod
    def pack(src, part: Optional[PackPart] = None, dtype: Optional[DType] = None): ...
    @staticmethod
    def unpack(src, part: Optional[PackPart] = None, dtype: Optional[DType] = None): ...
    @staticmethod
    def prelu(src, slope, preg, mode: Optional[MergeMode] = None): ...
    @staticmethod
    def mull(src0, src1, preg): ...
    @staticmethod
    def addc(src0, src1, carry_src, preg): ...
    @staticmethod
    def subc(src0, src1, borrow_src, preg): ...
    @staticmethod
    def load_unalign_init(): ...
    @staticmethod
    def load_unalign_pre(align_reg, tile): ...
    @staticmethod
    def load_unalign(
        align_reg, tile, stride=None, *, post_update: Optional[bool] = None
    ): ...
    @staticmethod
    def scatter(tile, src, index, preg): ...
    @staticmethod
    def unsqueeze(preg, dtype: Optional[DType] = None): ...
    @staticmethod
    def get_mask_spr(width: MaskWidth = MaskWidth.B32): ...
    @staticmethod
    def log2(
        src, preg, mode: Optional[MergeMode] = None, precision: Optional[bool] = None
    ): ...
    @staticmethod
    def log10(
        src, preg, mode: Optional[MergeMode] = None, precision: Optional[bool] = None
    ): ...
    @staticmethod
    def muls_cast(
        src, scalar, preg, dtype: DType, layout: Optional[CastLayout] = None
    ): ...
    @staticmethod
    def load(tile, stride=None): ...
    @staticmethod
    def store(tile, src, count=None): ...
    @staticmethod
    def truncate(src, preg, mode: Optional[MergeMode] = None): ...
    @staticmethod
    def mask_gen_with_reg_tensor(src, offset: Optional[int] = None): ...
    @staticmethod
    def create_addr_reg(
        stride0, stride1=None, stride2=None, stride3=None, dtype: Optional[DType] = None
    ): ...
    @staticmethod
    def move(src, preg, mode: Optional[MergeMode] = None): ...

# Source: python/pypto_pro/language/_api.py
def pto_assert(
    condition: bool, /, format_str: Optional[str] = None, *args, loc: bool = False
) -> None: ...
def printf(format_str: str, *args, loc: bool = False) -> None: ...
def dump_data(
    data: Union[Tensor, Tile],
    offsets: Optional[List[int]] = None,
    shapes: Optional[List[int]] = None,
    *,
    workspace: Optional[Tensor] = None,
    loc: bool = False,
    flag: Optional[str] = None,
) -> None: ...
def trap() -> None: ...
def min(lhs: Scalar, rhs: Scalar, /) -> Scalar: ...
def max(lhs: Scalar, rhs: Scalar, /) -> Scalar: ...
def const(value: Union[int, float], dtype: DType, /) -> Scalar: ...
def astype(x: Scalar, dtype: DType, /) -> Scalar: ...
def range(start: int, stop: Optional[int] = None, step: int = 1, /): ...
@contextmanager
def section_vector(): ...
@contextmanager
def section_cube(): ...
def get_block_idx() -> int: ...
def get_subblock_idx() -> int: ...
def get_block_num() -> int: ...
def get_subblock_num() -> int: ...
def make_tile_group(
    *,
    type: Any,
    addrs: int | list,
    mutex_ids: str | list[int | list[int] | tuple[int, ...]] | tuple | None = None,
    depth: int | None = None,
    fwd_ids: list[int] | tuple[int, ...] | None = None,
    bwd_ids: list[int] | tuple[int, ...] | None = None,
) -> Any: ...
def bind_crosscore_sync(tensor: Tensor, /, *, ids: "list[int]") -> None: ...

# Source: python/pypto_pro/runtime/jit.py
def jit(
    target=None,
    *,
    arch: str | None = None,
    auto_mutex: bool = True,
    name: str | None = None,
    pipeline=None,
    tiling_key=None,
    datatype=None,
    compile_timeout: int | None = None,
    sanitizer: bool = False,
    **kwargs,
): ...

# Source: python/pypto_pro/runtime/pipeline/__init__.py
class pipeline:
    @dataclass(frozen=True)
    class PipelineConfig:
        preload: int | tuple[int, ...] = 2
        def __post_init__(self) -> None: ...
        def preload_of(self, loop_index: int) -> int: ...

    @staticmethod
    def stage(fn): ...
    @staticmethod
    def transform_pipeline(
        func_def: ast.FunctionDef,
        closure_vars: dict,
        config: PipelineConfig,
        if_const_map: dict | None = None,
        var_types: dict | None = None,
        tilingkey_consts: dict | None = None,
        datatype_consts: dict | None = None,
        expr_types: dict | None = None,
    ) -> ast.FunctionDef: ...

# Source: python/src/bindings/ir/ir.cpp
class MaskPattern:
    ALL: MaskPattern
    ALLF: MaskPattern
    VL1: MaskPattern
    VL2: MaskPattern
    VL3: MaskPattern
    VL4: MaskPattern
    VL8: MaskPattern
    VL16: MaskPattern
    VL32: MaskPattern
    VL64: MaskPattern
    VL128: MaskPattern
    M3: MaskPattern
    M4: MaskPattern
    H: MaskPattern
    Q: MaskPattern

class MergeMode:
    ZEROING: MergeMode
    MERGING: MergeMode

class ReduceMode:
    SUM: ReduceMode
    MAX: ReduceMode
    MIN: ReduceMode

class CompareMode:
    EQ: CompareMode
    NE: CompareMode
    LT: CompareMode
    LE: CompareMode
    GT: CompareMode
    GE: CompareMode

class DuplicatePos:
    LOWEST: DuplicatePos
    HIGHEST: DuplicatePos

class CastLayout:
    ZERO: CastLayout
    ONE: CastLayout
    TWO: CastLayout
    THREE: CastLayout

class VFRoundMode:
    CAST_ROUND: VFRoundMode
    CAST_RINT: VFRoundMode
    CAST_FLOOR: VFRoundMode
    CAST_CEIL: VFRoundMode
    CAST_TRUNC: VFRoundMode
    CAST_ODD: VFRoundMode
    CAST_HYBRID: VFRoundMode

class SaturateMode:
    OFF: SaturateMode
    ON: SaturateMode

class SaturationFlagMode:
    FLOAT: SaturationFlagMode
    FLOAT8: SaturationFlagMode
    INT: SaturationFlagMode
    CAST: SaturationFlagMode

class BinType:
    BIN0: BinType
    BIN1: BinType

class HistType:
    ACCUMULATE: HistType
    FREQUENCY: HistType

class SqueezeMode:
    STORE_REG: SqueezeMode
    NO_STORE_REG: SqueezeMode

class PackPart:
    LOWER: PackPart
    UPPER: PackPart

class MaskWidth:
    B32: MaskWidth
    B16: MaskWidth

class LoadDist:
    NORM: LoadDist
    BRC: LoadDist
    BRC_B8: LoadDist
    BRC_B16: LoadDist
    BRC_B32: LoadDist
    US: LoadDist
    US_B8: LoadDist
    US_B16: LoadDist
    DS: LoadDist
    DS_B8: LoadDist
    DS_B16: LoadDist
    UNPK: LoadDist
    UNPK_B8: LoadDist
    UNPK_B16: LoadDist
    UNPK_B32: LoadDist
    UNPK4: LoadDist
    BLK: LoadDist
    E2B: LoadDist
    E2B_B16: LoadDist
    E2B_B32: LoadDist
    DINTLV_B8: LoadDist
    DINTLV_B16: LoadDist
    DINTLV_B32: LoadDist

class StoreDist:
    NORM: StoreDist
    NORM_B16: StoreDist
    FIRST_ELEMENT: StoreDist
    PACK: StoreDist
    PACK4: StoreDist
    INTLV: StoreDist
    INTLV_B32: StoreDist
    NORM_B8: StoreDist
    NORM_B32: StoreDist
    FIRST_ELEMENT_B8: StoreDist
    FIRST_ELEMENT_B16: StoreDist
    FIRST_ELEMENT_B32: StoreDist
    PACK_B16: StoreDist
    PACK_B32: StoreDist
    PACK_B64: StoreDist
    PACK4_B32: StoreDist
    INTLV_B8: StoreDist
    INTLV_B16: StoreDist

class DataCopyMode:
    NORM: DataCopyMode
    DATA_BLOCK_LOAD: DataCopyMode
    DATA_BLOCK_COPY: DataCopyMode

class MemBarMode:
    VST_VLD: MemBarMode
    VLD_VST: MemBarMode
    VST_VST: MemBarMode
    VST_LD: MemBarMode
    VST_ST: MemBarMode
    VLD_ST: MemBarMode
    ST_VLD: MemBarMode
    ST_VST: MemBarMode
    LD_VST: MemBarMode
    VV_ALL: MemBarMode
    VS_ALL: MemBarMode
    SV_ALL: MemBarMode

class IndexOrder:
    INCREASE_ORDER: IndexOrder
    DECREASE_ORDER: IndexOrder
