# TileLang-Ascend Programming Guide

## 1. Introduction

TileLang is a tile-level DSL for kernel programming. On Ascend, a practical and stable programming style is to organize computation around:

- L1 as Cube-side staging memory
- UB as Vector-side working memory
- L0A / L0B / L0C as matrix compute buffers

This guide focuses on that programming model and covers:

- kernel definition and launch
- L1 / UB / L0 memory allocation
- data movement with `T.copy`
- matrix compute with `T.mma`
- vector and tile compute with `T.tile.*`
- scope and synchronization control

### 1.1 Programming Guidelines

- Prefer `T.tile.*` APIs for compute whenever possible, and avoid scalar or element-by-element operations in hot paths.
- Use `T.tile.broadcast` sparingly because it can consume large UB temporary space, and prefer row-wise or column-wise tile compute patterns when UB is constrained.
- On the Vector side, in practice, you should copy inputs into UB and then cast them to `float32` at the beginning of `T.Scope("V")`, because this ensures better numerical stability and consistent behavior across the subsequent vector compute path.

> **TileLang Compiler Limitations**
>
> TileLang Ascend compiler is immature. **Do not attempt to fix tail-block or UB-slice related issues** (e.g. dynamic `elems` in slice bounds). **Skip TileLang verification directly** and proceed to AscendC. TileLang is design-only, not a correctness gate.

## 2. Basic Structure

### 2.1 JIT Kernel Definition

Use `@tilelang.jit(...)` to define a kernel generator and `@T.prim_func` to define the device kernel body.

The basic structure is:

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[...])
def kernel(...):
    @T.prim_func
    def main(...):
        with T.Kernel(..., is_npu=True) as (cid, vid):
            ...
    return main
```

### 2.2 Data Types

TileLang Ascend kernels commonly use dtype strings such as:

- `float16`
- `float32`
- `bfloat16`
- `int8`
- `int16`
- `int32`
- `uint8`
- `uint16`
- `uint32`

Example:

```python
dtype = "float16"
accum_dtype = "float32"
```

### 2.3 Kernel Launch

Use `T.Kernel(grid, is_npu=True)` to create an NPU kernel region:

```python
with T.Kernel(block_num, is_npu=True) as (cid, vid):
    ...
```

Typical usage:

- `cid`: block or tile id
- `vid`: Vector-side split id when a kernel has Cube/Vector cooperation

### 2.4 Loops and Control Flow

Use `T.serial` for explicit serial iteration:

```python
for k in T.serial(loop_k):
    if k == 0:
        ...
    else:
        ...
```

Ordinary `if`, `else`, `while`, `break`, and `continue` can also be used in supported contexts.

## 3. On-Chip Memory

Ascend kernels in this guide only use L1, UB, and L0 storage.

### 3.1 `T.alloc_L1`

Allocates an L1 buffer.

```python
A_L1 = T.alloc_L1((block_M, block_K), "float16")
B_L1 = T.alloc_L1((block_K, block_N), "float16")
```

Typical usage:

- load matrix tiles from GM to L1
- stage Cube-side inputs before moving to L0

### 3.2 `T.alloc_ub`

Allocates a UB buffer.

```python
x_ub = T.alloc_ub((block_M, block_N), "float16")
tmp_ub = T.alloc_ub((tmp_size,), "uint8")
```

Typical usage:

- load Vector-side tiles from GM or workspace
- hold intermediate tile results
- serve as temporary storage for `T.tile.*`

### 3.3 `T.alloc_L0A`

Allocates an L0A buffer for the left matrix tile.

```python
A_L0 = T.alloc_L0A((block_M, block_K), "float16")
```

### 3.4 `T.alloc_L0B`

Allocates an L0B buffer for the right matrix tile.

```python
B_L0 = T.alloc_L0B((block_K, block_N), "float16")
```

### 3.5 `T.alloc_L0C`

Allocates an L0C buffer for accumulation or matrix output tiles.

```python
C_L0 = T.alloc_L0C((block_M, block_N), "float32")
```

Typical usage:

- hold the accumulation tile for `T.mma`
- store Cube-side output before writing back

## 4. Data Movement

### 4.1 `T.copy`

`T.copy(src, dst)` is the primary movement primitive.

Common patterns:

- GM -> L1
- L1 -> L0A
- L1 -> L0B
- L0C -> GM
- GM -> UB
- UB -> GM
- UB -> UB

Examples:

```python
T.copy(A[bx * block_M, k * block_K], A_L1)
T.copy(B[k * block_K, by * block_N], B_L1)
T.copy(A_L1, A_L0)
T.copy(B_L1, B_L0)
T.copy(C_L0, C[bx * block_M, by * block_N])
```

### 4.2 Layout Annotation

When L1 tiles feed `T.mma`, layout annotation is commonly required.

```python
from tilelang.intrinsics import make_zn_layout

T.annotate_layout({
    A_L1: make_zn_layout(A_L1),
    B_L1: make_zn_layout(B_L1),
})
```

This is typically applied to L1 matrix tiles before they are copied into L0A and L0B.

## 5. Matrix Compute with `T.mma`

`T.mma` is the matrix intrinsic used in this guide.

```python
T.mma(A_L0, B_L0, C_L0, init=...)
```

Requirements:

- `A_L0` is allocated in L0A
- `B_L0` is allocated in L0B
- `C_L0` is allocated in L0C

Semantics:

- `init=True` initializes the accumulation tile
- `init=False` accumulates onto the existing tile

Example:

```python
T.mma(
    A_L0,
    B_L0,
    C_L0,
    init=T.And(k == 0, kk == 0),
)
```

## 6. Tile Compute with `T.tile.*`

`T.tile.*` APIs are UB-side tile operations for vectorized elementwise, logical, transform, and utility work. This section is aligned to the upstream `tilelang-ascend` guide for API names and signatures.

### 6.1 Math and Logical Ops

| Category | API | Semantics |
| --- | --- | --- |
| Add | `T.tile.add(dst, src0, src1)` | elementwise add, `dst = src0 + src1` |
| Subtract | `T.tile.sub(dst, src0, src1)` | elementwise subtract, `dst = src0 - src1` |
| Multiply | `T.tile.mul(dst, src0, src1)` | elementwise multiply, `dst = src0 * src1` |
| Divide | `T.tile.div(dst, src0, src1)` | elementwise divide, `dst = src0 / src1` |
| Maximum | `T.tile.max(dst, src0, src1)` | elementwise max, `dst = max(src0, src1)` |
| Minimum | `T.tile.min(dst, src0, src1)` | elementwise min, `dst = min(src0, src1)` |
| Exponential | `T.tile.exp(dst, src0)` | elementwise natural exponent, `dst = exp(src0)` |
| Natural Log | `T.tile.ln(dst, src0)` | elementwise natural log, `dst = ln(src0)` |
| Absolute Value | `T.tile.abs(dst, src0)` | elementwise absolute value, `dst = abs(src0)` |
| Reciprocal | `T.tile.reciprocal(dst, src0)` | elementwise reciprocal, `dst = 1 / src0` |
| Square Root | `T.tile.sqrt(dst, src0)` | elementwise square root, `dst = sqrt(src0)` |
| Reciprocal Square Root | `T.tile.rsqrt(dst, src0)` | elementwise reciprocal square root, `dst = 1 / sqrt(src0)` |
| ReLU | `T.tile.relu(dst, src0)` | elementwise ReLU, `dst = max(0, src0)` |
| Leaky ReLU | `T.tile.leaky_relu(dst, src0, scalar)` | elementwise Leaky ReLU, `dst = src0 if src0 >= 0 else src0 * scalar` |
| AXPY | `T.tile.axpy(dst, src0, scalar)` | fused axpy, `dst = scalar * src0 + dst` |
| Sine | `T.tile.sin(dst, src0, tmp)` | elementwise sine, `dst = sin(src0)` |
| Cosine | `T.tile.cos(dst, src0, tmp)` | elementwise cosine, `dst = cos(src0)` |
| Bitwise AND | `T.tile.bitwise_and(dst, src0, src1)` | elementwise bitwise AND, `dst = src0 & src1` |
| Bitwise OR | `T.tile.bitwise_or(dst, src0, src1)` | elementwise bitwise OR, `dst = src0 \| src1` |
| Bitwise NOT | `T.tile.bitwise_not(dst, src0)` | elementwise bitwise NOT, `dst = ~src0` |
| Bitwise XOR | `T.tile.bitwise_xor(dst, src0, src1)` | elementwise bitwise XOR, `dst = src0 ^ src1` |
| Left Shift | `T.tile.bitwise_lshift(dst, src0, scalar)` | elementwise left shift |
| Right Shift | `T.tile.bitwise_rshift(dst, src0, scalar)` | elementwise right shift |

Examples:

```python
T.tile.add(c_ub, a_ub, b_ub)
T.tile.ln(c_ub, a_ub)
T.tile.bitwise_and(c_ub, a_ub, b_ub)
```

Scalar usage note for `T.tile.mul`/`T.tile.add`/`T.tile.sub`/`T.tile.div` and similar ops:

- `src1` can be a scalar.
- When `src1` is a scalar, its dtype must match `src0` dtype.
- Use explicit constructors such as `T.float32(src1)` or `T.bfloat16(src1)` before passing the scalar.

Example:

```python
alpha = T.float32(1.0 / 127.0)
T.tile.mul(scale_ub, row_max_ub, alpha)
beta = T.bfloat16(0.5)
T.tile.mul(x_ub, x_ub, beta)
```

### 6.2 Compare and Select

#### `T.tile.compare(dst, src0, src1, mode)`

Elementwise comparison. The output bit is set when the comparison is true, otherwise cleared.

Supported `mode` values:

- `"EQ"`
- `"NE"`
- `"GT"`
- `"GE"`
- `"LT"`
- `"LE"`

Examples:

```python
T.tile.compare(c_ub, a_ub, b_ub, "EQ")
T.tile.compare(c_ub, a_ub, 1.0, "EQ")
```

#### `T.tile.select(dst, selMask, src0, src1, selMode)`

Selects elements from `src0` or `src1` according to `selMask`.

Supported `selMode` values:

- `"VSEL_CMPMASK_SPR"`
- `"VSEL_TENSOR_SCALAR_MODE"`
- `"VSEL_TENSOR_TENSOR_MODE"`

Examples:

```python
T.tile.select(c_ub, selmask_ub, a_ub, b_ub, "VSEL_CMPMASK_SPR")
T.tile.select(c_ub, selmask_ub, a_ub, 1.0, "VSEL_TENSOR_SCALAR_MODE")
```

#### `T.tile.gather_mask(dst, src, src1Pattern)`

Collects elements according to a fixed string pattern or a custom index buffer.

Common fixed patterns:

- `"P0101"`
- `"P1010"`
- `"P0001"`
- `"P0010"`
- `"P0100"`
- `"P1000"`
- `"P1111"`

Example:

```python
T.tile.gather_mask(b_ub, a_ub, "P0101")
```

### 6.3 Cast and Data Transform

#### `T.tile.cast(dst, src, mode, count)`

Converts data type from `src` to `dst`.

Common modes:

- `"CAST_NONE"`
- `"CAST_RINT"`
- `"CAST_FLOOR"`
- `"CAST_CEIL"`
- `"CAST_ROUND"`
- `"CAST_TRUNC"`
- `"CAST_ODD"`

Cast 组合与 `roundMode` 支持关系如下：

| src数据类型 | dst数据类型 | 支持的roundMode |
| --- | --- | --- |
| half | float | CAST_NONE |
| half | int32_t | CAST_RINT / CAST_FLOOR / CAST_CEIL / CAST_ROUND / CAST_TRUNC |
| half | int8_t | CAST_FLOOR / CAST_CEIL / CAST_ROUND / CAST_TRUNC / CAST_NONE |
| half | uint8_t | CAST_FLOOR / CAST_CEIL / CAST_ROUND / CAST_TRUNC / CAST_NONE |
| float | half | CAST_NONE / CAST_ODD |
| float | int32_t | CAST_RINT / CAST_FLOOR / CAST_CEIL / CAST_ROUND / CAST_TRUNC |
| uint8_t | half | CAST_NONE |
| int8_t | half | CAST_NONE |
| int32_t | float | CAST_NONE |

Example:

```python
T.tile.cast(c_scale, c_ub, mode="CAST_NONE", count=block_M_2 * block_N)
T.tile.cast(c_out, c_scale, mode="CAST_RINT", count=block_M_2 * block_N)
```

#### `T.tile.transpose(dst, src)`

Data transform API. Current upstream documentation notes support for `16 x 16` 2D matrix-tile transpose.

Example:

```python
T.tile.transpose(b_ub, a_ub)
```

### 6.4 Fill and Index Helpers

#### `T.tile.fill(buffer, value)`

Fills a buffer with a scalar value.

```python
T.tile.fill(scale_ub, scale_ub[0])
```

#### `T.tile.createvecindex(dst, first_value)`

Creates a vector index sequence starting from `first_value`.

```python
T.tile.createvecindex(c_ub, 0)
```

#### `T.tile.arith_progression(buffer, first_value, diff_value, count)`

Creates an arithmetic progression with the given start value, stride, and count.

```python
T.tile.arith_progression(sort_indices, 0, 1, block_N)
```

### 6.5 Sort and Gather

#### `T.tile.sort(dst, src, indices, tmp_buffer, repeat_time)`

Sorts values in descending order and writes original indices to `indices`.

```python
T.tile.sort(dst, src, indices, tmp_buffer, repeat_time)
```

#### `T.tile.merge_sort(dst, src, block_size, block_num, is_copy)`

Merges up to four already-sorted queues into one queue.

```python
T.tile.merge_sort(dst, src, block_size, block_num, 0)
```

#### `T.tile.topk(dst, src, tmp_buffer, block_size)`

Gets top-k values or indices from the last dimension.

```python
T.tile.topk(topk_global, sort_result, sort_temp, top_k)
```

#### `T.tile.gather(dst, src, src_offset, src_base_addr)`

Gathers elements according to per-element offsets plus an optional base address offset.

```python
T.tile.gather(c_ub, a_ub, b_ub, 0)
```

### 6.6 Broadcast

#### `T.tile.broadcast(dst, src, tmp)`

Broadcast requires additional UB and is prone to errors. Although it can be faster, please avoid using it unless you are specifically optimizing for performance.

The source and destination must have the same number of dimensions.

```python
a_ub = T.alloc_ub((1, N), dtype)
b_ub = T.alloc_ub((sub_block_M, N), dtype)
tmp = T.alloc_ub((2 * sub_block_M, N), "uint8")
T.tile.broadcast(b_ub, a_ub, tmp)
```

### 6.7 Reduce

#### `T.reduce_sum(buffer, out, tmp, dim)`

Reduce temporary buffer requirements (`T.reduce_sum`, `T.reduce_max`, `T.reduce_min`):

- `tmp` must be allocated in UB with dtype `uint8`.
- `tmp` size must be `2x` of `buffer` element count.

```python
tmp_ub = T.alloc_ub((2 * buffer_size,), "uint8")
T.reduce_sum(sum_square_ub, rms_ub, tmp_ub, dim=-1)
```

#### `T.reduce_max(buffer, out, tmp, dim)`

```python
tmp_ub = T.alloc_ub((2 * buffer_size,), "uint8")
T.reduce_max(a, tile_max, tmp_ub, dim=-1)
```

#### `T.reduce_min(buffer, out, tmp, dim)`

```python
tmp_ub = T.alloc_ub((2 * buffer_size,), "uint8")
T.reduce_min(a_ub, b_ub, tmp_ub, dim=-1)
```

## 7. Scope and Synchronization

This programming model supports explicit control over Cube-side and Vector-side execution, synchronization, and handoff.

### 7.1 Scope Control

#### `T.Scope("C")`

Use `T.Scope("C")` for Cube-side logic:

- GM/L1/L0 movement for matrix tiles
- `T.mma`
- writing Cube results to GM or workspace

#### `T.Scope("V")`

Use `T.Scope("V")` for Vector-side logic:

- loading tiles into UB
- `T.tile.*` operations
- writing final UB tiles back to GM

Pattern:

```python
with T.Scope("C"):
    ...

with T.Scope("V"):
    ...
```

### 7.2 Pipe Flags

#### `T.set_flag(src, dst, id)`

Sets a hardware event flag.

```python
T.set_flag("mte2", "m", 0)
```

#### `T.wait_flag(src, dst, id)`

Waits for a hardware event flag.

```python
T.wait_flag("mte2", "m", 0)
```

Common pipe names:

- `mte1`
- `mte2`
- `mte3`
- `m`
- `v`
- `fix`

### 7.3 Cross-Scope Synchronization

Use cross flags when Cube and Vector scopes communicate through workspace or GM.

#### `T.set_cross_flag(pipe, id)`

```python
T.set_cross_flag("FIX", 0)
```

#### `T.wait_cross_flag(id)`

```python
T.wait_cross_flag(0)
```

Typical pattern:

```python
with T.Scope("C"):
    T.copy(C_L0, workspace[...])
    T.set_cross_flag("FIX", 0)

with T.Scope("V"):
    T.wait_cross_flag(0)
    T.copy(workspace[...], x_ub)
```

### 7.4 Pipe Barrier

Use `T.pipe_barrier(pipe)` when a full pipe barrier is required.

```python
T.pipe_barrier("m")
T.pipe_barrier("v")
```

### 7.5 Auto Sync

When intra-kernel pipe synchronization should be inserted automatically, enable:

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}
```

This is especially useful when the kernel structure is correct and you want the compiler to infer pipe-level synchronization between data movement and compute stages.
