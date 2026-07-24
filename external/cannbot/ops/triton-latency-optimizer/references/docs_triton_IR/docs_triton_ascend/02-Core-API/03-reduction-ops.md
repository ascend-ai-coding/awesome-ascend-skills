# 归约操作 API

## 概述

本文档详细描述 Triton-Ascend 中的归约操作 API，包括 `reduce`、`sum`、`max`、`min`、`argmax`、`argmin` 和 `xor_sum`。归约操作将张量沿指定轴的元素合并为单个值（或缩减维度的张量），是并行计算中的核心操作。在 Ascend NPU 上，归约操作由 Vector 计算单元执行，与 GPU 的 CUDA Core 归约对应。NPU 上对窄整数类型（int8/int16）的归约有特殊处理，避免 UB 溢出问题。

关键词：归约, reduce, sum, max, min, argmax, argmin, xor_sum, Vector, NPU, UB 溢出, bfloat16, 精度提升

---

## API 参考

### tl.reduce

通用归约操作，将 `combine_fn` 应用于沿指定轴的所有元素。

```python
triton.language.reduce(
    input,
    axis,
    combine_fn,
    keep_dims=False,
    _builder=None,
    _generator=None
) -> tensor
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `tensor` 或 `tuple(tensor)` | 输入张量，或张量元组（多输出归约） |
| `axis` | `int` 或 `None` | 归约轴。若为 None，归约所有维度 |
| `combine_fn` | `Callable` | 组合函数，必须标记 `@triton.jit`。接受两组标量张量，返回组合结果 |
| `keep_dims` | `bool`，默认 `False` | 是否保留归约维度（长度为 1） |

**返回值**：归约后的张量（或张量元组）

### tl.sum

沿指定轴求和。

```python
triton.language.sum(input, axis=None, keep_dims=False) -> tensor
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `tensor` | 输入张量 |
| `axis` | `int` 或 `None` | 归约轴。若为 None，归约所有维度 |
| `keep_dims` | `bool`，默认 `False` | 是否保留归约维度 |

**实现**：`reduce(input, axis, lambda a, b: a + b)`

**NPU 注意**：bfloat16 输入会自动提升为 float32 执行求和。

### tl.max

沿指定轴求最大值。

```python
triton.language.max(
    input,
    axis=None,
    return_indices=False,
    return_indices_tie_break_left=True,
    keep_dims=False,
    propagate_nan=False
) -> tensor
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `tensor` | 输入张量 |
| `axis` | `int` 或 `None` | 归约轴 |
| `return_indices` | `bool`，默认 `False` | 是否同时返回最大值的索引 |
| `return_indices_tie_break_left` | `bool`，默认 `True` | 当有多个相等最大值时，是否返回最左边的索引 |
| `keep_dims` | `bool`，默认 `False` | 是否保留归约维度 |
| `propagate_nan` | `bool`，默认 `False` | 是否传播 NaN 值 |

**返回值**：若 `return_indices=False`，返回最大值张量；若 `return_indices=True`，返回 `(最大值张量, 索引张量)` 元组。

**NPU 注意**：
- bfloat16 输入会自动提升为 float32
- 窄整数类型（int8/int16）**不会**提升为 int32（避免 UB 溢出），依赖后端直接支持
- 浮点类型（primitive_bitwidth < 32）会提升为 float32

### tl.min

沿指定轴求最小值。参数与 `tl.max` 完全对称。

```python
triton.language.min(
    input,
    axis=None,
    return_indices=False,
    return_indices_tie_break_left=True,
    keep_dims=False
) -> tensor
```

### tl.argmax

沿指定轴返回最大值的索引。

```python
triton.language.argmax(input, axis, tie_break_left=True, keep_dims=False) -> tensor
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `tensor` | 输入张量 |
| `axis` | `int` | 归约轴（必须指定） |
| `tie_break_left` | `bool`，默认 `True` | 当有多个相等最大值时，是否返回最左边的索引 |
| `keep_dims` | `bool`，默认 `False` | 是否保留归约维度 |

**实现**：`max(input, axis, return_indices=True)[1]`

### tl.argmin

沿指定轴返回最小值的索引。参数与 `tl.argmax` 对称。

```python
triton.language.argmin(input, axis, tie_break_left=True, keep_dims=False) -> tensor
```

### tl.xor_sum

沿指定轴求异或和。

```python
triton.language.xor_sum(input, axis=None, keep_dims=False, _builder=None, _generator=None) -> tensor
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `tensor` | 输入张量，**仅支持整数类型** |
| `axis` | `int` 或 `None` | 归约轴 |
| `keep_dims` | `bool`，默认 `False` | 是否保留归约维度 |

**约束**：输入张量的标量类型必须为整数，否则抛出 `ValueError`。

---

## 代码示例

### 基础用法：sum / max / min

```python
import triton
import triton.language as tl

@triton.jit
def reduction_basic_kernel(
    x_ptr, sum_ptr, max_ptr, min_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    s = tl.sum(x, axis=0)
    m = tl.max(x, axis=0)
    n = tl.min(x, axis=0)

    tl.store(sum_ptr + pid, s)
    tl.store(max_ptr + pid, m)
    tl.store(min_ptr + pid, n)
```

### 进阶用法：argmax 与自定义归约

```python
@triton.jit
def argmax_kernel(
    x_ptr, idx_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=float('-inf'))

    idx = tl.argmax(x, axis=0)
    tl.store(idx_ptr + pid, idx)

@triton.jit
def custom_reduce_combine(a, b):
    return a + b * b

@triton.jit
def custom_reduce_kernel(
    x_ptr, out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    result = tl.reduce(x, axis=0, combine_fn=custom_reduce_combine)
    tl.store(out_ptr + pid, result)
```

---

## NPU 适配要点

### 1. Vector 归约实现

NPU 上的归约操作由 Vector 计算单元执行。与 GPU 的 tree-reduction 或 warp-level reduction 不同，NPU Vector 单元使用硬件级归约指令，沿指定轴对 UB 中的数据进行归约。这意味着：

- 归约操作的数据必须先加载到 UB
- 归约的中间结果也在 UB 中
- UB 容量有限，过大的归约范围可能导致 UB 溢出

### 2. 窄整数类型归约的特殊处理

在 GPU 上，int8/int16 类型的归约会先提升为 int32 再执行，以避免溢出。但在 NPU 上，这种提升会消耗大量 UB 内存，可能导致 "UB overflow" 错误。因此，Triton-Ascend **跳过了窄整数类型到 int32 的提升**：

```python
if core.constexpr(input.dtype.primitive_bitwidth) < core.constexpr(32):
    if core.constexpr(input.dtype.is_floating()):
        input = input.to(core.float32)
    else:
        assert input.dtype.is_int(), "Expecting input to be integer type"
        pass  # Do not promote to int32
```

这意味着 int8/int16 的归约结果可能溢出，需要用户自行确保数值范围安全。

### 3. argmax/argmin 的 NaN 处理

在 NPU 上，`argmax`/`argmin` 对 NaN 的处理行为：
- `propagate_nan=False`（默认）：NaN 与任何值的比较结果为 False，NaN 不会被选为最大/最小值
- `propagate_nan=True`：NaN 会被传播，如果输入包含 NaN，结果可能为 NaN

`max` 函数支持 `propagate_nan` 参数，底层使用 `maximum`（传播 NaN）或 `maxnumf`（不传播 NaN）。

### 4. bfloat16 归约精度

bfloat16 输入的归约会自动提升为 float32 执行（通过 `_promote_bfloat16_to_float32`），因为 NPU 不支持 bf16 的 FMAX/FMIN/FCMP 操作。提升为 float32 后的归约精度更高，但需要额外的类型转换开销。

### 5. 归约轴指定

- `axis=None`：归约所有维度，结果为标量
- `axis=int`：归约指定维度
- 支持负数轴索引（如 `axis=-1` 表示最后一维）
- `keep_dims=True` 时，归约维度保留为长度 1

---

## 常见问题

**Q1: 为什么 int8 的 sum 结果可能溢出？**

A: NPU 上窄整数类型的归约不会提升为 int32（避免 UB 溢出），因此 int8 的求和结果仍为 int8，可能溢出。建议在归约前手动提升类型：`tl.sum(x.to(tl.int32), axis=0)`。

**Q2: argmax 在有重复最大值时返回哪个索引？**

A: 默认 `tie_break_left=True`，返回最左边的索引。设置 `tie_break_left=False` 可使用快速模式（不保证返回哪个索引）。

**Q3: 如何实现跨 block 的归约？**

A: Triton 的归约操作仅在单个 program（block）内执行。跨 block 归约需要：每个 block 计算局部归约结果并 store 到全局内存，然后启动第二个 kernel 对局部结果做最终归约。或者使用 atomic 操作实现。

**Q4: xor_sum 支持浮点类型吗？**

A: 不支持。`xor_sum` 仅支持整数类型，传入浮点类型会抛出 `ValueError`。

**Q5: reduce 的 combine_fn 有什么要求？**

A: `combine_fn` 必须标记 `@triton.jit`，接受两个标量参数，返回组合结果。combine_fn 必须满足结合律（associative），但不要求满足交换律（commutative），因为归约的执行顺序由编译器决定。

---

## 相关文档

- [02-math-ops.md](./02-math-ops.md) - 数学运算 API
- [07-scan-sort-ops.md](./07-scan-sort-ops.md) - 扫描与排序操作
- [05-atomic-ops.md](./05-atomic-ops.md) - 原子操作

## 源码参考

- [standard.py - sum/max/min/argmax/argmin/xor_sum 实现](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/standard.py#L136-L302)
- [core.py - reduce 函数定义](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L2047-L2096)
- [core.py - _promote_bfloat16_to_float32](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L2100-L2106)
- [core.py - _reduce_with_indices](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L2109)
